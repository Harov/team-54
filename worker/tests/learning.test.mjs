import { test } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const bundled = await build({ entryPoints: ['src/learning.ts'], bundle: true, platform: 'node', format: 'esm', write: false });
const { parseLearningScope, normalizeMarketRows, marketFeatures, historicalExamples, fitRidge, predictReturn, evaluateChronologically, chronologicalSplit, selectFastModel, fitChoice, evaluatePaperPositions, resolveForecast, runLearning, getLearning } = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);

test('saved model reads use one collection and return the saved recent history', async () => {
  const snapshot = { ledger: [{ id: 'saved-forecast' }] };
  let reads = 0;
  const db = { collection(name) {
    assert.equal(name, 'learning_state');
    return { async findOne() { reads++; return { snapshot }; } };
  } };
  assert.deepEqual((await getLearning(db, { tickers: ['AAPL'], horizon: '1y' })).ledger, snapshot.ledger);
  assert.equal(reads, 1);
});

function history(count, ticker = 'AAPL', start = '2020-01-02') {
  const result = [], day = new Date(`${start}T21:00:00.000Z`);
  for (let i = 0; result.length < count; i++, day.setUTCDate(day.getUTCDate() + 1)) {
    if ([0, 6].includes(day.getUTCDay())) continue;
    const n = result.length;
    result.push({ ticker, observed_at: day.toISOString(), close: 100 + n * 0.12 + Math.sin(n / 5) * 2, volume: 10000 + (n % 17) * 100, source: 'test:actual' });
  }
  return result;
}
function pendingForecast(overrides = {}) {
  return { id: 'p1', scope_key: 'scope', model_version: 'v1', model_run_id: 'run1', ticker: 'AAPL', horizon: '1d', market_days: 1, as_of: '2026-01-05T21:00:00.000Z', feature_as_of: '2026-01-05T21:00:00.000Z', issued_at: '2026-01-09T22:00:00.000Z', base_price: 100, predicted_return: 2, predicted_price: 102, features: [1, 2, 3, 4], weights: [0, 0, 0, 0], bias: 2, contributions: [0, 0, 0, 0], status: 'pending', used_for_training: false, ...overrides };
}

// A small Mongo contract fake: $setOnInsert is distinct from $set, updates are
// atomic before their Promise resolves, and every query applies its full filter.
const copy = value => structuredClone(value);
function matches(doc, query) {
  return Object.entries(query).every(([key, expected]) => {
    const actual = doc[key];
    if (expected && typeof expected === 'object' && !(expected instanceof Date)) {
      return Object.entries(expected).every(([op, value]) => op === '$in' ? value.includes(actual) : op === '$lte' ? actual <= value : op === '$lt' ? actual < value : op === '$gt' ? actual > value : op === '$gte' ? actual >= value : false);
    }
    return actual instanceof Date && expected instanceof Date ? +actual === +expected : actual === expected;
  });
}
class MemoryCollection {
  rows = new Map(); writes = [];
  async createIndex() { return 'index'; }
  find(query = {}) {
    let sort = {}, limit = Infinity;
    const cursor = { sort(value) { sort = value; return cursor; }, limit(value) { limit = value; return cursor; }, toArray: async () => [...this.rows.values()].filter(row => matches(row, query)).sort((a, b) => {
      for (const [key, direction] of Object.entries(sort)) { if (a[key] < b[key]) return -direction; if (a[key] > b[key]) return direction; }
      return 0;
    }).slice(0, limit).map(copy) };
    return cursor;
  }
  async findOne(query) { return copy([...this.rows.values()].find(row => matches(row, query)) ?? null); }
  async updateOne(query, update, options = {}) {
    let row = [...this.rows.values()].find(row => matches(row, query));
    let upsertedCount = 0;
    if (!row && options.upsert) {
      row = copy({ ...query, ...(update.$setOnInsert ?? {}) });
      if (this.rows.has(row._id)) throw Object.assign(new Error('Duplicate _id'), { code: 11000 });
      this.rows.set(row._id, row); upsertedCount = 1;
    }
    if (!row) return { upsertedCount: 0, modifiedCount: 0 };
    if (update.$set) Object.assign(row, copy(update.$set));
    for (const [key, value] of Object.entries(update.$inc ?? {})) row[key] = (row[key] ?? 0) + value;
    this.writes.push(copy({ query, update }));
    return { upsertedCount, modifiedCount: 1 };
  }
  async findOneAndUpdate(query, update) {
    const row = [...this.rows.values()].find(row => matches(row, query));
    if (!row) return null;
    await this.updateOne({ _id: row._id }, update);
    return copy(row);
  }
  async bulkWrite(writes) { let upsertedCount = 0; for (const write of writes) { const { filter, update, upsert } = write.updateOne; upsertedCount += (await this.updateOne(filter, update, { upsert })).upsertedCount; } return { upsertedCount }; }
  async updateMany(query, update) { for (const row of [...this.rows.values()]) if (matches(row, query)) await this.updateOne({ _id: row._id }, update); }
}
class MemoryDb {
  collections = new Map();
  collection(name) { if (!this.collections.has(name)) this.collections.set(name, new MemoryCollection()); return this.collections.get(name); }
  addPrices(rows) { for (const row of rows) this.collection('market').rows.set(`${row.ticker}:${row.observed_at}`, copy({ _id: `${row.ticker}:${row.observed_at}`, ...row })); }
}

test('learning scopes are canonical and unsupported symbols/horizons are rejected', () => {
  assert.deepEqual(parseLearningScope({ tickers: ['msft', ' AAPL ', 'aapl'], horizon: '20d' }), { tickers: ['AAPL', 'MSFT'], horizon: '20d' });
  for (const input of [{ tickers: [] }, { tickers: Array(9).fill('AAPL') }, { tickers: ['../admin'] }, { horizon: 'toString' }, { horizon: '2d' }]) assert.throws(() => parseLearningScope(input));
});

test('market normalization deduplicates daily prices and excludes future and invalid records', () => {
  const source = history(25), last = source.at(-1);
  const rows = normalizeMarketRows([...source, { ...last, source: 'z-correction', close: 123 }, { ...last, observed_at: '2099-01-01', close: 900 }, { ...last, observed_at: '2020-02-30', close: 900 }, { ...last, ticker: 'MSFT' }, { ...last, observed_at: '2010-01-01', close: -1 }], 'AAPL', '2026-01-01T00:00:00.000Z');
  assert.equal(rows.length, 25);
  assert.equal(rows.at(-1).close, 123);
  assert.deepEqual(normalizeMarketRows([...source, { ...last, source: 'z-correction', close: 123 }].reverse(), 'AAPL', '2026-01-01T00:00:00.000Z'), rows);
  assert.equal(normalizeMarketRows([{ ...last, observed_at: '2026-01-01' }], 'AAPL', '2026-01-01T20:59:00.000Z').length, 0);
});

test('all four features use only the current and preceding twenty sessions', () => {
  const rows = history(50), before = marketFeatures(rows, 20);
  rows[21].close *= 99; rows[21].volume *= 99;
  assert.deepEqual(marketFeatures(rows, 20), before);
  assert.equal(marketFeatures(rows, 19), null);
  assert.equal(before[0], (rows[20].close / rows[19].close - 1) * 100);
  assert.equal(before[3], (rows[20].close / rows[0].close - 1) * 100);
});

test('ridge fits a real numeric relation, remains deterministic, and exposes exact contributions', () => {
  const examples = Array.from({ length: 100 }, (_, i) => ({ features: [i / 10, 0, 0, 0], actual_return: 3 + 2 * i / 10 }));
  const model = fitRidge(examples);
  assert.deepEqual(model, fitRidge(examples));
  assert.ok(Math.abs(model.weights[0] - 2) < 0.2);
  assert.ok(Math.abs(predictReturn(model, [5, 0, 0, 0]) - 13) < 0.1);
  assert.deepEqual(fitRidge(Array.from({ length: 40 }, () => ({ features: [0, 0, 0, 0], actual_return: 3 }))), { weights: [0, 0, 0, 0], bias: 3 });
});

test('chronological evaluation purges overlapping future labels before fitting and needs twenty test outcomes', () => {
  const examples = historicalExamples(history(250), 20), validation = evaluateChronologically(examples);
  assert.equal(validation.status, 'supported');
  assert.ok(validation.train_target_through <= validation.test_feature_from);
  const train = examples.filter(row => row.feature_as_of < validation.test_feature_from && row.target_observed_at <= validation.test_feature_from);
  const testRows = examples.filter(row => row.feature_as_of >= validation.test_feature_from);
  assert.ok(train.length < examples.length - testRows.length, 'overlapping labels were purged');
  const expected = testRows.reduce((sum, row) => sum + Math.abs(predictReturn(fitRidge(train), row.features) - row.actual_return), 0) / testRows.length;
  assert.equal(validation.mae_pct, expected);
  const changed = examples.map(row => row.target_observed_at > validation.test_feature_from && row.feature_as_of < validation.test_feature_from ? { ...row, actual_return: 100000 } : row);
  assert.deepEqual(evaluateChronologically(changed), validation, 'embargoed outcomes cannot affect the fitted test model');
  assert.equal(evaluateChronologically(examples.slice(0, 19)).status, 'insufficient_data');
});

test('five-year evaluation refuses a history whose train labels were still in the future', () => {
  const validation = evaluateChronologically(historicalExamples(history(1900), 1260));
  assert.equal(validation.status, 'insufficient_data');
  assert.equal(validation.train_samples, 0);
  assert.equal(validation.directional_accuracy, null);
  assert.equal(validation.mae_pct, null);
  const mature = evaluateChronologically(historicalExamples(normalizeMarketRows(history(3000), 'AAPL', '2040-01-01T00:00:00.000Z', 3000), 1260));
  assert.equal(mature.status, 'supported', 'the bounded five-year window can eventually contain enough mature validation data');
  assert.ok(mature.train_target_through <= mature.test_feature_from);
});

test('a stale-as-of live forecast only resolves from distinct market sessions strictly after actual issue', () => {
  const forecast = pendingForecast(), rows = history(10, 'AAPL', '2026-01-05').map((row, i) => ({ ...row, close: 100 + i }));
  const sameIssueDay = '2026-01-09T23:00:00.000Z';
  assert.equal(resolveForecast(forecast, rows, sameIssueDay), null, 'historical sessions after as-of but before issue cannot score it');
  const result = resolveForecast(forecast, rows, '2026-01-13T22:00:00.000Z');
  assert.equal(result.target_observed_at, '2026-01-12T21:00:00.000Z');
  assert.equal(result.actual_price, 105);
  assert.ok(Math.abs(result.actual_return - 5) < 1e-10);
  assert.equal(resolveForecast({ ...forecast, horizon: '5d', market_days: 5 }, [...rows.slice(0, 8), ...rows.slice(0, 8)], '2026-01-14T22:00:00.000Z'), null, 'duplicates do not advance the horizon');
  const fifth = resolveForecast({ ...forecast, horizon: '5d', market_days: 5 }, [...rows, ...rows], '2026-01-17T22:00:00.000Z');
  assert.equal(fifth.target_observed_at, '2026-01-16T21:00:00.000Z');
  assert.equal(forecast.status, 'pending', 'resolution does not mutate the immutable issued forecast');
});

test('Mongo workflow saves immutable forecasts, caches repeats, and learns only resolved actual returns', async t => {
  const rows = history(180), db = new MemoryDb(), scope = { tickers: ['AAPL'], horizon: '1d' };
  t.mock.timers.enable({ apis: ['Date'], now: Date.parse(rows.at(-1).observed_at) + 3600000 });
  db.addPrices(rows);
  const first = await runLearning(db, scope);
  assert.equal(first.status, 'ready'); assert.equal(first.new_predictions, 1);
  const issued = copy(first.predictions[0]);
  assert.equal(issued.predicted_return, issued.bias + issued.contributions.reduce((sum, value) => sum + value, 0));
  assert.equal(issued.base_price, rows.at(-1).close);
  assert.equal(issued.issued_at, new Date().toISOString());
  assert.equal(issued.status, 'pending'); assert.equal(issued.used_for_training, false);
  assert.equal(first.resolved_predictions, 0);
  assert.ok(!('probability_up' in issued)); assert.ok(!('confidence' in issued));
  const repeat = await runLearning(db, { tickers: ['aapl'], horizon: '1d' });
  assert.equal(repeat.cached, true); assert.equal(repeat.new_predictions, 0);
  assert.deepEqual(repeat.predictions[0], issued);
  assert.equal([...db.collection('learning_budget').rows.values()][0].runs, 1);
  assert.equal(db.collection('learning_runs').rows.size, 1);
  assert.equal(db.collection('learning_predictions').rows.size, 1);

  const next = history(181).at(-1);
  t.mock.timers.tick(Date.parse(next.observed_at) + 3600000 - Date.now());
  db.addPrices([next]);
  const after = await runLearning(db, scope);
  assert.equal(after.resolved_predictions, 1); assert.equal(after.learned_predictions, 1);
  const learned = after.ledger.find(row => row.id === issued.id);
  assert.equal(learned.status, 'resolved'); assert.equal(learned.used_for_training, true);
  assert.equal(learned.actual_return, (next.close / issued.base_price - 1) * 100);
  assert.equal(learned.predicted_return, issued.predicted_return);
  assert.equal(learned.issued_at, issued.issued_at); assert.deepEqual(learned.features, issued.features);
  assert.deepEqual(learned.weights, issued.weights); assert.deepEqual(learned.contributions, issued.contributions);
  assert.equal(learned.learned_model_id, after.model_run_id);
  const expectedModel = fitChoice(historicalExamples([...rows, next], 1), after.model_selection.selected);
  assert.deepEqual(after.weights, expectedModel.weights, 'historical/ledger duplicate learns one real target, not the prior prediction');
  assert.equal(after.bias, expectedModel.bias);
  assert.equal(after.predictions[0].status, 'pending');
  assert.equal((await getLearning(db, scope)).ledger.length, 2);
  assert.equal(await getLearning(db, { tickers: ['MSFT'], horizon: '1d' }), null);
});

test('shared budget rejects the thirteenth changed scope and a scope lease prevents concurrent fitting', async t => {
  const db = new MemoryDb(), rows = history(150);
  t.mock.timers.enable({ apis: ['Date'], now: Date.parse(rows.at(-1).observed_at) + 3600000 });
  for (const ticker of ['AAPL', 'MSFT', 'NVDA']) db.addPrices(rows.map(row => ({ ...row, ticker })));
  for (const ticker of ['AAPL', 'MSFT']) for (const horizon of ['1d', '5d', '20d', '1y', '3y', '5y']) await runLearning(db, { tickers: [ticker], horizon });
  await assert.rejects(runLearning(db, { tickers: ['NVDA'], horizon: '1d' }), error => error.status === 429);
  assert.equal([...db.collection('learning_budget').rows.values()][0].runs, 12);
  const scope = { tickers: ['AAPL'], horizon: '1d' }, saved = await getLearning(db, scope);
  const exhausted = await runLearning(db, scope);
  assert.equal(exhausted.cached, true); assert.equal(exhausted.new_predictions, 0);
  assert.ok(exhausted.notes.at(-1).includes('allowance'));
  t.mock.timers.tick(3600000);
  await db.collection('learning_locks').updateOne({ _id: saved.scope_key }, { $set: { expires_at: new Date(Date.now() + 60000), lease_id: 'other-run' } });
  await assert.rejects(runLearning(db, scope), error => error.status === 429 && error.message.includes('already running'));
});

test('cold reads use twelve seconds with two concurrent cursors; cleanup cannot mask a saved snapshot', async t => {
  const db = new MemoryDb(), rows = history(150), tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN'];
  t.mock.timers.enable({ apis: ['Date'], now: Date.parse(rows.at(-1).observed_at) + 3600000 });
  for (const ticker of tickers) db.addPrices(rows.map(row => ({ ...row, ticker })));
  const market = db.collection('market'), originalFind = market.find.bind(market);
  let active = 0, maximum = 0;
  market.find = (query, options) => {
    assert.equal(options.timeoutMS, 12000);
    const cursor = originalFind(query), read = cursor.toArray;
    cursor.toArray = async () => {
      active++; maximum = Math.max(maximum, active);
      try { await new Promise(resolve => setImmediate(resolve)); return await read(); }
      finally { active--; }
    };
    return cursor;
  };
  const locks = db.collection('learning_locks'), update = locks.updateOne.bind(locks);
  locks.updateOne = async (query, change, options) => {
    if (query.lease_id && change.$set?.expires_at?.getTime() === 0) throw new Error('Cleanup socket read timed out');
    return update(query, change, options);
  };
  const result = await runLearning(db, { tickers, horizon: '1d' });
  assert.equal(result.status, 'ready'); assert.equal(result.new_predictions, 4);
  assert.equal(maximum, 2);
  assert.equal(db.collection('learning_state').rows.size, 1);
  assert.equal(db.collection('learning_predictions').rows.size, 4);
  const failing = new MemoryDb(); let attempts = 0;
  failing.collection('learning_budget').findOne = async () => { attempts++; throw new Error('Timed out during socket read (12000ms)'); };
  await assert.rejects(runLearning(failing, { tickers: ['AAPL'], horizon: '1d' }), error => error.status === 503 && error.message.includes('allowance check') && error.message.includes('12000ms'));
  assert.equal(attempts, 1, 'a failed run is not blindly retried');
});

test('candidate selection cannot see the final test labels and purges inner overlapping outcomes', () => {
  const examples = historicalExamples(history(500),5);
  const split = chronologicalSplit(examples);
  const selection = selectFastModel(split.train);
  assert.equal(selection.status,'compared');
  assert.equal(selection.candidates.length,7);
  assert.ok(selection.train_target_through <= selection.validation_feature_from);
  const changed = examples.map(row => row.feature_as_of >= split.boundary ? {...row,actual_return:98765} : row);
  const second = selectFastModel(chronologicalSplit(changed).train);
  assert.deepEqual(second.selected,selection.selected);
  assert.deepEqual(second.candidates,selection.candidates);
  const validation = evaluateChronologically(examples,undefined,selection.selected);
  const expectedZero = split.test.reduce((sum,row) => sum + Math.abs(row.actual_return),0) / split.test.length;
  assert.equal(validation.zero_return_mae_pct,expectedZero);
  assert.equal(validation.skill_vs_zero,1 - validation.mae_pct / expectedZero);
  assert.equal(fitChoice(examples,{id:'mean',label:'Mean',feature_count:0,lambda:null}).weights.every(weight => weight === 0),true);
  const priceOnly = fitChoice(examples,{id:'price',label:'Price',feature_count:4,lambda:1});
  assert.deepEqual(priceOnly.weights.slice(4),[0,0,0,0]);
});

test('social inputs exclude same-day and later posts, and do not use later engagement totals', () => {
  const rows = history(40), date = rows[20].observed_at.slice(0,10);
  const prior = new Date(Date.parse(date)-86400000).toISOString().slice(0,10);
  const social = new Map([[prior,{mood_sum:1,mentions:2,engagement:100000}],[date,{mood_sum:500,mentions:500,engagement:500}]]);
  const features = marketFeatures(rows,20,social);
  assert.equal(features[4],0.5);
  assert.equal(features[7],Math.log1p(2));
  social.set(date,{mood_sum:-80000,mentions:90000,engagement:1000000});
  assert.deepEqual(marketFeatures(rows,20,social),features);
});

test('paper long/cash evaluation does not compound overlapping positions and subtracts round-trip costs', () => {
  const rows = [
    {ticker:'AAPL',feature_as_of:'2020-01-01',target_observed_at:'2020-01-06',features:[1],actual_return:10},
    {ticker:'AAPL',feature_as_of:'2020-01-02',target_observed_at:'2020-01-07',features:[1],actual_return:90},
    {ticker:'AAPL',feature_as_of:'2020-01-06',target_observed_at:'2020-01-11',features:[1],actual_return:-5},
  ];
  const result = evaluatePaperPositions(rows,{weights:[0],bias:1});
  assert.equal(result.trades,2);
  assert.equal(result.evaluated_windows,2);
  assert.ok(Math.abs(result.return_pct - ((1.099 * 0.949)-1)*100) < 1e-10);
  assert.ok(Math.abs(result.max_drawdown_pct - 5.1) < 1e-10);
  assert.equal(evaluatePaperPositions(rows,{weights:[0],bias:0}).return_pct,0);
});
