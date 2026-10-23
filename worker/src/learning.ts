import type { Db } from 'mongodb';

export type LearningHorizon = '1d' | '5d' | '20d' | '1y' | '3y' | '5y' | '10y';
export type LearningScope = { tickers: string[]; horizon: LearningHorizon };
export const LEARNING_MARKET_DAYS: Record<LearningHorizon, number> = { '1d': 1, '5d': 5, '20d': 20, '1y': 252, '3y': 756, '5y': 1260, '10y': 2520 };
export const LEARNING_FEATURE_NAMES = ['1-day return (%)', '20-day return volatility (%)', '5-day volume trend (%)', '20-day momentum (%)', 'Public text mood', 'Mention velocity', 'Sentiment change', 'Discussion volume'];
const ALGORITHM = 'market-ridge-v4-nested-selection';
const MIN_TRAIN = 40, MIN_TEST = 20, MAX_ROWS = 2000, MAX_LEDGER = 1000, MAX_PENDING = 200;
const DB_TIMEOUT = 12_000;
const rowLimit = (horizon: LearningHorizon) => horizon === '10y' ? 6500 : horizon === '5y' ? 3000 : MAX_ROWS;
export class LearningError extends Error {
  constructor(message: string, public status = 400, public retryAfter = 0) { super(message); }
}
export type LearningPrice = { ticker: string; observed_at: string; close: number; volume: number; source?: string };
export type SocialDay = { mood_sum: number; mentions: number; engagement: number };
export type SocialSeries = Map<string, SocialDay>;
export type LearningExample = { ticker: string; feature_as_of: string; target_observed_at: string; features: number[]; actual_return: number; forecast_ids: string[] };
export type LearningWeights = { weights: number[]; bias: number };
export type LearningValidation = {
  status: 'supported' | 'insufficient_data'; samples: number; train_samples: number;
  mae_pct: number | null; directional_accuracy: number | null; method: string;
  train_target_through: string | null; test_feature_from: string | null; test_target_through: string | null;
  zero_return_mae_pct?: number | null; historical_mean_mae_pct?: number | null; skill_vs_zero?: number | null;
  paper?: { trades: number; evaluated_windows: number; return_pct: number; max_drawdown_pct: number; transaction_cost_bps: number; method: string } | null;
};
export type ModelChoice = { id: string; label: string; feature_count: number; lambda: number | null };
export type ModelSelection = {
  status: 'compared' | 'fixed_fallback'; selected: ModelChoice; candidates: Array<ModelChoice & { mae_pct: number | null }>;
  train_samples: number; validation_samples: number; train_target_through: string | null; validation_feature_from: string | null;
  selection_ms: number; final_fit_ms: number; method: string;
};
export type LearningPrediction = LearningWeights & {
  id: string; scope_key: string; model_version: string; model_run_id: string;
  ticker: string; horizon: LearningHorizon; market_days: number; as_of: string; feature_as_of: string;
  issued_at: string; base_price: number; predicted_return: number; predicted_price: number;
  features: number[]; contributions: number[]; status: 'pending' | 'resolved';
  actual_price?: number; actual_return?: number; error_pct?: number; absolute_error_pct?: number;
  direction_correct?: boolean; target_observed_at?: string; resolved_at?: string;
  used_for_training: boolean; learned_model_id?: string; learned_at?: string;
};
export type LearningCoverage = { ticker: string; observations: number; first_observed_at: string | null; last_observed_at: string | null; eligible_examples: number; reason?: string };
export type LearningSocialEvidence = {
  ticker: string; total_records: number; reddit_posts: number; other_text_records: number;
  samples: Array<{ id: string; source: string; subreddit?: string; created_at: string; text: string; sentiment: number }>;
};
export type LearningCompute = {
  elapsed_ms: number; market_observations: number; social_records: number; feature_vectors: number;
  train_examples: number; held_out_tests: number; matrix_dimension: number;
  estimated_multiply_adds: number; source_breakdown: { reddit_posts: number; other_text_records: number; historical_signal_rows: number; market_observations: number };
};
export type LearningSnapshot = LearningWeights & {
  scope_key: string; scope: LearningScope; model_version: string; model_run_id: string;
  model_type: string; status: 'ready' | 'insufficient_data'; completed_at: string; cached: boolean;
  train_samples: number; market_observations: number; minimum_examples: number;
  feature_names: string[]; feature_formulas: string[]; validation: LearningValidation;
  model_selection: ModelSelection;
  predictions: LearningPrediction[]; ledger: LearningPrediction[]; coverage: LearningCoverage[]; social_evidence: LearningSocialEvidence[]; compute: LearningCompute;
  resolved_predictions: number; learned_predictions: number; new_predictions: number; notes: string[];
};
type StoredPrediction = LearningPrediction & { _id: string };
type StoredState = { _id: string; fingerprint: string; snapshot: LearningSnapshot };

export function parseLearningScope(value: unknown): LearningScope {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new LearningError('Learning settings must be an object.');
  const input = value as Record<string, unknown>;
  const tickers = input.tickers ?? ['AAPL', 'MSFT', 'NVDA', 'AMZN'];
  if (!Array.isArray(tickers) || !tickers.length || tickers.length > 8 || tickers.some(t => typeof t !== 'string' || !/^[A-Za-z]{1,5}(?:[.-][A-Za-z])?$/.test(t.trim()))) throw new LearningError('Choose one to eight stock symbols for learning.');
  const horizon = input.horizon ?? '1d';
  if (typeof horizon !== 'string' || !Object.hasOwn(LEARNING_MARKET_DAYS, horizon)) throw new LearningError('Choose a supported forecast horizon: 1d, 5d, 20d, 1y, 3y, 5y, or 10y.');
  return { tickers: [...new Set(tickers.map(t => t.trim().toUpperCase()))].sort(), horizon: horizon as LearningHorizon };
}

/** Date-only records follow the source's conservative 21:00 UTC availability convention. */
export function normalizeMarketRows(rows: LearningPrice[], ticker: string, now: string, limit = MAX_ROWS): LearningPrice[] {
  const byDay = new Map<string, LearningPrice>();
  const cutoff = Date.parse(now);
  for (const row of rows) {
    if (row.ticker !== ticker || typeof row.observed_at !== 'string' || !/^\d{4}-\d{2}-\d{2}(?:T|$)/.test(row.observed_at)) continue;
    const day = row.observed_at.slice(0, 10);
    const timestamp = Date.parse(row.observed_at.length === 10 ? `${day}T21:00:00.000Z` : row.observed_at);
    if (!Number.isFinite(timestamp) || timestamp > cutoff || new Date(timestamp).toISOString().slice(0, 10) !== day || !Number.isFinite(row.close) || row.close <= 0 || !Number.isFinite(row.volume) || row.volume < 0) continue;
    const observation = { ticker, observed_at: new Date(timestamp).toISOString(), close: row.close, volume: row.volume, source: row.source ?? 'stored market observation' };
    const previous = byDay.get(day);
    // The source tie-break makes duplicate-day selection independent of input ordering.
    if (!previous || observation.observed_at > previous.observed_at || (observation.observed_at === previous.observed_at && `${observation.source}:${observation.close}:${observation.volume}` > `${previous.source}:${previous.close}:${previous.volume}`)) byDay.set(day, observation);
  }
  return [...byDay.values()].sort((a, b) => a.observed_at.localeCompare(b.observed_at)).slice(-limit);
}

export function marketFeatures(rows: LearningPrice[], index: number, social?: SocialSeries): number[] | null {
  if (index < 20 || index >= rows.length) return null;
  const last = rows[index];
  const returns = rows.slice(index - 19, index + 1).map((row, n) => (row.close / rows[index - 20 + n].close - 1) * 100);
  const average = returns.reduce((sum, value) => sum + value, 0) / 20;
  const volatility = Math.sqrt(returns.reduce((sum, value) => sum + (value - average) ** 2, 0) / 20);
  const recentVolume = rows.slice(index - 4, index + 1).reduce((sum, row) => sum + row.volume, 0) / 5;
  const priorVolume = rows.slice(index - 19, index - 4).reduce((sum, row) => sum + row.volume, 0) / 15;
  const features = [returns[19], volatility, priorVolume > 0 ? (recentVolume / priorVolume - 1) * 100 : 0, (last.close / rows[index - 20].close - 1) * 100];
  // Daily aggregates can include posts after the market close. Use only a fully
  // completed UTC day, conservatively available before this feature timestamp.
  const day = new Date(Date.parse(last.observed_at.slice(0, 10)) - 86400_000).toISOString().slice(0, 10);
  const current = social?.get(day) ?? { mood_sum: 0, mentions: 0, engagement: 0 };
  const previous = [...(social?.entries() ?? [])].filter(([date]) => date < day).sort(([a], [b]) => b.localeCompare(a));
  const priorDays = previous.slice(0, 20).map(([, value]) => value);
  const priorMentions = priorDays.reduce((sum, value) => sum + value.mentions, 0) / Math.max(1, priorDays.length);
  const priorMood = priorDays.slice(0, 7).filter(value => value.mentions > 0);
  const mood = current.mentions ? current.mood_sum / current.mentions : 0;
  const priorMoodAverage = priorMood.length ? priorMood.reduce((sum, value) => sum + value.mood_sum / Math.max(1, value.mentions), 0) / priorMood.length : 0;
  const socialFeatures = [mood, priorMentions > 0 ? (current.mentions / priorMentions - 1) * 100 : 0, mood - priorMoodAverage, Math.log1p(current.mentions)];
  return [...features, ...socialFeatures].every(Number.isFinite) ? [...features, ...socialFeatures] : null;
}
export function historicalExamples(rows: LearningPrice[], marketDays: number, social?: SocialSeries): LearningExample[] {
  const examples: LearningExample[] = [];
  for (let i = 20; i + marketDays < rows.length; i++) {
    const features = marketFeatures(rows, i, social), target = rows[i + marketDays];
    const actualReturn = (target.close / rows[i].close - 1) * 100;
    if (features && Number.isFinite(actualReturn)) examples.push({ ticker: rows[i].ticker, feature_as_of: rows[i].observed_at, target_observed_at: target.observed_at, features, actual_return: actualReturn, forecast_ids: [] });
  }
  return examples;
}

/** Ridge uses only actual returns. Scaling is fitted exclusively on its input training set. */
export function fitRidge(examples: LearningExample[], lambda = 0.1): LearningWeights {
  if (!Number.isFinite(lambda) || lambda <= 0) throw new Error('Ridge penalty must be positive.');
  const featureCount = examples[0]?.features.length ?? LEARNING_FEATURE_NAMES.length;
  if (!examples.length) return { weights: Array(featureCount).fill(0), bias: 0 };
  const means = Array(featureCount).fill(0), scales = Array(featureCount).fill(0);
  const meanY = examples.reduce((sum, row) => sum + row.actual_return, 0) / examples.length;
  for (let j = 0; j < featureCount; j++) {
    means[j] = examples.reduce((sum, row) => sum + row.features[j], 0) / examples.length;
    scales[j] = Math.sqrt(examples.reduce((sum, row) => sum + (row.features[j] - means[j]) ** 2, 0) / examples.length) || 1;
  }
  // Ridge normal equations with an unpenalized intercept.
  const matrix = Array.from({ length: featureCount }, (_, i) => Array.from({ length: featureCount + 1 }, (_, j) => i === j ? lambda : 0));
  for (const row of examples) {
    const x = row.features.map((value, j) => (value - means[j]) / scales[j]);
    for (let i = 0; i < featureCount; i++) {
      for (let j = 0; j < featureCount; j++) matrix[i][j] += x[i] * x[j] / examples.length;
      matrix[i][featureCount] += x[i] * (row.actual_return - meanY) / examples.length;
    }
  }
  for (let i = 0; i < featureCount; i++) {
    let pivot = i;
    for (let k = i + 1; k < featureCount; k++) if (Math.abs(matrix[k][i]) > Math.abs(matrix[pivot][i])) pivot = k;
    [matrix[i], matrix[pivot]] = [matrix[pivot], matrix[i]];
    const divisor = matrix[i][i];
    for (let j = i; j <= featureCount; j++) matrix[i][j] /= divisor;
    for (let k = 0; k < featureCount; k++) if (k !== i) {
      const factor = matrix[k][i];
      for (let j = i; j <= featureCount; j++) matrix[k][j] -= factor * matrix[i][j];
    }
  }
  const weights = matrix.map((row, j) => row[featureCount] / scales[j]);
  return { weights, bias: meanY - weights.reduce((sum, weight, j) => sum + weight * means[j], 0) };
}
export function predictReturn(model: LearningWeights, features: number[]): number { return model.bias + features.reduce((sum, value, i) => sum + value * model.weights[i], 0); }

/** Equal timestamps always stay in the same partition; purge unmatured labels. */
export function chronologicalSplit(examples: LearningExample[], requestedTestSize?: number) {
  const ordered = [...examples].sort((a, b) => a.feature_as_of.localeCompare(b.feature_as_of) || a.ticker.localeCompare(b.ticker));
  const wanted = Math.min(200, Math.max(MIN_TEST, requestedTestSize ?? Math.floor(ordered.length * 0.2)));
  const boundary = ordered[Math.max(0, ordered.length - wanted)]?.feature_as_of;
  const test = boundary ? ordered.filter(row => row.feature_as_of >= boundary) : [];
  const train = boundary ? ordered.filter(row => row.feature_as_of < boundary && row.target_observed_at <= boundary) : [];
  return { train, test, boundary: boundary ?? null };
}
const choices: ModelChoice[] = [
  { id: 'mean', label: 'Historical mean', feature_count: 0, lambda: null },
  ...[4, 8].flatMap(feature_count => [0.01, 0.1, 1].map(lambda => ({ id: `ridge-${feature_count}-${lambda}`, label: feature_count === 4 ? 'Price & volume ridge' : 'Price + public text ridge', feature_count, lambda }))),
];
export function fitChoice(examples: LearningExample[], choice: ModelChoice): LearningWeights {
  const dimension = examples[0]?.features.length ?? LEARNING_FEATURE_NAMES.length;
  if (!choice.feature_count) return { weights: Array(dimension).fill(0), bias: examples.length ? examples.reduce((sum,row) => sum + row.actual_return, 0) / examples.length : 0 };
  const model = fitRidge(examples.map(row => ({ ...row, features: row.features.slice(0, choice.feature_count) })), choice.lambda ?? 0.1);
  return { bias: model.bias, weights: [...model.weights, ...Array(Math.max(0, dimension - model.weights.length)).fill(0)] };
}
const mae = (model: LearningWeights, rows: LearningExample[]) => rows.reduce((sum,row) => sum + Math.abs(predictReturn(model, row.features) - row.actual_return), 0) / rows.length;

/** Selection sees only the outer training partition, never the final test labels. */
export function selectFastModel(training: LearningExample[]): ModelSelection {
  const started = performance.now();
  const { train, test, boundary } = chronologicalSplit(training);
  const compared = train.length >= MIN_TRAIN && test.length >= MIN_TEST;
  const candidates = choices.map(choice => ({ ...choice, mae_pct: compared ? mae(fitChoice(train, choice), test) : null }));
  const selected = compared ? [...candidates].sort((a,b) => a.mae_pct! - b.mae_pct! || a.feature_count - b.feature_count)[0] : choices[2];
  return {
    status: compared ? 'compared' : 'fixed_fallback', selected: { id: selected.id, label: selected.label, feature_count: selected.feature_count, lambda: selected.lambda }, candidates,
    train_samples: train.length, validation_samples: test.length,
    train_target_through: train.length ? train.reduce((last,row) => row.target_observed_at > last ? row.target_observed_at : last, '') : null,
    validation_feature_from: boundary, selection_ms: performance.now() - started, final_fit_ms: 0,
    method: compared ? 'Seven small candidates compared by mean absolute error on an inner chronological holdout, with overlapping training targets purged. The final test period is excluded from selection.' : 'Not enough matured outcomes for an inner comparison. Use the fixed price-only ridge (λ = 0.1); final tests remain separate.',
  };
}

/** Hypothetical equal-capital ticker sleeves. Each sleeve holds one position at a time. */
export function evaluatePaperPositions(rows: LearningExample[], model: LearningWeights, costBps = 10) {
  const symbols = [...new Set(rows.map(row => row.ticker))];
  const next = new Map<string,string>(), equity = new Map(symbols.map(symbol => [symbol,1]));
  const events: Array<{ ticker: string; date: string; return_pct: number }> = [];
  let trades = 0, evaluated = 0;
  for (const row of [...rows].sort((a,b) => a.feature_as_of.localeCompare(b.feature_as_of) || a.ticker.localeCompare(b.ticker))) {
    if (row.feature_as_of < (next.get(row.ticker) ?? '')) continue;
    next.set(row.ticker, row.target_observed_at); evaluated++;
    if (predictReturn(model, row.features) > costBps / 100) {
      events.push({ ticker: row.ticker, date: row.target_observed_at, return_pct: row.actual_return - costBps / 100 }); trades++;
    }
  }
  let peak = 1, drawdown = 0;
  const byDate = new Map<string, typeof events>();
  for (const event of events) byDate.set(event.date, [...(byDate.get(event.date) ?? []),event]);
  for (const date of [...byDate.keys()].sort()) {
    for (const event of byDate.get(date)!) equity.set(event.ticker, equity.get(event.ticker)! * Math.max(0, 1 + event.return_pct / 100));
    const total = [...equity.values()].reduce((sum,value) => sum + value,0) / Math.max(1,symbols.length);
    peak = Math.max(peak,total); drawdown = Math.max(drawdown, (1 - total / peak) * 100);
  }
  return { trades, evaluated_windows: evaluated, return_pct: ([...equity.values()].reduce((sum,value) => sum + value,0) / Math.max(1,symbols.length) - 1) * 100, max_drawdown_pct: drawdown, transaction_cost_bps: costBps,
    method: 'Illustrative long/cash rule: enter only when forecast exceeds 0.10% round-trip costs and slippage. Equal initial capital per ticker, no overlapping positions per ticker. Returns and drawdown measured at exits, excluding dividends, intraperiod drawdowns and execution delays; not an executable trading backtest.' };
}

export function evaluateChronologically(examples: LearningExample[], requestedTestSize?: number, choice?: ModelChoice): LearningValidation {
  const { train, test, boundary } = chronologicalSplit(examples, requestedTestSize);
  const supported = train.length >= MIN_TRAIN && test.length >= MIN_TEST;
  // The standalone evaluator keeps its documented fixed-ridge default. The
  // production pipeline passes the candidate selected using training data only.
  const model = supported ? choice ? fitChoice(train,choice) : fitRidge(train) : null;
  const baseline = supported ? test.reduce((sum,row) => sum + Math.abs(row.actual_return),0) / test.length : null;
  const error = model ? mae(model,test) : null;
  return {
    status: supported ? 'supported' : 'insufficient_data', samples: test.length, train_samples: train.length,
    mae_pct: error,
    zero_return_mae_pct: baseline, historical_mean_mae_pct: supported ? mae(fitChoice(train, choices[0]),test) : null,
    skill_vs_zero: baseline && error !== null ? 1 - error / baseline : null,
    paper: model ? evaluatePaperPositions(test,model) : null,
    directional_accuracy: model ? test.filter(row => Math.sign(predictReturn(model, row.features)) === Math.sign(row.actual_return)).length / test.length : null,
    method: 'Chronological held-out actual returns; training targets mature by the first test feature time. Overlapping horizon windows are not independent. Accuracy is evaluation, not forecast confidence.',
    train_target_through: train.length ? train.reduce((last, row) => row.target_observed_at > last ? row.target_observed_at : last, '') : null,
    test_feature_from: boundary ?? null,
    test_target_through: test.length ? test.reduce((last, row) => row.target_observed_at > last ? row.target_observed_at : last, '') : null,
  };
}

/** A forecast issued today cannot be scored with a target already known today. */
export function resolveForecast(forecast: LearningPrediction, prices: LearningPrice[], now: string): Partial<LearningPrediction> | null {
  if (forecast.status !== 'pending' || !Number.isFinite(forecast.base_price) || forecast.base_price <= 0 || !Number.isFinite(Date.parse(forecast.issued_at))) return null;
  const cutoff = forecast.issued_at > forecast.feature_as_of ? forecast.issued_at : forecast.feature_as_of;
  const available = normalizeMarketRows(prices, forecast.ticker, now, rowLimit(forecast.horizon));
  // Refuse a shifted target if bounded history no longer reaches the issue cutoff.
  if (!available.some(row => row.observed_at <= cutoff)) return null;
  const later = available.filter(row => row.observed_at > cutoff && row.observed_at.slice(0, 10) > forecast.as_of.slice(0, 10));
  const actual = later[forecast.market_days - 1];
  if (!actual) return null;
  const actualReturn = (actual.close / forecast.base_price - 1) * 100;
  const error = forecast.predicted_return - actualReturn;
  return { status: 'resolved', actual_price: actual.close, actual_return: actualReturn, error_pct: error, absolute_error_pct: Math.abs(error), direction_correct: Math.sign(actualReturn) === Math.sign(forecast.predicted_return), target_observed_at: actual.observed_at, resolved_at: now };
}
function forecastExample(row: LearningPrediction, cutoff: string): LearningExample | null {
  if (!row.model_version.startsWith(ALGORITHM) || row.status !== 'resolved' || !row.target_observed_at || row.target_observed_at > cutoff || !Number.isFinite(row.actual_return) || row.features.length !== LEARNING_FEATURE_NAMES.length || !row.features.every(Number.isFinite)) return null;
  return { ticker: row.ticker, feature_as_of: row.feature_as_of, target_observed_at: row.target_observed_at, features: [...row.features], actual_return: row.actual_return!, forecast_ids: [row.id] };
}
function mergeExamples(history: LearningExample[], forecasts: LearningPrediction[], cutoff: string): LearningExample[] {
  const unique = new Map<string, LearningExample>();
  const key = (row: LearningExample) => `${row.ticker}:${row.feature_as_of}:${row.target_observed_at}`;
  for (const row of history) if (row.target_observed_at <= cutoff) unique.set(key(row), row);
  for (const forecast of forecasts) {
    const row = forecastExample(forecast, cutoff);
    if (!row) continue;
    const previous = unique.get(key(row));
    unique.set(key(row), { ...row, forecast_ids: [...(previous?.forecast_ids ?? []), ...row.forecast_ids] });
  }
  return [...unique.values()].sort((a, b) => a.feature_as_of.localeCompare(b.feature_as_of) || a.ticker.localeCompare(b.ticker));
}
async function hash(value: unknown): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(value)));
  return [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2, '0')).join('');
}
function publicPrediction(value: StoredPrediction): LearningPrediction { const { _id, ...row } = value; return row; }
function checkDeadline(deadline: number) { if (Date.now() > deadline) throw new LearningError('Learning reached its time limit. Saved forecasts remain available.', 503, 60); }
async function twoAtATime<T>(jobs: Array<() => Promise<T>>): Promise<T[]> {
  const results: T[] = []; let next = 0;
  const work = async () => { while (next < jobs.length) { const index = next++; results[index] = await jobs[index](); } };
  await Promise.all([work(), work()]);
  return results;
}

export async function getLearning(database: Db, input: LearningScope, includeLedger = true): Promise<LearningSnapshot | null> {
  const scope = parseLearningScope(input), key = await hash(scope);
  const stored = await database.collection<StoredState>('learning_state').findOne({ _id: key }, { timeoutMS: DB_TIMEOUT });
  if (!stored) return null;
  // The saved snapshot includes the most recent history from its completed run.
  // Returning it in one read avoids a second MongoDB cursor and its CPU cost on
  // Cloudflare's free tier. Full history remains in learning_predictions.
  if (!includeLedger) return stored.snapshot;
  return { ...stored.snapshot, ledger: Array.isArray(stored.snapshot.ledger) ? stored.snapshot.ledger : [] };
}

export async function runLearning(database: Db, input: LearningScope): Promise<LearningSnapshot> {
  let phase = 'settings';
  try { return await runLearningCore(database, input, value => { phase = value; }); }
  catch (error) {
    if (error instanceof LearningError) throw error;
    const detail = error instanceof Error ? error.message.slice(0, 180) : 'MongoDB request failed';
    throw new LearningError(`Learning ${phase} failed: ${detail}`, 503, 30);
  }
}

async function runLearningCore(database: Db, input: LearningScope, reportPhase: (phase: string) => void): Promise<LearningSnapshot> {
  const scope = parseLearningScope(input), key = await hash(scope), started = Date.now(), deadline = started + 150_000;
  const phase = (name: string) => { checkDeadline(deadline); reportPhase(name); };
  const pricesCollection = database.collection<LearningPrice>('market');
  const predictions = database.collection<StoredPrediction>('learning_predictions');
  const states = database.collection<StoredState>('learning_state');
  const locks = database.collection<{ _id: string; lease_id?: string; expires_at: Date }>('learning_locks');
  const budget = database.collection<{ _id: string; runs: number; expires_at: Date }>('learning_budget');
  const hour = new Date(started).toISOString().slice(0, 13);
  phase('allowance check');
  const [existingBudget, previousState] = await Promise.all([
    budget.findOne({ _id: hour }, { timeoutMS: DB_TIMEOUT }),
    states.findOne({ _id: key }, { timeoutMS: DB_TIMEOUT }),
  ]);
  if ((existingBudget?.runs ?? 0) >= 12) {
    if (previousState) return { ...previousState.snapshot, cached: true, new_predictions: 0, notes: [...previousState.snapshot.notes, 'The shared hourly learning allowance is used. Showing the last saved model; new prices and outcomes will be processed when the allowance reopens.'] };
    throw new LearningError('The shared learning allowance is used for this hour. Saved forecasts remain available.', 429, Math.ceil((3_600_000 - started % 3_600_000) / 1000));
  }
  const maxRows = rowLimit(scope.horizon);
  phase('index setup');
  await twoAtATime([
    () => pricesCollection.createIndex({ ticker: 1, observed_at: -1 }, { maxTimeMS: DB_TIMEOUT, timeoutMS: DB_TIMEOUT }),
    () => predictions.createIndex({ scope_key: 1, issued_at: -1 }, { maxTimeMS: DB_TIMEOUT, timeoutMS: DB_TIMEOUT }),
    () => { checkDeadline(deadline); return predictions.createIndex({ scope_key: 1, status: 1, target_observed_at: -1 }, { maxTimeMS: DB_TIMEOUT, timeoutMS: DB_TIMEOUT }); },
    () => { checkDeadline(deadline); return budget.createIndex({ expires_at: 1 }, { expireAfterSeconds: 0, maxTimeMS: DB_TIMEOUT, timeoutMS: DB_TIMEOUT }); },
  ]);
  phase('lease setup');
  await locks.updateOne({ _id: key }, { $setOnInsert: { expires_at: new Date(0) } }, { upsert: true, timeoutMS: DB_TIMEOUT });
  const leaseId = crypto.randomUUID();
  phase('lease acquisition');
  const lease = await locks.findOneAndUpdate({ _id: key, expires_at: { $lte: new Date(started) } }, { $set: { lease_id: leaseId, expires_at: new Date(started + 180_000) } }, { returnDocument: 'after', timeoutMS: DB_TIMEOUT });
  if (!lease) throw new LearningError('Learning for these symbols and this horizon is already running.', 429, 30);
  try {
    const now = new Date().toISOString();
    phase('saved forecast read');
    const [stored, pending] = await Promise.all([
      states.findOne({ _id: key }, { timeoutMS: DB_TIMEOUT }),
      predictions.find({ scope_key: key, status: 'pending' }, { timeoutMS: DB_TIMEOUT }).sort({ issued_at: 1 }).limit(MAX_PENDING).toArray(),
    ]);
    phase('market history read');
    const rawPrices = await twoAtATime(scope.tickers.map(ticker => () => {
      checkDeadline(deadline);
      return pricesCollection.find({ ticker, observed_at: { $lte: now } }, { projection: { _id: 0, ticker: 1, observed_at: 1, close: 1, volume: 1, source: 1 }, timeoutMS: DB_TIMEOUT }).sort({ observed_at: -1 }).limit(maxRows * 2).toArray();
    }));
    checkDeadline(deadline);
    const series = new Map(scope.tickers.map((ticker, i) => [ticker, normalizeMarketRows(rawPrices[i], ticker, now, maxRows)]));
    phase('social history read');
    const discussionRows = await database.collection<Record<string, unknown>>('discussions').find({ analyzed: true, sentiment: { $ne: null }, source: { $ne: 'googletrends' } }, { projection: { _id: 1, text: 1, subreddit: 1, source: 1, tickers: 1, created_at: 1, createdAt: 1, first_received_at: 1, sentiment: 1 }, timeoutMS: DB_TIMEOUT }).sort({ created_at: -1 }).limit(50_000).toArray();
    const socialByTicker = new Map<string, SocialSeries>(scope.tickers.map(ticker => [ticker, new Map()]));
    const socialEvidence = new Map(scope.tickers.map(ticker => [ticker, { ticker, total_records: 0, reddit_posts: 0, other_text_records: 0, samples: [] as LearningSocialEvidence['samples'] }]));
    const addSocial = (ticker: string, date: string, sentiment: number, mentions = 1, engagement = mentions) => {
      if (!socialByTicker.has(ticker) || !/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(sentiment)) return;
      const seriesForTicker = socialByTicker.get(ticker)!;
      const previous = seriesForTicker.get(date) ?? { mood_sum: 0, mentions: 0, engagement: 0 };
      previous.mood_sum += sentiment * mentions; previous.mentions += mentions; previous.engagement += Math.max(mentions, engagement); seriesForTicker.set(date, previous);
    };
    for (const row of discussionRows) {
      const rawTickers = row.tickers;
      const tickers = Array.isArray(rawTickers) ? rawTickers : typeof rawTickers === 'string' ? (() => { try { return JSON.parse(rawTickers); } catch { return []; } })() : [];
      const published = String(row.created_at ?? row.createdAt ?? '');
      const available = typeof row.first_received_at === 'string' && row.first_received_at > published ? row.first_received_at : published;
      const date = available.slice(0, 10), sentiment = Number(row.sentiment ?? 0);
      const source = String(row.source ?? 'public text'), subreddit = typeof row.subreddit === 'string' ? row.subreddit : undefined;
      const text = String(row.text ?? '').replace(/\s+/g, ' ').trim().slice(0, 320);
      for (const ticker of new Set(Array.isArray(tickers) ? tickers : [])) if (typeof ticker === 'string') {
        const symbol = ticker.toUpperCase();
        if (!Number.isFinite(Date.parse(available)) || available > now) continue;
        addSocial(symbol, date, sentiment);
        const evidence = socialEvidence.get(symbol);
        if (!evidence) continue;
        evidence.total_records += 1;
        if (source.toLowerCase().startsWith('reddit:')) evidence.reddit_posts += 1; else evidence.other_text_records += 1;
        if (evidence.samples.length < 8 && date && text) evidence.samples.push({ id: String(row._id ?? `${source}:${date}:${text.slice(0, 32)}`), source, ...(subreddit ? { subreddit } : {}), created_at: String(row.created_at ?? row.createdAt ?? date), text, sentiment });
      }
    }
    // Aggregated historical signals have no provenance proving when they were
    // available and may duplicate the raw posts. Keep them out of model features.
    const resolutions = pending.flatMap(row => {
      const outcome = resolveForecast(row, series.get(row.ticker) ?? [], now);
      return outcome ? [{ updateOne: { filter: { _id: row._id, status: 'pending' as const }, update: { $set: outcome } } }] : [];
    });
    phase('outcome resolution');
    if (resolutions.length) await predictions.bulkWrite(resolutions, { ordered: false, timeoutMS: DB_TIMEOUT });
    phase('resolved outcome read');
    const resolved = await predictions.find({ scope_key: key, status: 'resolved' }, { timeoutMS: DB_TIMEOUT }).sort({ target_observed_at: -1, _id: 1 }).limit(MAX_LEDGER).toArray();
    const fingerprint = await hash({ algorithm: ALGORITHM, scope, prices: [...series.values()], social: [...socialByTicker].map(([ticker,days]) => [ticker,[...days].sort(([a],[b]) => a.localeCompare(b))]), outcomes: resolved.map(row => [row.id, row.feature_as_of, row.target_observed_at, row.features, row.actual_return]) });
    // Rebuild once for snapshots written before the transparency fields were
    // added, so an old cached result cannot hide source and compute details.
    if (stored?.fingerprint === fingerprint && stored.snapshot.compute && stored.snapshot.social_evidence && typeof stored.snapshot.compute.source_breakdown?.historical_signal_rows === 'number') {
      phase('cached ledger read');
      const ledger = await predictions.find({ scope_key: key }, { timeoutMS: DB_TIMEOUT }).sort({ issued_at: -1, _id: 1 }).limit(500).toArray();
      return { ...stored.snapshot, ledger: ledger.map(publicPrediction), cached: true, new_predictions: 0 };
    }
    // A shared budget also bounds requests using many distinct scopes.
    phase('allowance setup');
    await budget.updateOne({ _id: hour }, { $setOnInsert: { runs: 0, expires_at: new Date(started + 86_400_000) } }, { upsert: true, timeoutMS: DB_TIMEOUT });
    phase('allowance reservation');
    const token = await budget.findOneAndUpdate({ _id: hour, runs: { $lt: 12 } }, { $inc: { runs: 1 } }, { returnDocument: 'after', timeoutMS: DB_TIMEOUT });
    if (!token) throw new LearningError('The shared learning allowance is used for this hour. Saved forecasts remain available.', 429, Math.ceil((3_600_000 - started % 3_600_000) / 1000));
    checkDeadline(deadline);
    const days = LEARNING_MARKET_DAYS[scope.horizon];
    const usable = [...series.values()].filter(rows => rows.length >= 21);
    // Shared training labels must be known by every issued forecast's feature timestamp.
    const cutoff = usable.map(rows => rows.at(-1)!.observed_at).sort()[0] ?? now;
    const examples = mergeExamples([...series].flatMap(([ticker, rows]) => historicalExamples(rows, days, socialByTicker.get(ticker))), resolved, cutoff);
    // Multi-year labels mature far after their feature date. A smaller held-out
    // window preserves earlier matured labels for training while retaining the
    // same minimum of twenty chronological tests.
    const testSize = days >= 756 ? MIN_TEST : undefined;
    const selection = selectFastModel(chronologicalSplit(examples, testSize).train);
    const validation = evaluateChronologically(examples, testSize, selection.selected);
    const ready = examples.length >= MIN_TRAIN && validation.status === 'supported';
    const fitStarted = performance.now();
    const model = examples.length >= MIN_TRAIN ? fitChoice(examples, selection.selected) : { weights: Array(LEARNING_FEATURE_NAMES.length).fill(0), bias: 0 };
    selection.final_fit_ms = performance.now() - fitStarted;
    const modelVersion = `${ALGORITHM}:${fingerprint.slice(0, 20)}`;
    const modelRunId = `${key}:${modelVersion}`;
    const coverage: LearningCoverage[] = [...series].map(([ticker, rows]) => {
      const eligible = examples.filter(row => row.ticker === ticker).length;
      return { ticker, observations: rows.length, first_observed_at: rows[0]?.observed_at ?? null, last_observed_at: rows.at(-1)?.observed_at ?? null, eligible_examples: eligible,
        ...(rows.length < days + 21 ? { reason: `Need at least ${days + 21} daily observations for one ${scope.horizon} actual outcome and a 20-session feature window.` } : !ready ? { reason: `Need ${MIN_TRAIN} training outcomes before the held-out feature window and ${MIN_TEST} held-out actual outcomes; this scope has ${validation.train_samples} and ${validation.samples}.` } : {}) };
    });
    const issuedAt = new Date().toISOString();
    const newForecasts: StoredPrediction[] = ready ? usable.flatMap(rows => {
      const latest = rows.at(-1)!, features = marketFeatures(rows, rows.length - 1, socialByTicker.get(latest.ticker))!;
      if (!examples.some(row => row.ticker === latest.ticker)) return [];
      const predictedReturn = predictReturn(model, features);
      if (!Number.isFinite(predictedReturn) || predictedReturn <= -100) return [];
      const id = `${key}:${latest.ticker}:${latest.observed_at.slice(0, 10)}:${modelVersion}`;
      return [{ _id: id, id, scope_key: key, model_version: modelVersion, model_run_id: modelRunId, ticker: latest.ticker, horizon: scope.horizon, market_days: days,
        as_of: latest.observed_at, feature_as_of: latest.observed_at, issued_at: issuedAt, base_price: latest.close, predicted_return: predictedReturn,
        predicted_price: latest.close * (1 + predictedReturn / 100), features, weights: [...model.weights], bias: model.bias,
        contributions: features.map((feature, i) => feature * model.weights[i]), status: 'pending' as const, used_for_training: false }];
    }) : [];
    checkDeadline(deadline);
    // $setOnInsert preserves original issue times, features, and predictions even after retries.
    phase('model save');
    await database.collection<{ _id: string }>('learning_runs').updateOne({ _id: modelRunId }, { $setOnInsert: { scope_key: key, scope, model_version: modelVersion, created_at: issuedAt, training_target_through: cutoff, train_samples: examples.length, weights: model.weights, bias: model.bias, ridge_lambda: selection.selected.lambda, model_selection: selection, feature_names: LEARNING_FEATURE_NAMES, validation, fingerprint, learned_forecast_ids: examples.flatMap(row => row.forecast_ids) } }, { upsert: true, timeoutMS: DB_TIMEOUT });
    let inserted = 0;
    if (newForecasts.length) {
      phase('forecast save');
      const written = await predictions.bulkWrite(newForecasts.map(row => ({ updateOne: { filter: { _id: row._id }, update: { $setOnInsert: row }, upsert: true } })), { ordered: false, timeoutMS: DB_TIMEOUT });
      inserted = written.upsertedCount;
    }
    const learnedIds = examples.length >= MIN_TRAIN ? [...new Set(examples.flatMap(row => row.forecast_ids))] : [];
    phase('learned outcome save');
    if (learnedIds.length) await predictions.updateMany({ _id: { $in: learnedIds }, status: 'resolved', used_for_training: false }, { $set: { used_for_training: true, learned_model_id: modelRunId, learned_at: new Date().toISOString() } }, { timeoutMS: DB_TIMEOUT });
    phase('saved result read');
    const [savedPredictions, ledger] = await Promise.all([
      newForecasts.length ? predictions.find({ _id: { $in: newForecasts.map(row => row.id) } }, { timeoutMS: DB_TIMEOUT }).sort({ ticker: 1 }).limit(8).toArray() : Promise.resolve([]),
      predictions.find({ scope_key: key }, { timeoutMS: DB_TIMEOUT }).sort({ issued_at: -1, _id: 1 }).limit(20).toArray(),
    ]);
    const marketObservationCount = [...series.values()].reduce((sum, rows) => sum + rows.length, 0);
    const socialEvidenceRows = [...socialEvidence.values()];
    const socialRecordCount = socialEvidenceRows.reduce((sum, row) => sum + row.total_records, 0);
    const redditPostCount = socialEvidenceRows.reduce((sum, row) => sum + row.reddit_posts, 0);
    const otherTextCount = socialEvidenceRows.reduce((sum, row) => sum + row.other_text_records, 0);
    const featureCount = LEARNING_FEATURE_NAMES.length;
    // This is a transparent upper-bound estimate of scalar multiply/add work,
    // not a billing or CPU-meter reading. Wall time below includes database I/O.
    const solveWork = (n: number, k: number) => 2 * n * (k * k + k) + 2 * k ** 3;
    const estimatedMultiplyAdds = solveWork(examples.length, selection.selected.feature_count) + solveWork(validation.train_samples, selection.selected.feature_count) + validation.samples * featureCount * 2 + (selection.status === 'compared' ? selection.candidates.reduce((sum,candidate) => sum + solveWork(selection.train_samples,candidate.feature_count) + selection.validation_samples * candidate.feature_count * 2,0) : 0);
    const snapshot: LearningSnapshot = {
      scope_key: key, scope, model_version: modelVersion, model_run_id: modelRunId, model_type: selection.selected.label, model_selection: selection,
      status: ready && savedPredictions.length ? 'ready' : 'insufficient_data', completed_at: new Date().toISOString(), cached: false,
      train_samples: examples.length, market_observations: [...series.values()].reduce((sum, rows) => sum + rows.length, 0), minimum_examples: MIN_TRAIN,
      feature_names: [...LEARNING_FEATURE_NAMES], feature_formulas: ['100 × (close / previous close − 1)', 'Population standard deviation of the last 20 daily percentage returns', '100 × (mean volume over last 5 sessions / mean volume over preceding 15 sessions − 1); zero when the denominator is zero', '100 × (close / close 20 sessions earlier − 1)', 'Mean public-text mood from the previous completed UTC day', 'Previous completed day mention count relative to the prior 20 observed social days', 'Previous completed day mood minus the prior 7 observed social-day mood average', 'ln(1 + previous completed UTC day mention count)'],
      weights: model.weights, bias: model.bias, validation, predictions: savedPredictions.map(publicPrediction), ledger: ledger.map(publicPrediction), coverage, social_evidence: socialEvidenceRows,
      compute: { elapsed_ms: Date.now() - started, market_observations: marketObservationCount, social_records: socialRecordCount, feature_vectors: examples.length, train_examples: validation.train_samples, held_out_tests: validation.samples, matrix_dimension: selection.selected.feature_count, estimated_multiply_adds: estimatedMultiplyAdds, source_breakdown: { reddit_posts: redditPostCount, other_text_records: otherTextCount, historical_signal_rows: 0, market_observations: marketObservationCount } },
      resolved_predictions: resolved.length, learned_predictions: learnedIds.length, new_predictions: inserted,
      notes: [
        `Selected ${selection.selected.label}${selection.selected.lambda === null ? '' : ` with λ = ${selection.selected.lambda}`}. Predicted return (%) = bias + feature × weight terms; unused features have zero weights. The ridge objective is mean squared error + λ × sum of squared standardized weights. No probability or confidence is inferred.`,
        selection.method,
        'Social features use the previous completed UTC day, dated no earlier than first receipt when recorded. Older records without receipt timestamps use publication dates as reconstructed availability. Missing days are encoded as zero, not verified neutral opinion. Unversioned historical aggregates and later engagement counts are excluded.',
        `Only selected symbols and horizon are used. Each fit reads up to ${maxRows} distinct market days per symbol and ${MAX_LEDGER} recent resolved forecasts.`,
        'History labels are observed closing-price returns. Saved forecasts enter training only after an actual target price resolves them; predicted targets are never training labels.',
        'A live target is the Nth distinct recorded market session strictly after the forecast issue time. Pending forecasts have no asserted calendar target date.',
        'Historical validation uses dated prices available at each feature timestamp. Provider revisions, dividends, and stock splits may affect unadjusted closes; validation is not a trading backtest.',
        ...(days >= 756 ? ['Long-horizon validation reserves twenty held-out outcomes so multi-year labels can mature before the chronological test window.'] : []),
        ...(!ready ? [`Insufficient chronological evidence: ${validation.train_samples}/${MIN_TRAIN} training examples before the test period; ${validation.samples}/${MIN_TEST} held-out actual outcomes. Long horizons may need more than two horizon windows of history.`] : []),
      ],
    };
    phase('snapshot save');
    await states.updateOne({ _id: key }, { $set: { fingerprint, snapshot } }, { upsert: true, timeoutMS: DB_TIMEOUT });
    return snapshot;
  } finally {
    // Cleanup failure must not hide a committed snapshot or its original error.
    // The existing three-minute expiry safely releases an abandoned lease.
    await locks.updateOne({ _id: key, lease_id: leaseId }, { $set: { expires_at: new Date(0) } }, { timeoutMS: DB_TIMEOUT }).catch(() => undefined);
  }
}
