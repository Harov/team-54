import { test } from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import http from 'node:http';
import { build } from 'esbuild';
import { createTrainingServer } from '../render-http.mjs';

const bundle = await build({ entryPoints: ['src/learning.ts'], bundle: true, platform: 'node', format: 'esm', write: false });
const { parseLearningScope, LearningError } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
const trainingKey = 'test-only-key-that-is-at-least-32-characters';
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
async function service(t, train) {
  const server = createTrainingServer({ trainingKey, parseScope: parseLearningScope, LearningError, train, reportError() {} });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  const url = `http://127.0.0.1:${server.address().port}`;
  const post = (body = '{}', headers = {}) => fetch(`${url}/train`, { method: 'POST', headers: { 'content-type': 'application/json', 'x-training-key': trainingKey, ...headers }, body });
  return { url, post };
}

test('Render health checks avoid training and database access', async t => {
  const { url } = await service(t, () => assert.fail('health must not train'));
  const response = await fetch(`${url}/health`);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { status: 'ok' });
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.equal((await fetch(`${url}/missing`)).status, 404);
});

test('Render rejects missing or incorrect credentials before training', async t => {
  const { post } = await service(t, () => assert.fail('unauthorized request must not train'));
  assert.equal((await post('{}', { 'x-training-key': '' })).status, 401);
  assert.equal((await post('{}', { 'x-training-key': 'x'.repeat(trainingKey.length) })).status, 401);
});

test('Render rejects invalid JSON, settings, and content types before training', async t => {
  const { post } = await service(t, () => assert.fail('invalid request must not train'));
  for (const body of ['{', 'null', '[]', '{"horizon":"2y"}', '{"tickers":["../secret"]}']) {
    assert.equal((await post(body)).status, 400, body);
  }
  assert.equal((await post('{}', { 'content-type': 'text/plain' })).status, 415);
});

test('Render authenticates and passes canonical settings to the training implementation', async t => {
  const { post } = await service(t, async scope => {
    assert.deepEqual(scope, { tickers: ['AAPL', 'MSFT'], horizon: '1y' });
    return { status: 'ready', scope, predictions: [{ ticker: 'AAPL', predicted_return: 1.23 }] };
  });
  const response = await post('{"tickers":["msft","AAPL","aapl"],"horizon":"1y"}');
  assert.equal(response.status, 200);
  assert.equal((await response.json()).predictions[0].predicted_return, 1.23);
});

test('Render limits declared and streamed request bodies by byte size', async t => {
  const { url, post } = await service(t, () => assert.fail('oversized request must not train'));
  assert.equal((await post(JSON.stringify({ extra: 'x'.repeat(4096) }))).status, 413);
  const responseStatus = await new Promise((resolve, reject) => {
    const request = http.request(`${url}/train`, { method: 'POST', headers: { 'content-type': 'application/json', 'x-training-key': trainingKey, 'transfer-encoding': 'chunked' } }, response => {
      response.resume();
      resolve(response.statusCode);
    });
    request.on('error', reject);
    request.write('{"extra":"');
    request.write('é'.repeat(2100));
    request.end('"}');
  });
  assert.equal(responseStatus, 413);
});

test('Render runs one job at a time and frees its slot after success', async t => {
  const entered = deferred(), finish = deferred();
  let calls = 0;
  const { post } = await service(t, async () => { calls++; entered.resolve(); await finish.promise; return { status: 'ready' }; });
  const first = post();
  await entered.promise;
  const second = await post();
  assert.equal(second.status, 429);
  assert.equal(second.headers.get('retry-after'), '30');
  assert.equal(calls, 1);
  finish.resolve();
  assert.equal((await first).status, 200);
  assert.equal((await post()).status, 200);
  assert.equal(calls, 2);
});

test('Render returns safe database errors and allows a subsequent job', async t => {
  let calls = 0;
  const { post } = await service(t, async () => {
    if (++calls === 1) throw new LearningError('mongodb://secret:password@example.invalid', 503);
    return { status: 'ready' };
  });
  const failure = await post();
  assert.equal(failure.status, 503);
  assert.equal(failure.headers.get('retry-after'), '30');
  assert.doesNotMatch(await failure.text(), /secret|password|example/);
  assert.equal((await post()).status, 200);
});

test('Render preserves the shared training budget response', async t => {
  const { post } = await service(t, async () => { throw new LearningError('Shared hourly allowance is used.', 429, 120); });
  const response = await post();
  assert.equal(response.status, 429);
  assert.equal(response.headers.get('retry-after'), '120');
  assert.deepEqual(await response.json(), { detail: 'Shared hourly allowance is used.', retry_after: 120 });
});
