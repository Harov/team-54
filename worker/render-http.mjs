import http from 'node:http';
import { timingSafeEqual } from 'node:crypto';

const MAX_BODY_BYTES = 4096;
const reply = (res, status, value, retryAfter = 0) => {
  const headers = { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
  if (retryAfter > 0) headers['Retry-After'] = String(Math.ceil(retryAfter));
  res.writeHead(status, headers);
  res.end(JSON.stringify(value));
};

/** A single training job protects a small instance; MongoDB also locks each scope. */
export function createTrainingServer({ trainingKey, parseScope, train, LearningError, reportError = error => console.error('Training failed:', error instanceof Error ? error.name : 'unknown error') }) {
  if (typeof trainingKey !== 'string' || trainingKey.length < 32) throw new Error('A training key of at least 32 characters is required.');
  const expected = Buffer.from(trainingKey);
  let training = false;
  return http.createServer({ requestTimeout: 15000, headersTimeout: 10000, keepAliveTimeout: 5000 }, async (req, res) => {
    // This checks process availability, without consuming a MongoDB connection.
    if (req.method === 'GET' && req.url === '/health') return reply(res, 200, { status: 'ok' });
    if (req.method !== 'POST' || req.url !== '/train') return reply(res, 404, { detail: 'Not found.' });
    const key = req.headers['x-training-key'];
    const actual = typeof key === 'string' ? Buffer.from(key) : Buffer.alloc(0);
    if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) return reply(res, 401, { detail: 'Unauthorized.' });
    if (req.headers['content-type']?.split(';')[0].trim().toLowerCase() !== 'application/json') return reply(res, 415, { detail: 'Use application/json.' });
    if (training) return reply(res, 429, { detail: 'Another training job is running. Check saved forecasts before trying again.', retry_after: 30 }, 30);
    if (Number(req.headers['content-length']) > MAX_BODY_BYTES) return reply(res, 413, { detail: 'Request too large.' });
    let ownsSlot = false;
    try {
      const chunks = [];
      let bytes = 0;
      for await (const chunk of req) {
        bytes += chunk.length;
        if (bytes > MAX_BODY_BYTES) return reply(res, 413, { detail: 'Request too large.' });
        chunks.push(chunk);
      }
      const scope = parseScope(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      // Another request can start training while this request receives its body.
      if (training) return reply(res, 429, { detail: 'Another training job is running. Check saved forecasts before trying again.', retry_after: 30 }, 30);
      training = true;
      ownsSlot = true;
      return reply(res, 200, await train(scope));
    } catch (error) {
      if (error instanceof SyntaxError) return reply(res, 400, { detail: 'Invalid JSON request.' });
      if (error instanceof LearningError && error.status < 500) {
        return reply(res, error.status, { detail: error.message, retry_after: error.retryAfter }, error.retryAfter);
      }
      // Database errors may contain connection details. Keep those off the API.
      reportError(error);
      return reply(res, 503, { detail: 'Training service temporarily unavailable.', retry_after: 30 }, 30);
    } finally {
      if (ownsSlot) training = false;
    }
  });
}
