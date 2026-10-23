import { MongoClient } from 'mongodb';
import { createTrainingServer } from './render-http.mjs';
import { parseLearningScope, runLearning, LearningError } from './dist/render-learning.mjs';

const { MONGODB_URI, MONGODB_DATABASE = 'cuda_research', RENDER_TRAINING_KEY, PORT = '10000' } = process.env;
if (!MONGODB_URI || !RENDER_TRAINING_KEY || RENDER_TRAINING_KEY.length < 32) throw new Error('MongoDB and a training key of at least 32 characters are required.');
if (!/^\d+$/.test(PORT) || Number(PORT) < 1 || Number(PORT) > 65535) throw new Error('PORT must be between 1 and 65535.');
const client = new MongoClient(MONGODB_URI, { maxPoolSize: 4, serverSelectionTimeoutMS: 12000 });
const server = createTrainingServer({
  trainingKey: RENDER_TRAINING_KEY,
  parseScope: parseLearningScope,
  LearningError,
  train: async scope => {
    await client.connect();
    return runLearning(client.db(MONGODB_DATABASE), scope);
  },
});
server.listen(Number(PORT), '0.0.0.0', () => console.info(`Forecast service listening on port ${PORT}.`));

for (const signal of ['SIGTERM', 'SIGINT']) {
  process.once(signal, () => server.close(() => { client.close().catch(() => { process.exitCode = 1; }); }));
}
