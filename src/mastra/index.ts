import { Mastra } from '@mastra/core/mastra';
import { PinoLogger } from '@mastra/loggers';
import { weatherWorkflow, webSearchWorkflow } from './workflows';
import { searchAgent } from './agents';

export const mastra = new Mastra({
  workflows: { weatherWorkflow, webSearchWorkflow },
  logger: new PinoLogger({
    name: 'Mastra',
    level: 'debug', // Enable verbose logging
  }),
});
