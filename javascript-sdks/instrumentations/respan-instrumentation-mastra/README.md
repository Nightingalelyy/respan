# @respan/instrumentation-mastra

Respan instrumentation for Mastra TypeScript observability. The package exposes a Mastra observability exporter that converts Mastra `span_ended` events into Respan OTEL spans and injects them into the active `@respan/tracing` pipeline.

```ts
import { createOpenAI } from "@ai-sdk/openai";
import { Agent } from "@mastra/core/agent";
import { Mastra } from "@mastra/core/mastra";
import { SpanType } from "@mastra/core/observability";
import { Observability, SamplingStrategyType } from "@mastra/observability";
import { MastraInstrumentor } from "@respan/instrumentation-mastra";
import { Respan } from "@respan/respan";

const mastraInstrumentor = new MastraInstrumentor();
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [mastraInstrumentor],
});
await respan.initialize();

const openai = createOpenAI({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL ?? "https://api.respan.ai/api",
});

const assistantAgent = new Agent({
  id: "assistant-agent",
  name: "Assistant Agent",
  instructions: "You are helpful.",
  model: openai("gpt-4.1-nano"),
});

const mastra = new Mastra({
  agents: { assistantAgent },
  observability: new Observability({
    configs: {
      default: {
        serviceName: "mastra-app",
        sampling: { type: SamplingStrategyType.ALWAYS },
        exporters: [mastraInstrumentor],
        excludeSpanTypes: [
          SpanType.MODEL_CHUNK,
          SpanType.MODEL_STEP,
          SpanType.MODEL_INFERENCE,
        ],
      },
    },
    sensitiveDataFilter: false,
  }),
});

try {
  const result = await respan.withWorkflow({ name: "mastra_assistant" }, async () => {
    const agent = mastra.getAgent("assistantAgent");
    return agent.generate("Hello from Mastra");
  });
  console.log(result.text);
} finally {
  await respan.flush();
}
```
