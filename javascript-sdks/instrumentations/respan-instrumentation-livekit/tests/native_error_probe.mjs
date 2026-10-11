import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import {
  MultiProcessorManager,
  RespanCompositeProcessor,
} from "@respan/tracing/dist/processor/index.js";
import * as LiveKit from "@livekit/agents";
import * as LiveKitOpenAI from "@livekit/agents-plugin-openai";
import { OpenAI } from "openai";
import { LiveKitInstrumentor } from "../dist/index.js";
LiveKit.initializeLogger({ pretty: false, level: "silent" });
const unhandled = [];
process.on("unhandledRejection", (error) => unhandled.push(error));
const exporter = new InMemorySpanExporter();
let processor = new SimpleSpanProcessor(exporter);
if (process.argv[2] === "instrumented") {
  const manager = new MultiProcessorManager({ disableBatch: true });
  manager.addProcessor({ exporter, name: "default", disableBatch: true });
  processor = new RespanCompositeProcessor(manager);
}
const provider = new BasicTracerProvider({ spanProcessors: [processor] });
LiveKit.telemetry.tracer.setProvider(provider);
const instrumentor = new LiveKitInstrumentor({
  livekitModule: LiveKit,
  syncTracerProvider: false,
});
if (process.argv[2] === "instrumented") await instrumentor.activate();
const client = new OpenAI({
  apiKey: "fixture-only",
  maxRetries: 0,
  fetch: async () =>
    new Response(
      JSON.stringify({
        error: {
          message: "controlled native transport error",
          type: "server_error",
        },
      }),
      { status: 400, headers: { "content-type": "application/json" } },
    ),
});
const model = new LiveKitOpenAI.LLM({ client, model: "native-model" });
const errors = [];
model.on("error", (error) => errors.push(error));
const stream = model.chat({
  chatCtx: LiveKit.llm.ChatContext.empty(),
  connOptions: { maxRetry: 0, retryIntervalMs: 0, timeoutMs: 1000 },
});
let chunks = 0;
for await (const _chunk of stream) chunks++;
stream.close();
await new Promise((resolve) => setTimeout(resolve, 30));
await provider.forceFlush();
const result = {
  chunks,
  errors: errors.map((item) => ({
    name: item.error.name,
    statusCode: item.error.statusCode,
    message: item.error.message,
  })),
  unhandled: unhandled.map((error) => ({
    name: error.name,
    statusCode: error.statusCode,
    message: error.message,
  })),
  completions: exporter
    .getFinishedSpans()
    .filter(
      (span) => span.attributes["gen_ai.completion.0.content"] !== undefined,
    ).length,
};
instrumentor.deactivate();
await provider.shutdown();
console.log(JSON.stringify(result));
