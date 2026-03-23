// Core
export { Respan } from "./_core.js";
export type { RespanOptions } from "./_core.js";

// Plugin protocol
export type { RespanInstrumentation } from "./_types.js";

// Instrumentor wrappers
export { OTELInstrumentor } from "./_otel_instrumentor.js";
export { OpenInferenceInstrumentor } from "./_openinference_instrumentor.js";

// Re-export decorators and utilities from @respan/tracing
export {
  withWorkflow,
  withTask,
  withAgent,
  withTool,
  propagateAttributes,
  buildReadableSpan,
  injectSpan,
} from "@respan/tracing";
export { getClient, getSpanBufferManager } from "@respan/tracing";
export type { ProcessorConfig } from "@respan/tracing";
