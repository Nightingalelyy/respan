export * from "./context.js";
export * from "./span.js";
export * from "./spanFactory.js";
export { logBatchResults } from "./batchLogging.js";
export type { BatchRequest, BatchResult } from "./batchLogging.js";

// Export tracing utils but avoid naming conflicts
export { startTracing, forceFlush, _resolveBaseURL } from "./tracing.js";

// Export client and span buffer
export * from "./client.js";
export * from "./spanBuffer.js";