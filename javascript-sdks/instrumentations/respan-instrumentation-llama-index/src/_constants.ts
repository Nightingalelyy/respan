import type { LogMethod } from "@respan/respan-sdk";

export const INSTRUMENTATION_LIBRARY_NAME =
  "@respan/instrumentation-llama-index";
export const PACKAGE_VERSION = "0.1.0";

export const RESPAN_LOG_METHOD_TS_TRACING: LogMethod = "ts_tracing";

export const LLAMA_INDEX_EVENTS = {
  LLM_START: "llm-start",
  LLM_END: "llm-end",
  LLM_TOOL_CALL: "llm-tool-call",
  LLM_TOOL_RESULT: "llm-tool-result",
  QUERY_START: "query-start",
  QUERY_END: "query-end",
  RETRIEVE_START: "retrieve-start",
  RETRIEVE_END: "retrieve-end",
  SYNTHESIZE_START: "synthesize-start",
  SYNTHESIZE_END: "synthesize-end",
  AGENT_START: "agent-start",
  AGENT_END: "agent-end",
  CHUNKING_START: "chunking-start",
  CHUNKING_END: "chunking-end",
  NODE_PARSING_START: "node-parsing-start",
  NODE_PARSING_END: "node-parsing-end",
} as const;

export type LlamaIndexEventName =
  (typeof LLAMA_INDEX_EVENTS)[keyof typeof LLAMA_INDEX_EVENTS];

export const MESSAGE_ROLE_SUFFIX = "role";
export const MESSAGE_CONTENT_SUFFIX = "content";
export const MESSAGE_TOOL_CALLS_SUFFIX = "tool_calls";
export const MESSAGE_TOOL_CALL_ID_SUFFIX = "tool_call_id";
