import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { VERCEL_PARENT_SPANS, VERCEL_SPAN_CONFIG } from "../constants/index.js";

export type SpanAttributes = Record<string, any>;

export const RESPAN_LOG_TYPE = RespanSpanAttributes.RESPAN_LOG_TYPE;
export const GEN_AI_REQUEST_MODEL = RespanSpanAttributes.GEN_AI_REQUEST_MODEL;
export const GEN_AI_USAGE_PROMPT_TOKENS = RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS;
export const GEN_AI_USAGE_COMPLETION_TOKENS = RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS;
export const LLM_REQUEST_TYPE = RespanSpanAttributes.LLM_REQUEST_TYPE;
export const CUSTOMER_ID = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID;
export const CUSTOMER_EMAIL = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL;
export const CUSTOMER_NAME = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME;
export const THREAD_ID = RespanSpanAttributes.RESPAN_THREADS_ID;
export const SESSION_ID = "respan.sessions.session_identifier";
export const TRACE_GROUP_ID = RespanSpanAttributes.RESPAN_TRACE_GROUP_ID;
export const RESPAN_SPAN_TOOLS = RespanSpanAttributes.RESPAN_SPAN_TOOLS;
export const RESPAN_SPAN_TOOL_CALLS = RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS;
export const RESPAN_METADATA_AGENT_NAME = RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME;
export const RESPAN_METADATA_PREFIX = RespanSpanAttributes.RESPAN_METADATA;

export const TL_SPAN_KIND = "traceloop.span.kind";
export const TL_ENTITY_INPUT = "traceloop.entity.input";
export const TL_ENTITY_OUTPUT = "traceloop.entity.output";
export const TL_REQUEST_FUNCTIONS = "llm.request.functions";

export const AI_PREFIX = "ai.";
export const AI_SDK = "ai.sdk";
export const AI_OPERATION_ID = "ai.operationId";
export const AI_MODEL_ID = "ai.model.id";
export const AI_EMBEDDING = "ai.embedding";
export const AI_EMBEDDINGS = "ai.embeddings";
export const AI_AGENT_ID = "ai.agent.id";
export const AI_WORKFLOW_ID = "ai.workflow.id";
export const AI_TRANSCRIPT = "ai.transcript";
export const AI_SPEECH = "ai.speech";
export const AI_TELEMETRY_METADATA_PREFIX = "ai.telemetry.metadata.";
export const AI_PROMPT = "ai.prompt";
export const AI_PROMPT_MESSAGES = "ai.prompt.messages";
export const AI_PROMPT_TOOLS = "ai.prompt.tools";
export const AI_PROMPT_TOOL_CHOICE = "ai.prompt.toolChoice";
export const AI_RESPONSE_OBJECT = "ai.response.object";
export const AI_RESPONSE_TEXT = "ai.response.text";
export const AI_RESPONSE_TOOL_CALLS = "ai.response.toolCalls";
export const AI_RESPONSE_MS_TO_FINISH = "ai.response.msToFinish";
export const AI_TOOL_CALL = "ai.toolCall";
export const AI_TOOL_CALLS = "ai.toolCalls";
export const AI_TOOL_CALL_PREFIX = "ai.toolCall.";
export const AI_TOOL_CALL_ID = "ai.toolCall.id";
export const AI_TOOL_CALL_NAME = "ai.toolCall.name";
export const AI_TOOL_CALL_ARGS = "ai.toolCall.args";
export const AI_TOOL_CALL_RESULT = "ai.toolCall.result";
export const GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens";
export const GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens";
export const GEN_AI_USAGE_COST = "gen_ai.usage.cost";
export const GEN_AI_USAGE_TTFT = "gen_ai.usage.ttft";
export const GEN_AI_USAGE_GENERATION_TIME = "gen_ai.usage.generation_time";
export const GEN_AI_USAGE_WARNINGS = "gen_ai.usage.warnings";
export const GEN_AI_USAGE_TYPE = "gen_ai.usage.type";

export function metadataKey(key: string): string {
  return `${RESPAN_METADATA_PREFIX}.${key}`;
}

export function setDefault(attrs: SpanAttributes, key: string, value: any): void {
  if (attrs[key] === undefined && value !== undefined && value !== null) {
    attrs[key] = value;
  }
}

export function safeJsonStr(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function safeJsonParse(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function isVercelAISpan(span: ReadableSpan): boolean {
  if (span.instrumentationLibrary?.name === "ai") {
    return true;
  }
  if (span.attributes[AI_SDK] !== undefined) {
    return true;
  }
  return span.name.startsWith(AI_PREFIX);
}

export function resolveLogType(name: string, attrs: SpanAttributes): string {
  const config = VERCEL_SPAN_CONFIG[name];
  if (config) {
    return config.logType;
  }

  const parentType = VERCEL_PARENT_SPANS[name];
  if (parentType) {
    return parentType;
  }

  const operationId = attrs[AI_OPERATION_ID]?.toString();
  if (operationId) {
    const operationConfig = VERCEL_SPAN_CONFIG[operationId];
    if (operationConfig) {
      return operationConfig.logType;
    }

    const operationParentType = VERCEL_PARENT_SPANS[operationId];
    if (operationParentType) {
      return operationParentType;
    }
  }

  if (
    attrs[AI_EMBEDDING] || attrs[AI_EMBEDDINGS] ||
    name.includes("embed") || operationId?.includes("embed")
  ) {
    return RespanLogType.EMBEDDING;
  }

  if (
    attrs[AI_TOOL_CALL_ID] || attrs[AI_TOOL_CALL_NAME] ||
    attrs[AI_TOOL_CALL_ARGS] || attrs[AI_TOOL_CALL_RESULT] ||
    attrs[AI_RESPONSE_TOOL_CALLS] ||
    name.includes("tool") || operationId?.includes("tool")
  ) {
    return RespanLogType.TOOL;
  }

  if (
    attrs[AI_AGENT_ID] ||
    name.includes("agent") || operationId?.includes("agent")
  ) {
    return RespanLogType.AGENT;
  }

  if (
    attrs[AI_WORKFLOW_ID] ||
    name.includes("workflow") || operationId?.includes("workflow")
  ) {
    return RespanLogType.WORKFLOW;
  }

  if (
    attrs[AI_TRANSCRIPT] ||
    name.includes("transcript") || operationId?.includes("transcript")
  ) {
    return RespanLogType.TRANSCRIPTION;
  }

  if (
    attrs[AI_SPEECH] ||
    name.includes("speech") || operationId?.includes("speech")
  ) {
    return RespanLogType.SPEECH;
  }

  if (name.includes("doGenerate") || name.includes("doStream")) {
    return RespanLogType.TEXT;
  }

  return RespanLogType.UNKNOWN;
}

export function normalizeModel(modelId: string): string {
  const model = modelId.toLowerCase();

  if (model.includes("gemini-2.0-flash-001")) return "gemini/gemini-2.0-flash";
  if (model.includes("gemini-2.0-pro")) return "gemini/gemini-2.0-pro-exp-02-05";
  if (model.includes("claude-3-5-sonnet")) return "claude-3-5-sonnet-20241022";
  if (model.includes("deepseek")) return `deepseek/${model}`;
  if (model.includes("o3-mini")) return "o3-mini";

  return model;
}
