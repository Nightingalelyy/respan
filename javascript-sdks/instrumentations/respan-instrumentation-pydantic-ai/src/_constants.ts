import {
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  AGENT_NAME,
  INPUT_VALUE,
  LLM_INVOCATION_PARAMETERS,
  LLM_MODEL_NAME,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_TOTAL,
  LLM_TOOLS,
  OUTPUT_VALUE,
  SemanticConventions as OpenInferenceSemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes as TraceloopSpanAttributes } from "@traceloop/ai-semantic-conventions";

// Traceloop / GenAI semantic convention keys.
export const TRACELOOP_ENTITY_NAME =
  TraceloopSpanAttributes.TRACELOOP_ENTITY_NAME;
export const TRACELOOP_ENTITY_PATH =
  TraceloopSpanAttributes.TRACELOOP_ENTITY_PATH;
export const TRACELOOP_ENTITY_INPUT =
  TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT;
export const TRACELOOP_ENTITY_OUTPUT =
  TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT;
export const TRACELOOP_WORKFLOW_NAME =
  TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME;
export const TRACELOOP_SPAN_KIND =
  TraceloopSpanAttributes.TRACELOOP_SPAN_KIND;
export const LLM_REQUEST_TYPE = TraceloopSpanAttributes.LLM_REQUEST_TYPE;
export const LLM_REQUEST_FUNCTIONS =
  TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS;
export const LLM_USAGE_TOTAL_TOKENS =
  TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS;
export const LLM_USAGE_CACHE_READ_INPUT_TOKENS =
  "llm.usage.cache_read_input_tokens";
export const GEN_AI_SYSTEM = ATTR_GEN_AI_SYSTEM;
export const GEN_AI_REQUEST_MODEL = ATTR_GEN_AI_REQUEST_MODEL;
export const GEN_AI_PROMPT_PREFIX = TraceloopSpanAttributes.LLM_PROMPTS;
export const GEN_AI_COMPLETION_PREFIX =
  TraceloopSpanAttributes.LLM_COMPLETIONS;
export const GEN_AI_USAGE_PROMPT_TOKENS =
  TraceloopSpanAttributes.LLM_USAGE_PROMPT_TOKENS;
export const GEN_AI_USAGE_COMPLETION_TOKENS =
  TraceloopSpanAttributes.LLM_USAGE_COMPLETION_TOKENS;
export const GEN_AI_USAGE_INPUT_TOKENS = ATTR_GEN_AI_USAGE_INPUT_TOKENS;
export const GEN_AI_USAGE_OUTPUT_TOKENS = ATTR_GEN_AI_USAGE_OUTPUT_TOKENS;
export const GEN_AI_PROVIDER_NAME = ATTR_GEN_AI_PROVIDER_NAME;

// Respan-owned keys.
export const RESPAN_LOG_TYPE = RespanSpanAttributes.RESPAN_LOG_TYPE;
export const RESPAN_LOG_METHOD = RespanSpanAttributes.RESPAN_LOG_METHOD;

// OpenInference semantic convention keys.
export const OI_SPAN_KIND =
  OpenInferenceSemanticConventions.OPENINFERENCE_SPAN_KIND;
export const OI_INPUT_VALUE = INPUT_VALUE;
export const OI_OUTPUT_VALUE = OUTPUT_VALUE;
export const OI_LLM_MODEL_NAME = LLM_MODEL_NAME;
export const OI_LLM_PROVIDER = LLM_PROVIDER;
export const OI_LLM_SYSTEM = LLM_SYSTEM;
export const OI_LLM_INVOCATION_PARAMETERS = LLM_INVOCATION_PARAMETERS;
export const OI_LLM_TOKEN_COUNT_PROMPT = LLM_TOKEN_COUNT_PROMPT;
export const OI_LLM_TOKEN_COUNT_COMPLETION = LLM_TOKEN_COUNT_COMPLETION;
export const OI_LLM_TOKEN_COUNT_TOTAL = LLM_TOKEN_COUNT_TOTAL;
export const OI_LLM_TOKEN_COUNT_CACHE_READ =
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ;
export const OI_LLM_TOOLS = LLM_TOOLS;
export const OI_AGENT_NAME = AGENT_NAME;

// Pydantic AI native / SDK-specific raw attributes.
export const PYDANTIC_AI_REQUEST_PARAMETERS = "model_request_parameters";
export const PYDANTIC_AI_TOOL_DEFINITIONS = ATTR_GEN_AI_TOOL_DEFINITIONS;
export const PYDANTIC_AI_INPUT_MESSAGES = ATTR_GEN_AI_INPUT_MESSAGES;
export const PYDANTIC_AI_OUTPUT_MESSAGES = ATTR_GEN_AI_OUTPUT_MESSAGES;
export const PYDANTIC_AI_AGENT_NAME = ATTR_GEN_AI_AGENT_NAME;
export const PYDANTIC_AI_TOOL_NAME = ATTR_GEN_AI_TOOL_NAME;
export const PYDANTIC_AI_TOOL_CALL_ID = ATTR_GEN_AI_TOOL_CALL_ID;
export const PYDANTIC_AI_TOOL_CALL_ARGUMENTS =
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS;
export const PYDANTIC_AI_TOOL_CALL_RESULT = ATTR_GEN_AI_TOOL_CALL_RESULT;
export const PYDANTIC_AI_OPERATION_NAME = ATTR_GEN_AI_OPERATION_NAME;
export const PYDANTIC_AI_RESPONSE_ID = ATTR_GEN_AI_RESPONSE_ID;
export const PYDANTIC_AI_RESPONSE_FINISH_REASONS =
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS;
export const PYDANTIC_AI_OPENAI_API_BASE = "gen_ai.openai.api_base";
export const PYDANTIC_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens";
export const PYDANTIC_AI_USAGE_DETAILS_INPUT_TOKENS =
  "gen_ai.usage.details.input_tokens";
export const PYDANTIC_AI_USAGE_DETAILS_OUTPUT_TOKENS =
  "gen_ai.usage.details.output_tokens";
export const PYDANTIC_AI_OPERATION_COST = "operation.cost";
export const PYDANTIC_AI_LEGACY_AGENT_NAME = "agent_name";
export const PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS = "tool_arguments";
export const PYDANTIC_AI_LEGACY_TOOL_RESULT = "tool_response";
export const PYDANTIC_AI_TOOLS = "tools";
export const PYDANTIC_AI_LOGFIRE_MESSAGE = "logfire.msg";
export const PYDANTIC_AI_MODEL_NAME = "model_name";
export const PYDANTIC_AI_FINAL_RESULT = "final_result";
export const PYDANTIC_AI_ALL_MESSAGES = "pydantic_ai.all_messages";
export const PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME = "running tools";

export const PYDANTIC_AI_SCOPE_MARKERS = [
  "pydantic-ai",
  "pydantic_ai",
  "pydantic.ai",
];

export const PYDANTIC_AI_RAW_EXACT_ATTRS = new Set([
  PYDANTIC_AI_REQUEST_PARAMETERS,
  PYDANTIC_AI_TOOL_DEFINITIONS,
  PYDANTIC_AI_INPUT_MESSAGES,
  PYDANTIC_AI_OUTPUT_MESSAGES,
  PYDANTIC_AI_AGENT_NAME,
  PYDANTIC_AI_TOOL_NAME,
  PYDANTIC_AI_TOOL_CALL_ID,
  PYDANTIC_AI_TOOL_CALL_ARGUMENTS,
  PYDANTIC_AI_TOOL_CALL_RESULT,
  PYDANTIC_AI_OPERATION_NAME,
  PYDANTIC_AI_RESPONSE_ID,
  PYDANTIC_AI_RESPONSE_FINISH_REASONS,
  PYDANTIC_AI_OPENAI_API_BASE,
  PYDANTIC_AI_USAGE_DETAILS_INPUT_TOKENS,
  PYDANTIC_AI_USAGE_DETAILS_OUTPUT_TOKENS,
  PYDANTIC_AI_OPERATION_COST,
  PYDANTIC_AI_LEGACY_AGENT_NAME,
  PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS,
  PYDANTIC_AI_LEGACY_TOOL_RESULT,
  PYDANTIC_AI_TOOLS,
  PYDANTIC_AI_LOGFIRE_MESSAGE,
  PYDANTIC_AI_MODEL_NAME,
  PYDANTIC_AI_FINAL_RESULT,
  PYDANTIC_AI_ALL_MESSAGES,
]);

export const OI_RAW_EXACT_ATTRS = new Set([
  OI_SPAN_KIND,
  OI_INPUT_VALUE,
  "input.mime_type",
  OI_OUTPUT_VALUE,
  "output.mime_type",
  OI_LLM_MODEL_NAME,
  OI_LLM_PROVIDER,
  OI_LLM_SYSTEM,
  OI_LLM_INVOCATION_PARAMETERS,
  OI_LLM_TOKEN_COUNT_PROMPT,
  OI_LLM_TOKEN_COUNT_COMPLETION,
  OI_LLM_TOKEN_COUNT_TOTAL,
  OI_LLM_TOKEN_COUNT_CACHE_READ,
  OI_LLM_TOOLS,
  OI_AGENT_NAME,
]);

export const OI_RAW_PREFIXES = [
  "llm.input_messages.",
  "llm.output_messages.",
  "llm.token_count.",
];

export const OTEL_NOISE_EXACT_ATTRS = new Set([
  "otel.scope.name",
  "otel.scope.version",
]);

export const OTEL_NOISE_PREFIXES = [
  "host.",
  "process.",
  "telemetry.sdk.",
];

export const OFF_CONTRACT_ALIAS_ATTRS = new Set([
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "span_tools",
  "tool_calls",
  "tools",
  "has_tool_calls",
  "parallel_tool_calls",
  RespanSpanAttributes.RESPAN_SPAN_TOOLS,
  RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
]);
