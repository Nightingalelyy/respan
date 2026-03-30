"""Span attribute key constants.

Single source of truth for all ``respan.*`` attribute keys.  The
``RespanSpanAttributes`` enum and flat convenience aliases both live here
(per BE convention: enums ARE constants, always in constants.py).

For attribute VALUES (e.g. log type strings like "chat", "workflow"),
see ``llm_logging.py``.
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Respan span attribute enum (single source of truth)
# ---------------------------------------------------------------------------


class RespanSpanAttributes(str, Enum):
    """Respan span attribute key constants.

    All ``respan.*`` attribute keys used across the pipeline.  The backend
    uses ``RespanSpanAttributes.X.value``; SDK code uses the flat constant
    aliases defined below.
    """

    # Span params
    RESPAN_SPAN_CUSTOM_ID = "respan.span_params.custom_identifier"

    # Customer params
    RESPAN_CUSTOMER_PARAMS_ID = "respan.customer_params.customer_identifier"
    RESPAN_CUSTOMER_PARAMS_EMAIL = "respan.customer_params.email"
    RESPAN_CUSTOMER_PARAMS_NAME = "respan.customer_params.name"

    # Evaluation params
    RESPAN_EVALUATION_PARAMS_ID = "respan.evaluation_params.evaluation_identifier"

    # Threads
    RESPAN_THREADS_ID = "respan.threads.thread_identifier"

    # Sessions
    RESPAN_SESSION_ID = "respan.sessions.session_identifier"

    # Trace
    RESPAN_TRACE_GROUP_ID = "respan.trace.trace_group_identifier"

    # Metadata
    RESPAN_METADATA = "respan.metadata"
    RESPAN_PROPERTIES = "respan.properties"

    # Prompt & environment
    RESPAN_PROMPT = "respan.prompt"
    RESPAN_ENVIRONMENT = "respan.environment"

    # Span links
    RESPAN_LINK_TIMESTAMP = "respan.link.timestamp"

    # Logging
    LOG_METHOD = "respan.entity.log_method"
    LOG_TYPE = "respan.entity.log_type"
    LOG_ID = "respan.entity.log_id"
    LOG_PARENT_ID = "respan.entity.log_parent_id"
    LOG_ROOT_ID = "respan.entity.log_root_id"
    LOG_SOURCE = "respan.entity.log_source"


# ---------------------------------------------------------------------------
# Flat convenience aliases (avoid .value boilerplate in SDK code)
# ---------------------------------------------------------------------------

# Span params
RESPAN_SPAN_CUSTOM_ID = RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID.value

# Customer params
RESPAN_CUSTOMER_PARAMS_ID = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID.value
RESPAN_CUSTOMER_PARAMS_EMAIL = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL.value
RESPAN_CUSTOMER_PARAMS_NAME = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME.value

# Evaluation params
RESPAN_EVALUATION_PARAMS_ID = RespanSpanAttributes.RESPAN_EVALUATION_PARAMS_ID.value

# Threads
RESPAN_THREADS_ID = RespanSpanAttributes.RESPAN_THREADS_ID.value

# Sessions
RESPAN_SESSION_ID = RespanSpanAttributes.RESPAN_SESSION_ID.value

# Trace
RESPAN_TRACE_GROUP_ID = RespanSpanAttributes.RESPAN_TRACE_GROUP_ID.value

# Metadata (pattern: "respan.metadata.<key>" where key is customizable)
RESPAN_METADATA = RespanSpanAttributes.RESPAN_METADATA.value
RESPAN_PROPERTIES = RespanSpanAttributes.RESPAN_PROPERTIES.value

# Prompt & environment
RESPAN_PROMPT = RespanSpanAttributes.RESPAN_PROMPT.value
RESPAN_ENVIRONMENT = RespanSpanAttributes.RESPAN_ENVIRONMENT.value

# Span links
RESPAN_LINK_TIMESTAMP = RespanSpanAttributes.RESPAN_LINK_TIMESTAMP.value

# Logging
RESPAN_LOG_METHOD = RespanSpanAttributes.LOG_METHOD.value
RESPAN_LOG_TYPE = RespanSpanAttributes.LOG_TYPE.value
RESPAN_LOG_ID = RespanSpanAttributes.LOG_ID.value
RESPAN_LOG_PARENT_ID = RespanSpanAttributes.LOG_PARENT_ID.value
RESPAN_LOG_ROOT_ID = RespanSpanAttributes.LOG_ROOT_ID.value
RESPAN_LOG_SOURCE = RespanSpanAttributes.LOG_SOURCE.value

# ---------------------------------------------------------------------------
# Mapping: user-facing kwargs → span attribute keys
# Used by propagate_attributes(), RespanClient, and RespanParams processing.
# ---------------------------------------------------------------------------
RESPAN_SPAN_ATTRIBUTES_MAP = {
    "customer_identifier": RESPAN_CUSTOMER_PARAMS_ID,
    "customer_email": RESPAN_CUSTOMER_PARAMS_EMAIL,
    "customer_name": RESPAN_CUSTOMER_PARAMS_NAME,
    "evaluation_identifier": RESPAN_EVALUATION_PARAMS_ID,
    "thread_identifier": RESPAN_THREADS_ID,
    "session_identifier": RESPAN_SESSION_ID,
    "custom_identifier": RESPAN_SPAN_CUSTOM_ID,
    "trace_group_identifier": RESPAN_TRACE_GROUP_ID,
    "group_identifier": RESPAN_TRACE_GROUP_ID,
    "metadata": RESPAN_METADATA,
    "properties": RESPAN_PROPERTIES,
    "prompt": RESPAN_PROMPT,
    "environment": RESPAN_ENVIRONMENT,
}

# ---------------------------------------------------------------------------
# Respan metadata attributes (agent-specific)
# ---------------------------------------------------------------------------
RESPAN_METADATA_AGENT_NAME = "respan.metadata.agent_name"
RESPAN_METADATA_FROM_AGENT = "respan.metadata.from_agent"
RESPAN_METADATA_TO_AGENT = "respan.metadata.to_agent"
RESPAN_METADATA_GUARDRAIL_NAME = "respan.metadata.guardrail_name"
RESPAN_METADATA_TRIGGERED = "respan.metadata.triggered"
RESPAN_SPAN_TOOLS = "respan.span.tools"
RESPAN_SPAN_TOOL_CALLS = "respan.span.tool_calls"
RESPAN_SPAN_HANDOFFS = "respan.span.handoffs"

# ---------------------------------------------------------------------------
# LLM attributes (removed from opentelemetry-semantic-conventions-ai 0.5.0)
# Defined locally for backward compatibility with the Respan pipeline.
# ---------------------------------------------------------------------------
LLM_REQUEST_TYPE = "llm.request.type"
LLM_REQUEST_MODEL = "gen_ai.request.model"
LLM_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
LLM_REQUEST_TOP_P = "gen_ai.request.top_p"
LLM_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
LLM_REQUEST_FUNCTIONS = "llm.request.functions"
LLM_REQUEST_REPETITION_PENALTY = "llm.request.repetition_penalty"
LLM_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
LLM_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens"
LLM_REQUEST_REASONING_EFFORT = "llm.request.reasoning_effort"
LLM_TOP_K = "llm.top_k"
LLM_CHAT_STOP_SEQUENCES = "llm.chat.stop_sequences"
LLM_FREQUENCY_PENALTY = "llm.frequency_penalty"
LLM_PRESENCE_PENALTY = "llm.presence_penalty"

# ---------------------------------------------------------------------------
# OTEL incubating GenAI attributes
# Not yet available in opentelemetry.semconv_ai.SpanAttributes — defined
# locally until the upstream package includes them.
# ---------------------------------------------------------------------------
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

# ---------------------------------------------------------------------------
# Pydantic AI vendor attributes (non-standard)
# ---------------------------------------------------------------------------
PYDANTIC_AI_AGENT_NAME = "agent_name"
PYDANTIC_AI_TOOL_ARGUMENTS = "tool_arguments"
PYDANTIC_AI_TOOL_RESPONSE = "tool_response"

# ---------------------------------------------------------------------------
# OpenInference vendor attributes
# Only OPENINFERENCE_SPAN_KIND is here because respan-tracing uses it
# for span filtering without depending on the openinference package.
# Missing/floor-mismatched OI attributes are defined here as string fallbacks.
# ---------------------------------------------------------------------------
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
OPENINFERENCE_INPUT_MIME_TYPE = "input.mime_type"
OPENINFERENCE_OUTPUT_MIME_TYPE = "output.mime_type"
OPENINFERENCE_AGENT_NAME = "agent.name"
OPENINFERENCE_LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = (
    "llm.token_count.prompt_details.cache_read"
)
OPENINFERENCE_INPUT_MESSAGES_PREFIX = "llm.input_messages."
OPENINFERENCE_OUTPUT_MESSAGES_PREFIX = "llm.output_messages."
OPENINFERENCE_TOKEN_COUNT_PREFIX = "llm.token_count."
OPENINFERENCE_TOOLS_PREFIX = "llm.tools."
OPENINFERENCE_MESSAGE_ROLE = "message.role"
OPENINFERENCE_MESSAGE_CONTENT = "message.content"
OPENINFERENCE_MESSAGE_CONTENT_PREFIX = "message.content."
OPENINFERENCE_MESSAGE_TOOL_CALLS_PREFIX = "message.tool_calls."
OPENINFERENCE_MESSAGE_FUNCTION_CALL_NAME = "message.function_call_name"
OPENINFERENCE_MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = "message.function_call_arguments_json"
OPENINFERENCE_MESSAGE_FINISH_REASON = "message.finish_reason"
OPENINFERENCE_TOOL_PREFIX = "tool."
OPENINFERENCE_TOOL_JSON_SCHEMA = "tool.json_schema"
OPENINFERENCE_TOOL_CALL_PREFIX = "tool_call."
