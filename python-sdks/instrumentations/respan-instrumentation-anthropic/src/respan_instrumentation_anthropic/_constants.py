"""Anthropic instrumentation constants."""

from respan_sdk.constants.span_attributes import RESPAN_SESSION_ID

ANTHROPIC_INSTRUMENTATION_NAME = "anthropic"
ANTHROPIC_SYSTEM_NAME = "anthropic"
ANTHROPIC_CHAT_SPAN_NAME = "anthropic.chat"
ANTHROPIC_MANAGED_AGENT_SPAN_NAME = "anthropic.managed_agent"

ANTHROPIC_RESOURCES_MODULE = "anthropic.resources"
ANTHROPIC_BETA_SESSIONS_MODULE = "anthropic.resources.beta.sessions"
ASYNC_EVENTS_CLASS_NAME = "AsyncEvents"
ASYNC_MESSAGES_CLASS_NAME = "AsyncMessages"
CREATE_METHOD_NAME = "create"
EVENTS_CLASS_NAME = "Events"
MESSAGES_CLASS_NAME = "Messages"
STREAM_METHOD_NAME = "stream"
CLOSE_METHOD_NAME = "close"

ASSISTANT_ROLE = "assistant"
FUNCTION_KEY = "function"
FUNCTION_TOOL_TYPE = "function"
MCP_SERVER_KEY = "mcp_server"
MCP_SERVER_NAME_KEY = "mcp_server_name"
ROLE_KEY = "role"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
USER_ROLE = "user"

ARGUMENTS_KEY = "arguments"
CACHE_CONTROL_KEY = "cache_control"
CITATIONS_KEY = "citations"
CONTENT_KEY = "content"
DESCRIPTION_KEY = "description"
ID_KEY = "id"
INPUT_KEY = "input"
INPUT_SCHEMA_KEY = "input_schema"
INPUT_TOKENS_KEY = "input_tokens"
IS_ERROR_KEY = "is_error"
MODEL_KEY = "model"
MODEL_USAGE_KEY = "model_usage"
MESSAGES_KEY = "messages"
NAME_KEY = "name"
OUTPUT_TOKENS_KEY = "output_tokens"
PARAMETERS_KEY = "parameters"
PROPERTIES_KEY = "properties"
REQUIRED_KEY = "required"
SOURCE_KEY = "source"
STOP_REASON_KEY = "stop_reason"
SYSTEM_KEY = "system"
TEXT_KEY = "text"
TOOLS_KEY = "tools"
TOOL_CALL_ID_KEY = "tool_call_id"
PENDING_TOOL_CALL_KEY = "tool_call"
PENDING_TOOL_DEFINITION_KEY = "tool_definition"
PENDING_EXPIRES_AT_NS_KEY = "expires_at_ns"
PENDING_PARENT_ID_KEY = "parent_id"
PENDING_START_NS_KEY = "start_ns"
TOOL_USE_ID_KEY = "tool_use_id"
TYPE_KEY = "type"
UNIT_KEY = "unit"
USAGE_KEY = "usage"
CACHE_CREATION_INPUT_TOKENS_KEY = "cache_creation_input_tokens"
CACHE_READ_INPUT_TOKENS_KEY = "cache_read_input_tokens"

GEN_AI_COMPLETION_ROLE_ATTR = "gen_ai.completion.0.role"
GEN_AI_COMPLETION_TOOL_CALLS_ATTR = "gen_ai.completion.0.tool_calls"
GEN_AI_TOOL_CALL_ID_ATTR = "gen_ai.tool.call.id"
GEN_AI_TOOL_DEFINITIONS_ATTR = "gen_ai.tool.definitions"
MANAGED_AGENT_STOP_REASON_ATTR = "respan.managed_agent.stop_reason"
MANAGED_AGENT_SESSION_ID_ATTR = RESPAN_SESSION_ID
TOOL_CALLS_OVERRIDE = "tool_calls"
TOOLS_OVERRIDE = "tools"

GET_FINAL_MESSAGE_METHOD_NAME = "get_final_message"

TEXT_BLOCK_TYPE = "text"
TOOL_RESULT_BLOCK_TYPE = "tool_result"
TOOL_USE_BLOCK_TYPE = "tool_use"

AGENT_CUSTOM_TOOL_USE_EVENT = "agent.custom_tool_use"
AGENT_MCP_TOOL_USE_EVENT = "agent.mcp_tool_use"
AGENT_MESSAGE_EVENT = "agent.message"
AGENT_TOOL_USE_EVENT = "agent.tool_use"
MODEL_REQUEST_END_EVENT = "span.model_request_end"
SESSION_ERROR_EVENT = "session.error"
SESSION_STATUS_IDLE_EVENT = "session.status_idle"
SESSION_STATUS_RUNNING_EVENT = "session.status_running"
USER_MESSAGE_EVENT = "user.message"

AGENT_TOOL_USE_EVENTS = (
    AGENT_TOOL_USE_EVENT,
    AGENT_MCP_TOOL_USE_EVENT,
    AGENT_CUSTOM_TOOL_USE_EVENT,
)

SERIALIZED_CONTENT_FIELD_NAMES = (
    TYPE_KEY,
    ID_KEY,
    NAME_KEY,
    INPUT_KEY,
    TOOL_USE_ID_KEY,
    TEXT_KEY,
    CONTENT_KEY,
    SOURCE_KEY,
    CITATIONS_KEY,
    CACHE_CONTROL_KEY,
)
