"""AgentOps-owned attribute keys and translation tables."""

from __future__ import annotations

AGENTOPS_INSTRUMENTATION_NAME = "agentops"
AGENTOPS_SCOPE_PREFIX = "agentops"

AGENTOPS_ENTITY_INPUT = "agentops.entity.input"
AGENTOPS_ENTITY_OUTPUT = "agentops.entity.output"
AGENTOPS_SPAN_KIND = "agentops.span.kind"
AGENTOPS_ENTITY_NAME = "agentops.entity.name"
AGENTOPS_SESSION_END_STATE = "agentops.session.end_state"
AGENTOPS_TAGS = "agentops.tags"
AGENTOPS_DECORATOR_INPUT_TEMPLATE = "agentops.{kind}.input"
AGENTOPS_DECORATOR_OUTPUT_TEMPLATE = "agentops.{kind}.output"

AGENT_NAME = "agent.name"
TOOL_NAME = "tool.name"
OPERATION_NAME = "operation.name"
OPERATION_VERSION = "operation.version"

AGENTOPS_REQUEST_TYPE = "gen_ai.request.type"
AGENTOPS_REQUEST_FUNCTIONS = "gen_ai.request.functions"
AGENTOPS_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

AGENTOPS_KIND_LOG_TYPES = {
    "session": "workflow",
    "workflow": "workflow",
    "agent": "agent",
    "task": "task",
    "operation": "task",
    "chain": "task",
    "tool": "tool",
    "guardrail": "guardrail",
    "http": "task",
    "llm": "chat",
    "text": "text",
}

OFF_CONTRACT_ALIASES = {
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
}
