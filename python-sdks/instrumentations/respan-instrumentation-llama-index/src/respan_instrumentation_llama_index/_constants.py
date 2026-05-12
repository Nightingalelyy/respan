"""LlamaIndex instrumentation-local constants."""

from __future__ import annotations

CHAT_EVENT_KEY = "chat"
COMPLETION_EVENT_KEY = "completion"
EMBEDDING_EVENT_KEY = "embedding"

LLAMA_INDEX_INSTRUMENTATION_NAME = "llama-index"
LLAMA_INDEX_ROOT_MODULE = "llama_index.core.instrumentation"

LLAMA_INDEX_CHAT_SPAN_NAME = "llama_index.chat"
LLAMA_INDEX_COMPLETION_SPAN_NAME = "llama_index.completion"
LLAMA_INDEX_EMBEDDING_SPAN_NAME = "llama_index.embedding"
LLAMA_INDEX_TOOL_SPAN_PREFIX = "llama_index.tool."
LLAMA_INDEX_DEFAULT_TOOL_NAME = "llama_index_tool"

LLAMA_INDEX_SPAN_ID_ATTR = "llama_index.span_id"
LLAMA_INDEX_PARENT_SPAN_ID_ATTR = "llama_index.parent_span_id"
LLAMA_INDEX_SYNTHETIC_PARENT_ATTR = "llama_index.synthetic_parent"
LLAMA_INDEX_TAGS_ATTR = "llama_index.tags"

MESSAGE_ROLE_ASSISTANT = "assistant"
MESSAGE_ROLE_SYSTEM = "system"
MESSAGE_ROLE_USER = "user"

# Not available in the installed Traceloop GenAI semantic-convention package.
LLAMA_INDEX_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
LLAMA_INDEX_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
LLM_EMBEDDINGS_0 = "llm.embeddings.0"

# Backend override keys used by Respan ingestion for provider-reported usage.
RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR = "prompt_tokens"
RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR = "completion_tokens"
RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR = "total_request_tokens"
