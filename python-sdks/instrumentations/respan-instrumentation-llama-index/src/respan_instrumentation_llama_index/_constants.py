"""LlamaIndex instrumentation-local constants."""

from __future__ import annotations

CHAT_EVENT_KEY = "chat"
COMPLETION_EVENT_KEY = "completion"
EMBEDDING_EVENT_KEY = "embedding"

LLAMA_INDEX_INSTRUMENTATION_NAME = "llama-index"
LLAMA_INDEX_ROOT_MODULE = "llama_index.core.instrumentation"

GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
