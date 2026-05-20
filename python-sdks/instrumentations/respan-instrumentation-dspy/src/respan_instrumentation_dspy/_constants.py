"""Constants for DSPy instrumentation."""

from opentelemetry.semconv_ai import SpanAttributes

DSPY_INSTRUMENTATION_NAME = "dspy"

DSPY_CALL_KIND_ADAPTER_FORMAT = "adapter_format"
DSPY_CALL_KIND_ADAPTER_PARSE = "adapter_parse"
DSPY_CALL_KIND_EVALUATE = "evaluate"
DSPY_CALL_KIND_LANGUAGE_MODEL = "lm"
DSPY_CALL_KIND_MODULE = "module"
DSPY_CALL_KIND_TOOL = "tool"

DSPY_LANGUAGE_MODEL_SPAN_NAME = "dspy.lm"
DSPY_MODULE_SPAN_NAME = "dspy.module"
DSPY_TOOL_SPAN_NAME = "dspy.tool"
DSPY_ADAPTER_SPAN_NAME = "dspy.adapter"
DSPY_EVALUATE_SPAN_NAME = "dspy.evaluate"

DSPY_PROVIDER_NAME = "dspy"
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"

CHAT_MODEL_TYPE = "chat"
TEXT_MODEL_TYPE = "text"
RESPONSES_MODEL_TYPE = "responses"

DSPY_USAGE_INPUT_TOKENS_ATTR = getattr(
    SpanAttributes,
    "GEN_AI_USAGE_INPUT_TOKENS",
    "gen_ai.usage.input_tokens",
)
DSPY_USAGE_OUTPUT_TOKENS_ATTR = getattr(
    SpanAttributes,
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "gen_ai.usage.output_tokens",
)

PROMPT_TOKENS_KEY = "prompt_tokens"
COMPLETION_TOKENS_KEY = "completion_tokens"
INPUT_TOKENS_KEY = "input_tokens"
OUTPUT_TOKENS_KEY = "output_tokens"
TOTAL_TOKENS_KEY = "total_tokens"

OPENAI_PROVIDER_PREFIX = "openai"
ANTHROPIC_PROVIDER_PREFIX = "anthropic"
GOOGLE_PROVIDER_PREFIX = "google"
GEMINI_PROVIDER_PREFIX = "gemini"
BEDROCK_PROVIDER_PREFIX = "bedrock"
AZURE_PROVIDER_PREFIX = "azure"
OLLAMA_PROVIDER_PREFIX = "ollama"
