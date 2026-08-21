"""OpenRouter instrumentation-local constants."""

OPENROUTER_INSTRUMENTATION_NAME = "openrouter"
OPENROUTER_INSTRUMENTATION_SCOPE = "respan.instrumentation.openrouter"
OPENROUTER_SYSTEM_NAME = "openrouter"

MAX_ATTRIBUTE_BYTES = 16_000
MAX_COLLECTION_ITEMS = 128
MAX_ERROR_BYTES = 4_000

SENSITIVE_KEY_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "client_secret",
        "credential",
        "db_password",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)

OPENAI_INSTRUMENTATION_SCOPE_FRAGMENT = "openai"
OPENROUTER_HOST_MARKERS = (
    "openrouter.ai",
    "openrouter",
)

OPENROUTER_URL_ATTRIBUTE_KEYS = (
    "url.full",
    "http.url",
    "server.address",
    "net.peer.name",
    "http.host",
    "openai.base_url",
    "base_url",
)
