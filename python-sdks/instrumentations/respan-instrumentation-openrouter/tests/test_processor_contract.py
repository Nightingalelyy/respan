from __future__ import annotations

import json

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_openrouter._constants import (
    MAX_ATTRIBUTE_BYTES,
    OPENROUTER_INSTRUMENTATION_SCOPE,
)
from respan_instrumentation_openrouter._processor import (
    OpenRouterSpanProcessor,
    _redact_text,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.exporters.respan import _span_to_otlp_json
from respan_tracing.utils.span_factory import build_readable_span


def _span(attributes: dict, *, error_message: str | None = None):
    return build_readable_span(
        "openai.chat",
        start_time_ns=1,
        end_time_ns=2,
        attributes=attributes,
        error_message=error_message,
        merge_propagated=False,
    )


def _otlp_attributes(payload: dict) -> dict:
    result = {}
    for item in payload["attributes"]:
        value = item["value"]
        primitive = next(iter(value.values()))
        result[item["key"]] = primitive
    return result


def test_real_readable_span_has_openrouter_contract_and_scope() -> None:
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS: 7,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS: 5,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS: 12,
            SpanAttributes.TRACELOOP_SPAN_KIND: "llm",
            f"{SpanAttributes.LLM_PROMPTS}.0.role": "user",
            f"{SpanAttributes.LLM_PROMPTS}.0.content": "hello",
        }
    )

    OpenRouterSpanProcessor().on_end(span)

    attrs = dict(span.attributes)
    assert attrs[SpanAttributes.LLM_SYSTEM] == "openrouter"
    assert attrs[GEN_AI_PROVIDER_NAME] == "openrouter"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[RESPAN_LOG_TYPE] == LOG_TYPE_CHAT
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "openrouter.chat"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert attrs[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == 12
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    assert span.instrumentation_scope.name == OPENROUTER_INSTRUMENTATION_SCOPE


def test_provider_429_is_an_otel_error_in_the_real_export_payload() -> None:
    secret = "sk-or-v1-1234567890abcdef"
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: '[{"role":"user","content":"hi"}]',
        },
        error_message=f"Error code: 429 - rate limited; api_key={secret}",
    )

    OpenRouterSpanProcessor().on_end(span)

    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["http.response.status_code"] == 429
    assert secret not in span.attributes["error.message"]
    error_output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert error_output["status_code"] == 429
    assert secret not in json.dumps(error_output)

    otlp = _span_to_otlp_json(span)
    assert otlp["status"]["code"] == 2
    assert secret not in otlp["status"]["message"]
    exported_attrs = _otlp_attributes(otlp)
    assert int(exported_attrs["http.response.status_code"]) == 429
    assert secret not in exported_attrs["error.message"]


def test_unrelated_status_code_text_is_not_promoted_to_provider_404() -> None:
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
        },
        error_message=(
            "cache HTTP 404 while formatting application output; status code: 404"
        ),
    )

    OpenRouterSpanProcessor().on_end(span)

    assert span.attributes["http.response.status_code"] == 500


def test_generic_error_code_text_is_not_promoted_without_provider_scope() -> None:
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
        },
        error_message="application cache error code: 404 during tool execution",
    )

    OpenRouterSpanProcessor().on_end(span)

    assert span.attributes["http.response.status_code"] == 500


def test_capture_content_false_keeps_identity_usage_and_error_only() -> None:
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS: 3,
            SpanAttributes.TRACELOOP_ENTITY_INPUT: "secret prompt",
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: "secret response",
            SpanAttributes.LLM_REQUEST_FUNCTIONS: '[{"type":"function"}]',
            f"{SpanAttributes.LLM_PROMPTS}.0.content": "secret prompt",
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content": "secret response",
        },
        error_message="OpenRouter error code: 401 - credential rejected",
    )

    OpenRouterSpanProcessor(capture_content=False).on_end(span)

    attrs = dict(span.attributes)
    assert attrs[GEN_AI_PROVIDER_NAME] == "openrouter"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert attrs["http.response.status_code"] == 401
    assert "error.message" in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS not in attrs
    assert not any(
        key.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for key in attrs
    )


def test_capture_content_false_still_redacts_sensitive_url_fields() -> None:
    secret = "plain-secret-value"
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            "url.full": (
                "https://user:password@openrouter.ai/api/v1/chat/completions"
                f"?api_key={secret}&model=openai%2Fgpt-4o-mini"
                "&client_secret=another-secret"
            ),
        }
    )

    OpenRouterSpanProcessor(capture_content=False).on_end(span)

    url = span.attributes["url.full"]
    assert "user:password" not in url
    assert secret not in url
    assert "another-secret" not in url
    assert "api_key=%5BREDACTED%5D" in url
    assert "model=openai%2Fgpt-4o-mini" in url
    assert len(url.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES


def test_quoted_and_suffix_sensitive_fields_are_redacted() -> None:
    secret_values = {
        "api_key": "plain-secret-value",
        "authorization": "Basic plaintext-credential",
        "nested_client_secret": "client-secret-value",
        "auth_token": "auth-token-value",
        "accessToken": "camel-access-token-value",
        "db_password": "database-password-value",
        "token": "plain-token-value",
    }
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: json.dumps(secret_values),
        }
    )

    OpenRouterSpanProcessor().on_end(span)

    safe_input = span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    parsed = json.loads(safe_input)
    assert all(value == "[REDACTED]" for value in parsed.values())
    assert not any(value in safe_input for value in secret_values.values())
    assert _redact_text('{"api_key":"plain-secret-value"}') == (
        '{"api_key":"[REDACTED]"}'
    )
    assert _redact_text("'authorization': 'Basic plaintext-credential'") == (
        "'authorization': '[REDACTED]'"
    )


def test_content_is_private_finite_and_utf8_byte_bounded() -> None:
    secret = "sk-or-v1-1234567890abcdef"
    huge = "界" * (MAX_ATTRIBUTE_BYTES * 2)
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
            SpanAttributes.TRACELOOP_ENTITY_INPUT: json.dumps(
                {
                    "messages": [{"role": "user", "content": huge}],
                    "authorization": f"Bearer {secret}",
                    "temperature": float("nan"),
                }
            ),
            SpanAttributes.LLM_REQUEST_FUNCTIONS: json.dumps(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"api_key": secret},
                        },
                    }
                ]
            ),
        }
    )

    OpenRouterSpanProcessor().on_end(span)

    for key in (
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        SpanAttributes.LLM_REQUEST_FUNCTIONS,
    ):
        value = span.attributes[key]
        assert len(value.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES
        assert secret not in value
        assert "NaN" not in value
        json.loads(value)


def test_retained_model_is_redacted_and_utf8_byte_bounded() -> None:
    secret = "plain-secret-value"
    model = f"openai/{'界' * MAX_ATTRIBUTE_BYTES}?api_key={secret}"
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: model,
        }
    )

    OpenRouterSpanProcessor(capture_content=False).on_end(span)

    safe_model = span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
    assert safe_model.startswith("openai/")
    assert secret not in safe_model
    assert len(safe_model.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES


def test_embedding_vector_is_not_truncated_by_chat_bounds() -> None:
    vector = [0.125] * (MAX_ATTRIBUTE_BYTES + 1)
    output = json.dumps(vector)
    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/text-embedding-3-small",
            SpanAttributes.LLM_REQUEST_TYPE: "embedding",
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT: output,
        }
    )

    OpenRouterSpanProcessor().on_end(span)

    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == output
    assert span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_EMBEDDING
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "openrouter.embeddings"
    )
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""


def test_hostile_readable_span_status_properties_do_not_break_export() -> None:
    class HostileStatus:
        @property
        def status_code(self):
            raise RuntimeError("hostile status property invoked")

        @property
        def description(self):
            raise RuntimeError("hostile status description invoked")

    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
        }
    )
    span._status = HostileStatus()

    OpenRouterSpanProcessor().on_end(span)

    assert span.attributes[GEN_AI_PROVIDER_NAME] == "openrouter"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "openrouter.chat"
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""


def test_stream_flag_uses_current_and_traceloop_conventions() -> None:
    from respan_instrumentation_openrouter._processor import (
        openrouter_emission_context,
    )

    span = _span(
        {
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_REQUEST_MODEL: "openai/gpt-4o-mini",
        }
    )
    with openrouter_emission_context(
        kind="chat",
        request_kwargs={"stream": True},
        error_message=None,
        status_code=200,
    ):
        OpenRouterSpanProcessor().on_end(span)

    assert span.attributes[GEN_AI_REQUEST_STREAM] is True
    assert span.attributes[SpanAttributes.LLM_IS_STREAMING] is True
