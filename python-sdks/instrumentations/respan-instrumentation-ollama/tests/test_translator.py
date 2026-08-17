from __future__ import annotations

import json
from typing import Any

from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_ollama._otel_emitter import build_chat_attrs
from respan_instrumentation_ollama._translator import (
    StreamResponseAccumulator,
    safe_json,
)


class _SecretRepr:
    def __repr__(self) -> str:
        return "Bearer repr-secret-must-not-escape"

    def __str__(self) -> str:
        return "Bearer str-secret-must-not-escape"


def test_safe_json_is_bounded_redacted_and_avoids_arbitrary_repr() -> None:
    encoded = safe_json(
        {
            "Authorization": "Bearer mapping-secret-must-not-escape",
            "nested": {"api_key": "key-secret-must-not-escape"},
            "opaque": _SecretRepr(),
            "long": "x" * 20_000,
        }
    )

    parsed = json.loads(encoded)
    assert len(encoded) <= 16_000
    assert "mapping-secret-must-not-escape" not in encoded
    assert "key-secret-must-not-escape" not in encoded
    assert "repr-secret-must-not-escape" not in encoded
    assert "str-secret-must-not-escape" not in encoded
    assert parsed["Authorization"] == "[REDACTED]"
    assert parsed["nested"]["api_key"] == "[REDACTED]"
    assert parsed["opaque"] == {}
    assert parsed["long"].endswith("...[truncated]")

    escape_encoded = safe_json({"escape_heavy": '"\\\n' * 20_000})
    assert len(escape_encoded) <= 16_000
    assert json.loads(escape_encoded)["truncated"] is True


def test_direct_prompt_and_completion_attributes_are_bounded_and_redacted() -> None:
    secret = "do not expose Authorization: Bearer private-value "
    response: dict[str, Any] = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": secret + "y" * 20_000},
    }
    attrs = build_chat_attrs(
        request_kwargs={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": secret + "x" * 20_000}],
        },
        response_or_chunks=response,
    )

    prompt = attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    completion = attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
    assert len(prompt) <= 8_000
    assert len(completion) <= 8_000
    assert "private-value" not in prompt
    assert "private-value" not in completion
    assert prompt.endswith("...[truncated]")
    assert completion.endswith("...[truncated]")


def test_stream_tool_call_retention_is_bounded_and_keeps_first_valid_calls() -> None:
    captured = StreamResponseAccumulator()
    for index in range(100):
        captured.append(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": f"tool_{index}",
                                "arguments": {"index": index},
                            }
                        }
                    ],
                },
                "prompt_eval_count": index,
                "eval_count": index + 1,
            }
        )

    response = captured.response()
    calls = response["message"]["tool_calls"]
    assert len(calls) == 50
    assert calls[0]["function"]["name"] == "tool_0"
    assert calls[-1]["function"]["name"] == "tool_49"
    assert response["prompt_eval_count"] == 99
    assert response["eval_count"] == 100
