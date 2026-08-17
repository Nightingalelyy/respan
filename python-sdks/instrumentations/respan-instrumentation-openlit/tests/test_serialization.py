from __future__ import annotations

import json
import math
from collections.abc import Mapping

from respan_instrumentation_openlit._serialization import (
    REDACTED,
    input_messages,
    json_string,
    json_value,
    safe_url,
    tool_definitions,
)


class Hostile:
    def __str__(self) -> str:
        raise AssertionError("custom __str__ must not run")

    def __repr__(self) -> str:
        raise AssertionError("custom __repr__ must not run")


class HostileMapping(Mapping):
    def __getitem__(self, key):
        raise AssertionError(key)

    def __iter__(self):
        raise AssertionError("hostile iterator")

    def __len__(self):
        raise AssertionError("hostile length")

    def items(self):
        raise RuntimeError("hostile items")


def test_json_value_is_finite_redacting_and_cycle_safe() -> None:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    normalized = json_value(
        {
            "authorization": "Bearer raw-secret-value",
            "api_key": "sk-this-should-never-leak",
            "message": "token=private-value",
            "dsn": "postgresql://alice:private-value@db.internal/example",
            "environment": "OPENAI_API_KEY=private-value",
            "nested": {
                "db_password": "nested-password",
                "service_token": "nested-token",
                "tenant_secret": "nested-secret",
                "cloud_api_key": "nested-api-key",
                "service_credential": "nested-credential",
            },
            "address": "object at 0xDEADBEEF",
            "nan": math.nan,
            "infinity": math.inf,
            "cycle": cycle,
            "hostile": Hostile(),
            "hostile_mapping": HostileMapping(),
        }
    )
    assert normalized["authorization"] == REDACTED
    assert normalized["api_key"] == REDACTED
    assert "private-value" not in normalized["message"]
    assert "private-value" not in normalized["dsn"]
    assert "private-value" not in normalized["environment"]
    assert normalized["nested"] == {
        "cloud_api_key": REDACTED,
        "db_password": REDACTED,
        "service_credential": REDACTED,
        "service_token": REDACTED,
        "tenant_secret": REDACTED,
    }
    assert "0xDEADBEEF" not in normalized["address"]
    assert normalized["nan"] is None
    assert normalized["infinity"] is None
    assert normalized["cycle"]["self"] == "<cycle>"
    assert normalized["hostile"] == "<Hostile>"
    assert normalized["hostile_mapping"]["__serialization_error__"] == (
        "HostileMapping"
    )


def test_json_string_is_valid_json_with_exact_utf8_byte_cap() -> None:
    encoded = json_string(
        {"unicode": "界" * 20_000, "authorization": "Bearer secret-value"},
        max_bytes=512,
    )
    assert len(encoded.encode("utf-8")) <= 512
    payload = json.loads(encoded)
    assert payload["truncated"] is True
    assert payload["original_bytes"] > 512
    assert "secret-value" not in encoded


def test_safe_url_drops_userinfo_query_and_fragment() -> None:
    assert (
        safe_url("https://alice:private@db.internal:8443/v1?api_key=secret#fragment")
        == "https://db.internal:8443/v1"
    )
    assert safe_url("https://[::1]:4318/v1?token=secret") == ("https://[::1]:4318/v1")
    assert safe_url(Hostile()) == ""


def test_openai_message_and_tool_normalization_does_not_invoke_hostile_text() -> None:
    messages = input_messages(
        [
            {"role": "user", "content": Hostile(), "password": "secret"},
            {"role": "user", "content": "authorization=private-value"},
        ]
    )
    assert messages[0]["parts"][0]["content"] == "<Hostile>"
    assert "private-value" not in json.dumps(messages)

    tools = tool_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "api_key=private-value",
                    "parameters": {"type": "object", "token": "private-value"},
                },
            },
            Hostile(),
        ]
    )
    assert tools[0]["name"] == "lookup"
    assert "private-value" not in json.dumps(tools)
