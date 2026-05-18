"""Serialization helpers for Superagent instrumentation."""

from __future__ import annotations

import json
from typing import Any

from respan_sdk.utils.serialization import serialize_value

from respan_instrumentation_superagent._constants import (
    INPUT_KEY,
    MODEL_KEY,
    REPO_KEY,
)


def safe_json_dumps(value: Any) -> str:
    """Serialize a value to JSON, falling back to ``str`` on unsupported shapes."""
    try:
        return json.dumps(serialize_value(value=value), default=str)
    except Exception:
        return str(value)


def normalize_call_input(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, JSON-safe representation of a Superagent call."""
    payload: dict[str, Any] = {"method": method_name}

    if args:
        payload["args"] = serialize_value(value=list(args))

    if kwargs:
        payload["kwargs"] = serialize_value(value=kwargs)

    return payload


def extract_model(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Extract the model argument from direct kwargs or option objects."""
    model = kwargs.get(MODEL_KEY)
    if isinstance(model, str) and model:
        return model

    if not args:
        return None

    first_arg = args[0]
    option_model = getattr(first_arg, MODEL_KEY, None)
    if isinstance(option_model, str) and option_model:
        return option_model

    return None


def extract_primary_input(
    *,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Extract the user-facing input/repo value when it is easy to identify."""
    if INPUT_KEY in kwargs:
        return kwargs[INPUT_KEY]
    if REPO_KEY in kwargs:
        return kwargs[REPO_KEY]

    if not args:
        return None

    first_arg = args[0]
    if isinstance(first_arg, (str, bytes)):
        return first_arg

    for field_name in (INPUT_KEY, REPO_KEY):
        field_value = getattr(first_arg, field_name, None)
        if field_value is not None:
            return field_value

    return None
