import pytest
from pydantic import ValidationError

from respan_sdk import (
    LiteLLMProviderCompletionParams,
    OpenRouterCompletionParams,
)
from respan_sdk.respan_types._internal_types import LiteLLMCompletionParams


def test_service_tier_is_preserved_for_provider_request():
    params = LiteLLMCompletionParams.model_validate(
        {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hello"}],
            "service_tier": "flex",
        }
    )

    assert params.service_tier == "flex"
    assert params.model_dump()["service_tier"] == "flex"


def test_openrouter_request_params_are_preserved():
    params = LiteLLMProviderCompletionParams.model_validate(
        {
            "model": "openrouter/deepseek/deepseek-v4-flash-0731",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning": {"enabled": False},
            "provider": {"zdr": True},
        }
    )

    assert params.model_dump()["reasoning"] == {"enabled": False}
    assert params.model_dump()["provider"] == {"zdr": True}


def test_openrouter_reasoning_rejects_log_output_list():
    with pytest.raises(ValidationError):
        LiteLLMProviderCompletionParams.model_validate(
            {
                "model": "openrouter/deepseek/deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning": [{"type": "reasoning"}],
            }
        )


def test_generic_completion_params_do_not_claim_openrouter_fields():
    params = LiteLLMCompletionParams.model_validate(
        {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning": {"enabled": False},
            "provider": {"zdr": True},
        }
    )

    dumped = params.model_dump()
    assert "reasoning" not in dumped
    assert "provider" not in dumped


def test_previous_openrouter_type_name_remains_compatible():
    assert OpenRouterCompletionParams is LiteLLMProviderCompletionParams
