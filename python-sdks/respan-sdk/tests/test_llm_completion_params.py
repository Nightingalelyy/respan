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
