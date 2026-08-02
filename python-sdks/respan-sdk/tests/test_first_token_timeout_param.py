from respan_sdk.respan_types.param_types import RespanParams


def test_first_token_timeout_is_preserved_for_gateway_request():
    params = RespanParams.model_validate({"first_token_timeout": "120"})

    assert params.first_token_timeout == 120.0
    assert params.model_dump()["first_token_timeout"] == 120.0
