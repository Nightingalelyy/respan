# Respan Exporter for Anthropic Agent SDK

**[respan.ai](https://respan.ai)** | **[Documentation](https://respan.ai/docs)**

Exporter for Anthropic Agent SDK telemetry to Respan.

## Configuration

### 1. Install

```bash
pip install claude-agent-sdk respan-exporter-anthropic-agents
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Respan API key used for telemetry export. |
| `RESPAN_BASE_URL` | No | Respan base URL for telemetry export. Defaults to `https://api.respan.ai`. |
| `ANTHROPIC_BASE_URL` | No | Inference/proxy base URL used by the Anthropic SDK. |
| `ANTHROPIC_API_KEY` | Usually | Key used by the Anthropic SDK for inference calls. |
| `ANTHROPIC_AUTH_TOKEN` | Optional | Alternate auth token used by some Anthropic client flows. |

Set both groups together when needed. `RESPAN_*` controls tracing export, while `ANTHROPIC_*` controls where model requests are sent.

```bash
# Tracing export (Respan telemetry)
RESPAN_API_KEY=your_respan_key
RESPAN_BASE_URL=https://api.respan.ai/api

# Inference/proxy routing (Anthropic SDK)
# Optional: set only if you use a custom proxy/gateway base URL
# ANTHROPIC_BASE_URL=https://your-anthropic-base-url
ANTHROPIC_API_KEY=your_inference_key
ANTHROPIC_AUTH_TOKEN=your_inference_key
```

`RESPAN_BASE_URL` controls telemetry export only. The exporter automatically appends `/api/v1/traces/ingest` to build the full ingest endpoint.
In normal usage, instantiate `RespanAnthropicAgentsExporter()` with no arguments and configure via environment variables.
`ANTHROPIC_BASE_URL` is optional. If you use a gateway/proxy, set it to that gateway's Anthropic-compatible base URL.

## Quickstart

### 3. Run Script

Save this as `quickstart.py`:

```python
import asyncio
import os
from claude_agent_sdk import ClaudeAgentOptions
from respan_exporter_anthropic_agents.respan_anthropic_agents_exporter import (
    RespanAnthropicAgentsExporter,
)

respan_api_key = os.environ["RESPAN_API_KEY"]
anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", respan_api_key)
anthropic_auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", anthropic_api_key)

exporter = RespanAnthropicAgentsExporter()

async def main() -> None:
    anthropic_env = {
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "ANTHROPIC_AUTH_TOKEN": anthropic_auth_token,
    }
    if anthropic_base_url:
        anthropic_env["ANTHROPIC_BASE_URL"] = anthropic_base_url

    options = exporter.with_options(
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="acceptEdits",
            env=anthropic_env,
        )
    )

    async for message in exporter.query(
        prompt="Analyze this repository and summarize architecture.",
        options=options,
    ):
        print(message)

asyncio.run(main())
```

Run it:

```bash
python quickstart.py
```

### 4. View Dashboard

Open:

- `https://platform.respan.ai/platform/traces`

## Further Reading

Runnable examples with full setup instructions:

- **Python examples root:** [python/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/anthropic-agents-sdk)
- **Python basic scripts:**
  - [hello_world_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/hello_world_test.py)
  - [wrapped_query_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/wrapped_query_test.py)
  - [tool_use_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/tool_use_test.py)
  - [gateway_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/gateway_test.py)
- **TypeScript examples root:** [typescript/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/typescript/tracing/anthropic-agents-sdk)

## Dev Guide

### Running Tests

```bash
# Unit tests
python -m unittest tests.test_exporter -v

# Live integration test (opt-in, makes real API calls)
export RESPAN_API_KEY="your_respan_key"
export IS_REAL_GATEWAY_TESTING_ENABLED=1
python -m unittest tests.test_real_gateway_integration -v
```
