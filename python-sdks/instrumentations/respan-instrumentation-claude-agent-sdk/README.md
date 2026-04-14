# respan-instrumentation-claude-agent-sdk

Respan instrumentation plugin for the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).

This package enables the Claude Agent SDK's native OpenTelemetry emission and
normalizes those spans into the Respan/Traceloop conventions used by the OTLP
pipeline.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-claude-agent-sdk
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

All vendor-specific variables (for example `ANTHROPIC_API_KEY`) are derived
from these in your application code.

## Quickstart

### 3. Run Script

```python
import asyncio
import os

import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from respan import Respan
from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

os.environ["ANTHROPIC_API_KEY"] = respan_api_key
os.environ["ANTHROPIC_AUTH_TOKEN"] = respan_api_key
os.environ["ANTHROPIC_BASE_URL"] = f"{respan_base_url}/anthropic"

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[ClaudeAgentSDKInstrumentor(capture_content=True)],
)


async def main() -> None:
    options = ClaudeAgentOptions(
        model="sonnet",
        max_turns=1,
        permission_mode="bypassPermissions",
        cwd=os.getcwd(),
        env={
            "ANTHROPIC_API_KEY": respan_api_key,
            "ANTHROPIC_AUTH_TOKEN": respan_api_key,
            "ANTHROPIC_BASE_URL": f"{respan_base_url}/anthropic",
        },
    )

    async for message in claude_agent_sdk.query(
        prompt="Reply with exactly hello_from_claude_sdk.",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            print(message.result)


asyncio.run(main())
respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [python/tracing/claude-agent-sdk](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/claude-agent-sdk)
example for a runnable end-to-end workflow that covers tool use, multi-turn
sessions, and edge cases.
