# respan-instrumentation-openai-agents

Respan instrumentation plugin for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Captures agent traces, tool calls, handoffs, and LLM generations via the OTEL pipeline.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-openai-agents
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai`. |

All vendor-specific variables (e.g. `OPENAI_API_KEY`) are derived from these in your application code.

## Quickstart

### 3. Run Script

```python
import asyncio
import os
from agents import Agent, Runner
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai")

os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = f"{respan_base_url}/api/openai"

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[OpenAIAgentsInstrumentor()],
)

agent = Agent(name="Assistant", instructions="You are a helpful assistant.")

async def main():
    result = await Runner.run(agent, "Hello!")
    print(result.final_output)

asyncio.run(main())
respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [examples/openai-agents-sdk/](https://github.com/RespanAI/respan/tree/main/examples/openai-agents-sdk) directory for runnable examples including tool use, handoffs, multi-agent workflows, guardrails, and streaming.
