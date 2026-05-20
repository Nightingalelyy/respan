# respan-instrumentation-autogen

Respan instrumentation plugin for AutoGen AgentChat. Wraps OpenInference's
AutoGen AgentChat instrumentor and translates spans into the Respan tracing
shape automatically.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-autogen
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates gateway requests and trace export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |
| `RESPAN_MODEL` | No | Defaults to `gpt-4o-mini`. |

## Quickstart

```python
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from respan import Respan
from respan_instrumentation_autogen import AutoGenInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
respan_model = os.getenv("RESPAN_MODEL", "gpt-4o-mini")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[AutoGenInstrumentor()],
)

model_client = OpenAIChatCompletionClient(
    model=respan_model,
    api_key=respan_api_key,
    base_url=respan_base_url,
)


async def main() -> None:
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )
    result = await agent.run(task="Write one sentence about recursive functions.")
    print(result.messages[-1].content)
    await model_client.close()
    respan.flush()


asyncio.run(main())
```

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects)
for runnable scripts.
