# respan-instrumentation-smolagents

Respan instrumentation plugin for smolagents. Wraps OpenInference's smolagents instrumentor and translates spans into the Respan tracing shape automatically.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-smolagents
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates trace export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

```python
import os

from dotenv import load_dotenv
from respan import Respan
from respan_instrumentation_smolagents import SmolagentsInstrumentor
from smolagents import CodeAgent, LiteLLMModel

load_dotenv()

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[SmolagentsInstrumentor()],
)

model = LiteLLMModel(
    model_id="openai/gpt-4o-mini",
    api_key=respan_api_key,
    api_base=respan_base_url,
)
agent = CodeAgent(tools=[], model=model)

print(agent.run("Return a one-sentence explanation of recursion."))
respan.flush()
```

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/smolagents) for runnable scripts.
