# respan-instrumentation-agno

Respan instrumentation plugin for [Agno](https://docs.agno.com/).

This package patches Agno's native `Agent` and `Team` run methods and emits
Respan-compatible OpenTelemetry spans directly. It does not use Agno's
OpenInference integration.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-agno
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

### 3. Run Script

```python
import os

from dotenv import load_dotenv

load_dotenv()

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from respan import Respan
from respan_instrumentation_agno import AgnoInstrumentor

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[AgnoInstrumentor()],
)

agent = Agent(
    name="Haiku Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
)

result = agent.run("Write a one-line haiku about tracing.")
print(result.content)

respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

- [Agno examples in respan-example-projects](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/agno)
- [Respan example projects](https://github.com/respanai/respan-example-projects)
- [Agno documentation](https://docs.agno.com/)
