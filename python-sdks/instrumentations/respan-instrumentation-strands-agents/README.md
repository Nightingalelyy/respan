# respan-instrumentation-strands-agents

Respan instrumentation plugin for [Strands Agents](https://strandsagents.com/).

This package consumes Strands Agents' native OpenTelemetry spans and maps them
directly into the Respan span contract used by the OTLP pipeline. It does not
require OpenInference at runtime.

## Install

```bash
pip install respan-instrumentation-strands-agents
```

## Quickstart

```python
import os

from respan import Respan
from respan_instrumentation_strands_agents import StrandsAgentsInstrumentor
from strands import Agent, tool
from strands.models.openai import OpenAIModel

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[StrandsAgentsInstrumentor()],
)

model = OpenAIModel(
    model_id="gpt-4o-mini",
    client_args={"api_key": respan_api_key, "base_url": respan_base_url},
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72F."


agent = Agent(
    name="WeatherAgent",
    model=model,
    tools=[get_weather],
    system_prompt="You are a concise weather assistant.",
)

result = agent("What is the weather in Seattle?")
print(result)

respan.flush()
```

## Notes

- Initialize `Respan(...)` before running the agent so Strands uses the active
  Respan OpenTelemetry provider.
- The instrumentor can refresh an already-created Strands tracer singleton when
  possible, which helps when an agent was constructed before activation.
- Tool definitions are enabled by default via Strands' `gen_ai_tool_definitions`
  semantic-convention opt-in and are exported as canonical `llm.request.functions`
  attributes.
