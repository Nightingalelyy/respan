# respan-instrumentation-agentspec

Respan instrumentation plugin for AgentSpec (`pyagentspec`).

This package activates the upstream OpenInference AgentSpec span processor
through Respan's OpenInference translator, so AgentSpec spans are exported
through the same Respan OTLP pipeline used by `Respan`.

## Installation

```bash
pip install respan-ai respan-instrumentation-agentspec "pyagentspec[langgraph]"
```

## Usage

```python
from pyagentspec.adapters.langgraph import AgentSpecLoader
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig
from respan import Respan
from respan_instrumentation_agentspec import AgentSpecInstrumentor

respan = Respan(
    app_name="agentspec-haiku-agent",
    instrumentations=[
        AgentSpecInstrumentor(workflow_name="agentspec_haiku_agent")
    ],
)

try:
    agent = Agent(
        name="haiku_assistant",
        description="A helpful assistant that writes haikus.",
        llm_config=OpenAiConfig(name="openai", model_id="gpt-4.1-nano"),
        system_prompt="You are a helpful assistant. Respond only with a haiku.",
    )

    langgraph_agent = AgentSpecLoader().load_component(agent)
    result = langgraph_agent.invoke(
        input={"messages": [{"role": "user", "content": "Write a haiku about tracing."}]}
    )

    print(result["messages"][-1].content)
finally:
    respan.shutdown()
    respan.flush()
```
