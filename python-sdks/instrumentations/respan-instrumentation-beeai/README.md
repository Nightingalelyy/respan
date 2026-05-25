# Respan BeeAI Instrumentation

Respan instrumentation plugin for the [BeeAI Framework](https://framework.beeai.dev/).

The package wraps `openinference-instrumentation-beeai` with Respan's
instrumentation lifecycle and registers Respan's OpenInference translator so
BeeAI agent, workflow, tool, chat model, and embedding spans are exported
through the Respan OTLP pipeline.

## Install

```bash
pip install respan-ai respan-instrumentation-beeai beeai-framework
```

## Usage

Initialize Respan before creating or running BeeAI Framework components.

```python
import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from respan import Respan
from respan_instrumentation_beeai import BeeAIInstrumentor


async def main() -> None:
    respan = Respan(instrumentations=[BeeAIInstrumentor()])

    agent = RequirementAgent(
        llm=ChatModel.from_name("openai:gpt-4.1-nano"),
        role="friendly AI assistant",
        instructions="Answer clearly and concisely.",
    )
    response = await agent.run("Explain why observability matters for agents.")
    print(response.last_message.text)

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
```

Any keyword arguments passed to `BeeAIInstrumentor(...)` are forwarded to the
underlying OpenInference BeeAI instrumentor.
