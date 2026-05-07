# respan-instrumentation-crewai

Respan instrumentation plugin for CrewAI. Wraps OpenInference's CrewAI instrumentor and translates spans into the Respan tracing shape automatically.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-crewai
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

# Route OpenAI traffic through the Respan gateway (no separate OpenAI key needed)
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

from respan import Respan
from respan_instrumentation_crewai import CrewAIInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[CrewAIInstrumentor()],
)

from crewai import Agent, Task, Crew

agent = Agent(
    role="Poet",
    goal="Write a short haiku about recursion in programming",
    backstory="You are a programmer who writes haikus.",
    llm="gpt-4o-mini",
)

task = Task(
    description="Write a haiku about recursion in programming.",
    expected_output="A single haiku (3 lines: 5-7-5 syllables).",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
print(result.raw)

respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects) for runnable scripts.
