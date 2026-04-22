# respan-instrumentation-crewai

Respan instrumentation plugin for CrewAI. Wraps `opentelemetry-instrumentation-crewai` to automatically trace agent runs, task executions, and tool calls.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-crewai
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai`. |

## Quickstart

### 3. Run Script

```python
from dotenv import load_dotenv

load_dotenv()

from respan import Respan
from respan_instrumentation_crewai import CrewAIInstrumentor

respan = Respan(instrumentations=[CrewAIInstrumentor()])

from crewai import Agent, Task, Crew

agent = Agent(
    role="Poet",
    goal="Write a short haiku about recursion in programming",
    backstory="You are a programmer who writes haikus.",
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

See the [examples/crewai/](../../examples/crewai/) directory for runnable examples.
