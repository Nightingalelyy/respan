# Respan instrumentation for AgentOps

This package sends spans created by AgentOps' Python decorators through the
active Respan OpenTelemetry pipeline.

It adapts AgentOps' own span kinds and content:

- `trace` / session and workflow decorators become Respan workflow spans
- agent decorators become agent spans
- task and operation decorators become task spans
- tool and guardrail decorators retain their matching Respan log types
- AgentOps LLM spans keep their GenAI messages and usage, with AgentOps'
  request-type and function fields promoted to the canonical Respan contract

The adapter does not initialize the AgentOps exporter. Activate Respan before
calling AgentOps-decorated code:

```python
from agentops import task, trace
from respan import Respan
from respan_instrumentation_agentops import AgentOpsInstrumentor

respan = Respan(
    api_key="...",
    instrumentations=[AgentOpsInstrumentor()],
)

@task
def prepare(value: str) -> str:
    return value.upper()

@trace
def workflow(value: str) -> str:
    return prepare(value)

workflow("hello")
respan.shutdown()
```

Set `capture_content=False` on `AgentOpsInstrumentor` to omit decorator inputs
and outputs while retaining operation identity and status.
