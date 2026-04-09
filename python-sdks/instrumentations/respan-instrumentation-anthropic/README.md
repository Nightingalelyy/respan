# respan-instrumentation-anthropic

Respan instrumentation plugin for the
[Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

This package patches Anthropic client calls and emits spans using the
Respan/Traceloop GenAI attribute shape used across this repository.

## Install

```bash
pip install respan-instrumentation-anthropic
```

## Quickstart

```python
import os

from anthropic import Anthropic
from respan import Respan
from respan_instrumentation_anthropic import AnthropicInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[AnthropicInstrumentor()],
)

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=128,
    messages=[{"role": "user", "content": "Write one line about tracing."}],
)

print(message.content)
respan.flush()
```

## Notes

- The instrumentor patches both `Anthropic` and `AsyncAnthropic`.
- `messages.create()` and streaming responses are traced.
- Managed agent session streaming is also captured when available through the Anthropic SDK.
