# Respan Mistral AI instrumentation

Trace the official `mistralai` Python SDK with Respan.

This package wraps `openinference-instrumentation-mistralai` and registers
Respan's OpenInference translator so Mistral AI spans are emitted with the
canonical `traceloop.*`, `gen_ai.*`, `llm.*`, and `respan.*` fields expected by
the Respan OTLP pipeline. It covers synchronous and asynchronous chat,
streaming content and usage, request tool definitions, current-turn tool calls,
precise provider error status, and interrupted-stream finalization. Duplicate
native Mistral SDK spans are suppressed only for the SDK's own instrumentation
scope. Tested dependency support starts at `mistralai==2.9.3` and
`openinference-instrumentation-mistralai==2.0.6`.

## Install

```bash
pip install respan-ai respan-instrumentation-mistralai mistralai
```

## Usage

```python
import os

from mistralai.client import Mistral
from respan import Respan
from respan_instrumentation_mistralai import MistralAIInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[MistralAIInstrumentor()],
)

with Mistral(api_key=os.environ["MISTRAL_API_KEY"]) as client:
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": "Reply with one concise sentence about tracing.",
            }
        ],
    )
    print(response.choices[0].message.content)

respan.flush()
respan.shutdown()
```

Any keyword arguments passed to `MistralAIInstrumentor(...)` are forwarded to the
underlying OpenInference instrumentor. Multiple Respan Mistral instrumentor
instances share one OpenInference delegate and one normalization processor;
the final owner deactivates them. Shared instances must use identical keyword
arguments. An activation with different arguments is rejected and remains
inactive rather than silently inheriting another instance's configuration.
