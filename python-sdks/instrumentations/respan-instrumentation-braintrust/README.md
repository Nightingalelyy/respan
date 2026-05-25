# Respan Braintrust Instrumentation

Send Braintrust spans to Respan through the current Respan OTLP tracing pipeline.

## Installation

```bash
pip install respan-ai respan-instrumentation-braintrust braintrust
```

## Quick Start

```python
import braintrust
from respan import Respan
from respan_instrumentation_braintrust import BraintrustInstrumentor

respan = Respan(instrumentations=[BraintrustInstrumentor()])

logger = braintrust.init_logger(
    project="braintrust-demo",
    async_flush=False,
)

with respan.propagate_attributes(group_identifier="Braintrust Demo"):
    with logger.start_span(name="Braintrust Demo", type="task") as root:
        with root.start_span(name="answer_question", type="llm") as span:
            span.log(
                input=[{"role": "user", "content": "What is Respan?"}],
                output="Respan captures production LLM traces.",
                metadata={"model": "gpt-4o-mini"},
                metrics={"prompt_tokens": 8, "completion_tokens": 9},
            )

    logger.flush()

respan.flush()
respan.shutdown()
```

## How It Works

`BraintrustInstrumentor` installs itself as Braintrust's background logger.
When Braintrust flushes, the instrumentor translates Braintrust records into
canonical Respan/OpenTelemetry spans and injects them into the same pipeline used
by `respan-tracing`.

The instrumentation maps Braintrust span types as follows:

- `llm` and `chat` -> `chat`
- `tool` and `function` -> `tool`
- `eval` and `automation` -> `workflow`
- `task`, `score`, `facet`, and `preprocessor` -> `task`

Braintrust-specific record fields are stored as metadata, while shared Respan
and GenAI span fields use constants from `respan-sdk` and upstream semantic
convention packages.
