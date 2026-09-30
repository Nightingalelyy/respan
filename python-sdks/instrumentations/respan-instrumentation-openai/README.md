# respan-instrumentation-openai

Respan instrumentation for direct OpenAI Python SDK usage. The native
integration captures typed sync and async Chat Completions, Responses,
Completions, and Embeddings calls, including Chat and Responses `parse`
methods, tools, usage, and provider failures.

## SDK compatibility

Supports `openai>=3.0.0,<4.0.0` on Python 3.11–3.13. Verified against OpenAI
**3.0.0** and **3.19.2** (September 27, 2026); CI runs the real SDK tests
against the minimum and newest available 3.x release with released Respan
runtime dependencies.

Chat Completions and Responses support both `create(stream=True)` and the
`.stream()` context manager helpers. Consuming a stream or explicitly closing
it emits one span. An early close retains the text and function arguments
received so far; token usage is included only when the provider has supplied
it. Async streams must be consumed or closed with `await stream.close()` or
an async context manager. Responses `response.failed` events mark the span
as an error while leaving the SDK event stream unchanged.

Content capture applies to these typed methods. Raw HTTP response wrappers
(`with_raw_response` and `with_streaming_response`), existing-response
retrieval, Realtime, and other OpenAI resources are outside this capture
support.

## Install

```bash
pip install respan-ai 'respan-instrumentation-openai[instruments]'
```

The base instrumentation package keeps OpenAI optional. Use the `instruments`
extra above to install a compatible SDK, or manage `openai` yourself.

## Configuration

Set `RESPAN_API_KEY` for trace export and `OPENAI_API_KEY` for model requests.
`RESPAN_BASE_URL` optionally changes the Respan export endpoint;
`OPENAI_BASE_URL` optionally changes the OpenAI client endpoint.

```python
import os

from openai import OpenAI
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[OpenAIInstrumentor()],
)

with OpenAI() as client:
    response = client.responses.create(
        model="gpt-4.1-nano",
        input="Hello!",
    )
    print(response.output_text)

respan.flush()
respan.shutdown()
```

View exported traces on the [Respan dashboard](https://platform.respan.ai).

## Examples and validation

See `respan-example-projects/python/tracing/openai-sdk` for deterministic
examples that execute the real SDK request and response parsing layers
through an in-process HTTP transport, plus an opt-in live-provider example.

Run the package regressions locally with:

```bash
pip install -e '.[instruments]' pytest pytest-asyncio
pytest -q tests
```
