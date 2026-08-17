# respan-instrumentation-openai

Respan instrumentation plugin for direct OpenAI 3.x SDK usage. The native,
Traceloop-free integration patches sync and async Chat Completions, Responses,
Completions, and Embeddings resources. Chat and Responses structured-output
`parse` methods, streaming, tools, usage, and provider failures are included.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-openai
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

Use `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` when calling OpenAI
directly. A Respan gateway deployment may instead use its gateway credential
and OpenAI-compatible base URL.

## Quickstart

### 3. Run Script

```python
import os
from openai import OpenAI
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[OpenAIInstrumentor()],
)

client = OpenAI(
    api_key=respan_api_key,
    base_url=f"{respan_base_url}/api/openai",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

respan.flush()
respan.shutdown()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See `respan-example-projects/python/tracing/openai-sdk` for deterministic
OpenAI 3.x examples. Set `RESPAN_OPENAI_LIVE=1` to opt into a configured live
provider; deterministic mode still executes the real OpenAI SDK request and
response parsing layers through an in-process HTTP transport.
