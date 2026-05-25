# respan-instrumentation-guardrails

Respan instrumentation plugin for [Guardrails AI](https://www.guardrailsai.com/). It wraps Guardrails `Guard` execution methods and emits `guardrail` spans into the Respan tracing pipeline.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-guardrails guardrails-ai
```

`guardrails-ai` is the Guardrails runtime package. It must be available in your package index for runnable Guardrails applications.

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

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

from guardrails import Guard
from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_guardrails import GuardrailsInstrumentor


class SupportReply(BaseModel):
    answer: str
    priority: str


respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[GuardrailsInstrumentor()],
)

guard = Guard.for_pydantic(output_class=SupportReply)
result = guard(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Return JSON with answer and priority for a late shipment.",
        }
    ],
)

print(result.validated_output)
respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/guardrails) for runnable scripts.
