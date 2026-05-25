# respan-instrumentation-haystack

Respan instrumentation plugin for Haystack by deepset. Wraps OpenInference's Haystack instrumentor and translates pipeline, component, and LLM spans into the Respan tracing contract.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-haystack
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai`. |

All vendor-specific variables are derived from these in your application code.

## Quickstart

### 3. Run Script

```python
import os

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai").rstrip("/")

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = f"{respan_base_url}/api/openai"

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[HaystackInstrumentor()],
)

template = """Answer the following question: {{question}}"""

pipeline = Pipeline()
pipeline.add_component("prompt_builder", PromptBuilder(template=template))
pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))
pipeline.connect("prompt_builder", "generator")

result = pipeline.run(
    {"prompt_builder": {"question": "What is the capital of France?"}}
)
print(result["generator"]["replies"][0])

respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [Haystack tracing examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/haystack) for runnable scripts covering gateway routing, RAG, routing, conversion, evaluators, and tool invocation.
