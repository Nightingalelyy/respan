# respan-instrumentation-haystack

Respan instrumentation plugin for Haystack by deepset. Wraps `opentelemetry-instrumentation-haystack` to automatically trace pipeline runs, component executions, and LLM calls.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-haystack
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai`. |

## Quickstart

### 3. Run Script

```python
import os
from dotenv import load_dotenv

load_dotenv()

from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor

respan = Respan(instrumentations=[HaystackInstrumentor()])

from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

template = """Answer the following question: {{question}}"""

pipe = Pipeline()
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))
pipe.connect("prompt_builder", "llm")

result = pipe.run({"prompt_builder": {"question": "What is the capital of France?"}})
print(result["llm"]["replies"][0])

respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).
