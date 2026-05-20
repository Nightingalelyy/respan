# respan-instrumentation-dspy

Respan instrumentation plugin for [DSPy](https://dspy.ai/).

This package registers a native DSPy callback and emits module, language model,
tool, adapter, and evaluation spans into the Respan tracing pipeline.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-dspy
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

### 3. Run Script

```python
import os

import dspy
from respan import Respan
from respan_instrumentation_dspy import DSPyInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url
os.environ["OPENAI_API_BASE"] = respan_base_url

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    app_name="dspy-quickstart",
    instrumentations=[DSPyInstrumentor()],
)

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))
question_answerer = dspy.Predict("question -> answer")
prediction = question_answerer(question="What is DSPy in one sentence?")

print(prediction.answer)
respan.flush()
```

### 4. View Dashboard

Open the Respan dashboard and inspect the `dspy-quickstart` trace.

## Further Reading

- [DSPy examples](https://github.com/Keywords-AI/respan-example-projects/tree/main/python/tracing/dspy)
- [respan-ai](https://pypi.org/project/respan-ai/)
- [respan-tracing](https://pypi.org/project/respan-tracing/)
- [DSPy](https://dspy.ai/)
