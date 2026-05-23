# respan-instrumentation-google-genai

Respan instrumentation plugin for the official
[Google Gen AI SDK](https://googleapis.github.io/python-genai/).

The package patches `google.genai.models.Models` and `AsyncModels` generation
methods and emits canonical Respan chat spans through the active OTEL pipeline.
It captures sync calls, async calls, streaming calls, prompt and completion
content, token usage, tool definitions, and model function calls.

## Installation

```bash
pip install respan-ai respan-instrumentation-google-genai google-genai
```

## Usage

```python
from google import genai
from respan import Respan
from respan_instrumentation_google_genai import GoogleGenAIInstrumentor

respan = Respan(instrumentations=[GoogleGenAIInstrumentor()])
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in three languages.",
)
print(response.text)
respan.flush()
```
