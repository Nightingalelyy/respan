# respan-instrumentation-pipecat

Respan instrumentation plugin for Pipecat. Wraps OpenInference's upstream
Pipecat instrumentor and translates Pipecat spans locally into the Respan
tracing shape automatically.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-pipecat
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

```python
import os

from respan import Respan
from respan_instrumentation_pipecat import PipecatInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[PipecatInstrumentor()],
)

# Create PipelineTask instances after Respan activates the instrumentor.
# Pipecat pipeline, LLM, STT, TTS, and tool spans are captured automatically.

respan.flush()
```

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/pipecat) for runnable scripts.
