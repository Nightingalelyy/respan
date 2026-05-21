# respan-instrumentation-google-adk

Respan instrumentation plugin for [Google Agent Development Kit](https://adk.dev/).

This package wraps the upstream OpenInference Google ADK instrumentor and
registers a Google-ADK-specific span processor. The processor composes Respan's
generic OpenInference translation and applies ADK-only normalization in this
package, so ADK runner, agent, LLM, and tool spans are exported through the same
Respan OTEL pipeline as the rest of the Python SDK.

## Installation

```bash
pip install respan-ai respan-instrumentation-google-adk
```

Install ADK's LiteLLM extension if you want to route models through an
OpenAI-compatible gateway:

```bash
pip install "google-adk[extensions]"
```

## Usage

```python
from respan import Respan
from respan_instrumentation_google_adk import GoogleADKInstrumentor

respan = Respan(instrumentations=[GoogleADKInstrumentor()])
```

Any Google ADK runs started after initialization are traced and exported to
Respan.
