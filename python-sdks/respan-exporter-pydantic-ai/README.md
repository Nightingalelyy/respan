# Respan Exporter for Pydantic AI

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-pydantic-ai/)**

This package provides a Respan exporter for the [Pydantic AI](https://ai.pydantic.dev/) framework.
It seamlessly instruments Pydantic AI's agents using OpenTelemetry underneath so that all traces, spans, and metrics
are sent to Respan using standard semantic conventions.

**Requirements:** This package requires [respan-tracing](https://pypi.org/project/respan-tracing/) for telemetry setup; it is installed automatically as a dependency. For a full install from PyPI: `pip install respan-exporter-pydantic-ai respan-tracing`.

## Installation

```bash
pip install respan-exporter-pydantic-ai
```

(`respan-tracing` is installed automatically as a dependency.)

## Configuration

You can pass the Respan API key explicitly or use environment variables:

| Option | Description |
|--------|-------------|
| `RESPAN_API_KEY` | API key for Respan (used when `api_key` is not passed to `RespanTelemetry`) |
| `RESPAN_BASE_URL` | Optional; API base URL (default: `https://api.respan.ai/api`) |

Example: `export RESPAN_API_KEY="your-respan-key"` so you don't need to pass `api_key` inline.

## Usage

```python
from pydantic_ai.agent import Agent
from respan_tracing import RespanTelemetry
from respan_exporter_pydantic_ai import instrument_pydantic_ai

# 1. Initialize Respan Telemetry (required)
# Pass api_key or set RESPAN_API_KEY in the environment
telemetry = RespanTelemetry(app_name="my-app", api_key="YOUR_RESPAN_API_KEY")

# 2. Instrument Pydantic AI
instrument_pydantic_ai()

# 3. Create and use your Agent
agent = Agent('openai:gpt-4o')

result = agent.run_sync('What is the capital of France?')
print(result.output)
```

**Tested with:** Pydantic AI 0.x using the `InstrumentationSettings` API (e.g. `version=2`). If you use an older Pydantic AI release, behavior may differ.

## Instrumenting Specific Agents

If you only want to instrument specific agents instead of globally, initialize Respan telemetry first, then instrument the agent:

```python
from respan_tracing import RespanTelemetry
from respan_exporter_pydantic_ai import instrument_pydantic_ai
from pydantic_ai.agent import Agent

# After initializing RespanTelemetry as shown above:
telemetry = RespanTelemetry(app_name="my-app", api_key="YOUR_RESPAN_API_KEY")

agent = Agent('openai:gpt-4o')
instrument_pydantic_ai(agent=agent)
```

## Development

### Setup

Clone the repo and install the package with its local dependencies:

```bash
cd python-sdks/respan-exporter-pydantic-ai
pip install -e ../respan-tracing -e .
```

### Running Tests

**Unit tests** — verify instrumentation wiring without network calls:

```bash
pytest tests/test_instrument.py -v
```

**Integration test** — sends a real LLM call through Respan and verifies spans are captured. Requires API keys:

```bash
IS_REAL_GATEWAY_TESTING_ENABLED=1 \
RESPAN_API_KEY="your-respan-key" \
OPENAI_API_KEY="your-openai-key" \
pytest tests/test_real_gateway_integration.py -v
```

The integration test is skipped by default. Set `IS_REAL_GATEWAY_TESTING_ENABLED=1` to opt in.

**All tests:**

```bash
pytest tests/ -v
```

Integration tests auto-skip when env vars are not set, so this is always safe to run.
