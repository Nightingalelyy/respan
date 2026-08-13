# Respan Python SDK

**[respan.ai](https://respan.ai)** | **[Documentation](https://www.respan.ai/docs)** | **[PyPI](https://pypi.org/project/respan-ai/)**

`respan-ai` is the unified Python entry point for Respan tracing, decorators, context propagation, and first-party instrumentation plugins. It automatically discovers eligible direct LLM SDK adapters while keeping frameworks, agents, wrappers, and observability bridges explicit to avoid duplicate spans.

## Requirements

- Python 3.11, 3.12, or 3.13
- A Respan API key for telemetry export
- The provider SDK used by your application

## Installation

Install the facade and your provider SDK. For OpenAI:

```bash
pip install respan-ai openai
```

With Poetry:

```bash
poetry add respan-ai openai
```

Replace `openai` with the provider SDK your application uses. `respan-ai` already bundles the supported first-party instrumentation adapters and `respan-tracing`; provider SDKs remain application-owned dependencies.

## Automatic Onboarding

Set your Respan credentials:

```bash
export RESPAN_API_KEY="your-api-key-here"
# Optional; this is also the Respan gateway base URL used below.
export RESPAN_BASE_URL="https://api.respan.ai/api"
```

Construct `Respan` before the first provider call. Python discovery and activation happen synchronously in the constructor, so there is no separate `initialize()` call:

```python
import os

from openai import OpenAI
from respan import Respan

# Omit `instrumentations` to use curated direct-LLM automatic discovery.
respan = Respan()

client = OpenAI(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "Say hello in three languages."}],
)
print(response.choices[0].message.content)

print(respan.get_auto_instrumentation_status())
respan.shutdown()
```

This example routes inference through the Respan gateway. When calling a provider directly, configure that provider's credentials and endpoint normally; Respan still reads `RESPAN_API_KEY` for telemetry export.

## Automatically Supported Providers

The facade bundles these adapters. Install only the provider SDKs your application actually imports.

| Provider | Application SDK |
|----------|-----------------|
| OpenAI and Azure OpenAI | `openai` |
| Anthropic | `anthropic` |
| Vertex AI | `google-cloud-aiplatform` |
| Google GenAI | `google-genai` |
| AWS Bedrock | `boto3` |
| Together AI | `together` |
| Ollama | `ollama` |

Only registry entries classified as direct LLM integrations and enabled by default are activated. A missing provider SDK is reported as `missing` and does not stop application startup. Broad OpenTelemetry entry-point discovery remains disabled.

## Inspect Instrumentation Status

```python
for entry in respan.get_auto_instrumentation_status():
    print(entry["id"], entry["status"], entry.get("reason"))
```

Each entry includes the registry ID, runtime name, provider, provider SDK, instrumentation package, status, and optional reason. Status is one of `enabled`, `disabled`, `missing`, or `failed`.

The dataclass form is also available as `respan.auto_instrumentation_status`.

## Control Automatic Discovery

Disable automatic discovery completely:

```python
from respan import Respan

respan = Respan(is_auto_instrument=False)
```

Supplying an explicit instrumentation list selects explicit mode and disables automatic discovery by default:

```bash
pip install respan-ai respan-instrumentation-openai-agents openai-agents
```

```python
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

respan = Respan(
    instrumentations=[OpenAIAgentsInstrumentor()],
)
```

To deliberately combine an explicit plugin with direct-SDK automatic discovery:

```python
respan = Respan(
    instrumentations=[OpenAIAgentsInstrumentor()],
    is_auto_instrument=True,
)
```

Explicit plugins activate first, so a matching automatic adapter is skipped rather than activated twice. Python currently provides a global automatic-discovery switch, but not a per-provider disable option.

## Tracing Helpers

The facade re-exports Respan's tracing helpers:

```python
from respan import agent, respan_span_attributes, task, tool, workflow
```

Use `@workflow`, `@task`, `@agent`, and `@tool` to structure traces, and use `respan_span_attributes` or `propagate_attributes` to attach customer, session, thread, environment, and metadata values.

## Lifecycle

- `Respan()` initializes telemetry and activates eligible adapters.
- `respan.flush()` exports buffered spans without deactivating adapters.
- `respan.shutdown()` deactivates adapters and flushes telemetry.

## Examples

- [OpenAI SDK examples](../examples/openai-sdk/)
- [OpenAI Agents SDK examples](../examples/openai-agents-sdk/)
- [CrewAI example](../examples/crewai/)

## Public API

The package exports `Respan`, the automatic instrumentation registry and status types, `OTELInstrumentor`, the instrumentation protocol, tracing decorators, `RespanClient`, `get_client`, `respan_span_attributes`, and `propagate_attributes`.

## License

Apache 2.0 — see the repository [LICENSE](../../LICENSE).

## Support

- Email: [team@respan.ai](mailto:team@respan.ai)
- Documentation: [https://www.respan.ai/docs](https://www.respan.ai/docs)
- Issues: [GitHub Issues](https://github.com/respanai/respan/issues)
