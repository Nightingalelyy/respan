<p align="center">
<a href="https://www.respan.ai#gh-light-mode-only">
<img width="800" src="https://cdn.respan.ai/Respan-Brand-Assets/wordmark-dark.png">
</a>
<a href="https://www.respan.ai#gh-dark-mode-only">
<img width="800" src="https://cdn.respan.ai/Respan-Brand-Assets/wordmark-light.png">
</a>
</p>
<p align="center">
  <p align="center">Observability, prompt management, and evals for LLM engineering teams.</p>
</p>

<div align="center">
  <a href="https://www.ycombinator.com/companies/respan"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange" alt="Y Combinator W24"></a>
  <a href="https://www.respan.ai"><img src="https://img.shields.io/badge/Platform-green.svg?style=flat-square" alt="Platform" style="height: 20px;"></a>
  <a href="https://www.respan.ai/docs/documentation/get-started/quickstart"><img src="https://img.shields.io/badge/Documentation-blue.svg?style=flat-square" alt="Documentation" style="height: 20px;"></a>
  <a href="https://x.com/respan/"><img src="https://img.shields.io/twitter/follow/respan?style=social" alt="Twitter" style="height: 20px;"></a>
  <a href="https://discord.com/invite/KEanfAafQQ"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="Discord" style="height: 20px;"></a>

</div>

# Respan Tracing
<div align="center">
<img src="https://cdn.respan.ai/respan_landing/respan/og-home-tracing.png" width="800"></img>
</div>

Respan's library for sending telemetries of LLM applications in [OpenLLMetry](https://github.com/traceloop/openllmetry) format.


## Integrations
<div align="center" style="background-color: white; padding: 20px; border-radius: 10px; margin: 0 auto; max-width: 800px;">
  <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 120px; margin-bottom: 20px;">
    <a href="https://www.respan.ai/docs/integrations/openai-agents-sdk"><img src="https://cdn.respan.ai/github/openai-agents-sdk.jpg" height="45" alt="OpenAI Agents SDK"></a>
        <a href="https://www.respan.ai/docs/integrations/langgraph"><img src="https://cdn.respan.ai/github/langgraph.jpg" height="45" alt="LangGraph"></a>
    <a href="https://www.respan.ai/docs/integrations/vercel-ai-sdk"><img src="https://cdn.respan.ai/github/vercel.jpg" height="45" alt="Vercel AI SDK"></a>
  </div>

</div>


## Configuration

### 1. Install

#### Python

```bash
pip install respan-ai openai
```

#### TypeScript/JavaScript

```bash
npm install @respan/respan openai
```

Replace `openai` with the provider SDK your application uses. The facade supplies the eligible first-party instrumentation adapters; TypeScript users should not install with `--omit=optional` because those adapters are optional dependencies of `@respan/respan`.

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. Get it from the [platform](https://platform.respan.ai/platform/api/api-keys). |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

The quickstart below routes OpenAI through the Respan gateway, so `RESPAN_API_KEY` authenticates both inference and telemetry export. When calling a provider directly, configure that provider's credentials and endpoint normally; Respan still uses `RESPAN_API_KEY` for telemetry.

## Quickstart

### 3. Run Script

#### Python
```python
import os
from openai import OpenAI
from respan import Respan

# Omit `instrumentations` to auto-discover the matching bundled adapter.
respan = Respan()

# Respan API key authenticates both proxy and tracing
respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

client = OpenAI(api_key=respan_api_key, base_url=respan_base_url)

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "Say hello in three languages."}],
)
print(response.choices[0].message.content)
respan.shutdown()
```

#### TypeScript/JavaScript
```typescript
import OpenAI from "openai";
import { Respan } from "@respan/respan";

const respanBaseURL =
  process.env.RESPAN_BASE_URL ?? "https://api.respan.ai/api";

// Omit `instrumentations` to auto-discover the matching bundled adapter.
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: respanBaseURL,
});
await respan.initialize();

const client = new OpenAI({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: respanBaseURL,
});

const response = await client.chat.completions.create({
  model: "gpt-4.1-nano",
  messages: [{ role: "user", content: "Say hello in three languages." }],
});
console.log(response.choices[0].message.content);
await respan.shutdown();
```

### 4. View Dashboard

See your traces in the [Respan platform](https://platform.respan.ai).

<div align="center">
<img src="https://cdn.respan.ai/github/traces-output.png" width="800"> </img>
</div>

## Further Reading

### Examples

- [Python OpenAI SDK examples](python-sdks/examples/openai-sdk/) — hello world, decorators, attributes, batch, streaming, tool calls
- [Python OpenAI Agents SDK examples](python-sdks/examples/openai-agents-sdk/) — hello world, handoffs, routing, guardrails
- [TypeScript OpenAI SDK examples](javascript-sdks/examples/openai-sdk/) — hello world, decorators, attributes

### Automatic Direct LLM Integrations

Install the facade and the provider SDK, omit the `instrumentations` option, and initialize Respan before the first provider call. Missing provider SDKs are skipped without failing startup and are visible through the instrumentation status API.

| Provider | Python SDK | TypeScript SDK |
|----------|------------|----------------|
| OpenAI | `openai` | `openai` |
| Azure OpenAI | `openai` | `openai` |
| Anthropic | `anthropic` | `@anthropic-ai/sdk` |
| Vertex AI | `google-cloud-aiplatform` | `@google-cloud/vertexai` |
| Google GenAI | `google-genai` | — |
| AWS Bedrock | `boto3` | `@aws-sdk/client-bedrock-runtime` |
| Cohere | — | `cohere-ai` |
| Together AI | `together` | `together-ai` |
| OpenRouter | — | `@openrouter/sdk` |
| Writer | — | `writer-sdk` |
| Ollama | `ollama` | — |

`respan-ai` bundles the seven eligible Python adapters. A normal `@respan/respan` install includes the nine TypeScript adapters as optional dependencies.

Inspect the result after construction in Python with `respan.get_auto_instrumentation_status()`, or after `await respan.initialize()` in TypeScript with `respan.getInstrumentationStatus()`.

### Explicit Framework and Wrapper Integrations

LLM wrappers, agent frameworks, application frameworks, protocol/tooling integrations, observability bridges, and vector databases remain explicit opt-ins to prevent duplicate spans. Supplying `instrumentations` selects explicit mode and disables automatic discovery by default.

Python can intentionally combine explicit plugins with direct-SDK automatic discovery by setting `is_auto_instrument=True`. Use `Respan(is_auto_instrument=False)` to disable automatic discovery entirely.

In TypeScript, use `disabledInstrumentations: ["openrouter"]` to disable one automatic adapter, or `instrumentations: []` to disable all automatic discovery. TypeScript explicit mode is currently exclusive and cannot be combined with automatic discovery.

### Workflow and Task Decorators

Structure traces with `@workflow` / `@task` (Python) or `withWorkflow` / `withTask` (TypeScript). See the [decorators example](python-sdks/examples/openai-sdk/decorators.py) for details.

### Propagate Attributes

Attach `customer_identifier`, `thread_identifier`, and `metadata` to all spans in scope. See the [attributes example](python-sdks/examples/openai-sdk/attributes.py).

## Star us
Please star us if you found this helpful!
