<p align="center">
<a href="https://www.respan.ai#gh-light-mode-only">
<img width="800" src="https://respan-static.s3.us-east-1.amazonaws.com/social_media_images/logo-header.jpg">
</a>
<a href="https://www.respan.ai#gh-dark-mode-only">
<img width="800" src="https://respan-static.s3.us-east-1.amazonaws.com/social_media_images/logo-header-dark.jpg">
</a>
</p>
<p align="center">
  <p align="center">Observability, prompt management, and evals for LLM engineering teams.</p>
</p>

<div align="center">
  <a href="https://www.ycombinator.com/companies/respan"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange" alt="Y Combinator W24"></a>
  <a href="https://www.respan.ai"><img src="https://img.shields.io/badge/Platform-green.svg?style=flat-square" alt="Platform" style="height: 20px;"></a>
  <a href="https://docs.respan.ai/get-started/overview"><img src="https://img.shields.io/badge/Documentation-blue.svg?style=flat-square" alt="Documentation" style="height: 20px;"></a>
  <a href="https://x.com/respan/"><img src="https://img.shields.io/twitter/follow/respan?style=social" alt="Twitter" style="height: 20px;"></a>
  <a href="https://discord.com/invite/KEanfAafQQ"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="Discord" style="height: 20px;"></a>

</div>

# Respan Tracing
<div align="center">
<img src="https://respan-static.s3.us-east-1.amazonaws.com/social_media_images/github-cover.jpg" width="800"></img>
</div>

Respan's library for sending telemetries of LLM applications in [OpenLLMetry](https://github.com/traceloop/openllmetry) format.


## Integrations
<div align="center" style="background-color: white; padding: 20px; border-radius: 10px; margin: 0 auto; max-width: 800px;">
  <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 120px; margin-bottom: 20px;">
    <a href="https://docs.respan.ai/features/monitoring/traces/integrations/openai-agents-sdk"><img src="https://respan-static.s3.us-east-1.amazonaws.com/github/openai-agents-sdk.jpg" height="45" alt="OpenAI Agents SDK"></a>
        <a href="https://docs.respan.ai/features/monitoring/traces/integrations/langgraph"><img src="https://respan-static.s3.us-east-1.amazonaws.com/github/langgraph.jpg" height="45" alt="LangGraph"></a>
    <a href="https://docs.respan.ai/features/monitoring/traces/integrations/vercel-ai-sdk"><img src="https://respan-static.s3.us-east-1.amazonaws.com/github/vercel.jpg" height="45" alt="Vercel AI SDK"></a>
  </div>

</div>


## Configuration

### 1. Install

#### Python

```bash
pip install respan respan-tracing respan-instrumentation-openai
```

#### TypeScript/JavaScript

```bash
npm install @respan/respan @respan/tracing @respan/instrumentation-openai openai @traceloop/instrumentation-openai
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. Get it from the [platform](https://platform.respan.ai/platform/api/api-keys). |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

The Respan API key is used for both LLM inference (proxy) and telemetry export (tracing). Vendor-specific keys (OPENAI_API_KEY, etc.) are derived from the Respan key in code.

## Quickstart

### 3. Run Script

#### Python
```python
import os
from openai import OpenAI
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

# Respan API key authenticates both proxy and tracing
respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

client = OpenAI(api_key=respan_api_key, base_url=respan_base_url)

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "Say hello in three languages."}],
)
print(response.choices[0].message.content)
respan.flush()
```

#### TypeScript/JavaScript
```typescript
import OpenAI from "openai";
import { Respan } from "@respan/respan";
import { OpenAIInstrumentor } from "@respan/instrumentation-openai";

// Respan API key authenticates both proxy and tracing
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [new OpenAIInstrumentor()],
});
await respan.initialize();

const client = new OpenAI({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
});

const response = await client.chat.completions.create({
  model: "gpt-4.1-nano",
  messages: [{ role: "user", content: "Say hello in three languages." }],
});
console.log(response.choices[0].message.content);
await respan.flush();
```

### 4. View Dashboard

See your traces in the [Respan platform](https://platform.respan.ai).

<div align="center">
<img src="https://respan-static.s3.us-east-1.amazonaws.com/github/traces-output.png" width="800"> </img>
</div>

## Further Reading

### Examples

- [Python OpenAI SDK examples](python-sdks/examples/openai-sdk/) — hello world, decorators, attributes, batch, streaming, tool calls
- [Python OpenAI Agents SDK examples](python-sdks/examples/openai-agents-sdk/) — hello world, handoffs, routing, guardrails
- [TypeScript OpenAI SDK examples](javascript-sdks/examples/openai-sdk/) — hello world, decorators, attributes

### Supported Integrations

The plugin system supports 50+ tools via OTEL instrumentation wrappers:

| Package | Python | TypeScript |
|---------|--------|------------|
| OpenAI SDK | `respan-instrumentation-openai` | `@respan/instrumentation-openai` |
| OpenAI Agents SDK | `respan-instrumentation-openai-agents` | `@respan/instrumentation-openai-agents` |
| Anthropic SDK | `respan-instrumentation-anthropic` | `@respan/instrumentation-anthropic` |
| OpenInference (Arize) | `respan-instrumentation-openinference` | `@respan/instrumentation-openinference` |
| Any OTEL instrumentor | `OTELInstrumentor(cls)` | `new OTELInstrumentor(cls)` |

Auto-discovery also supports: Azure OpenAI, Cohere, Bedrock, Vertex AI, LangChain, LlamaIndex, Pinecone, ChromaDB, Qdrant, Together AI, and more.

### Workflow and Task Decorators

Structure traces with `@workflow` / `@task` (Python) or `withWorkflow` / `withTask` (TypeScript). See the [decorators example](python-sdks/examples/openai-sdk/decorators.py) for details.

### Propagate Attributes

Attach `customer_identifier`, `thread_identifier`, and `metadata` to all spans in scope. See the [attributes example](python-sdks/examples/openai-sdk/attributes.py).

## Star us
Please star us if you found this helpful!
