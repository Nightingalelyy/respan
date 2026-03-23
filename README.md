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


## Quickstart

### 1. Get an API key
Go to Respan platform and [get your API key](https://platform.respan.ai/platform/api/api-keys).

### 2. Install

#### Python

```bash
pip install respan respan-tracing respan-instrumentation-openai
```

#### TypeScript/JavaScript

```bash
npm install @respan/respan @respan/tracing @respan/instrumentation-openai openai @traceloop/instrumentation-openai
```

### 3. Trace your LLM calls

#### Python
```python
import os
from openai import OpenAI
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)

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

### 4. Structure traces with workflows and tasks

#### Python
```python
from respan import Respan, workflow, task
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(instrumentations=[OpenAIInstrumentor()])

@task(name="generate_outline")
def generate_outline(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": topic}],
    )
    return response.choices[0].message.content

@workflow(name="content_pipeline")
def run(topic: str):
    outline = generate_outline(topic)
    return outline

run("Benefits of open-source software")
respan.flush()
```

#### TypeScript/JavaScript
```typescript
async function generateOutline(topic: string) {
  return respan.withTask({ name: "generate_outline" }, async () => {
    const response = await client.chat.completions.create({
      model: "gpt-4.1-nano",
      messages: [{ role: "user", content: topic }],
    });
    return response.choices[0].message.content;
  });
}

async function run(topic: string) {
  return respan.withWorkflow({ name: "content_pipeline" }, async () => {
    return generateOutline(topic);
  });
}

await run("Benefits of open-source software");
await respan.flush();
```

### 5. Attach customer info with `propagateAttributes`

#### Python
```python
from respan import propagate_attributes

with propagate_attributes(customer_identifier="user_123", thread_identifier="conv_001"):
    response = client.chat.completions.create(...)
```

#### TypeScript/JavaScript
```typescript
await respan.propagateAttributes(
  { customer_identifier: "user_123", thread_identifier: "conv_001" },
  async () => {
    const response = await client.chat.completions.create(...);
  }
);
```

### 6. See traces in [Respan](https://www.respan.ai)
<div align="center">
<img src="https://respan-static.s3.us-east-1.amazonaws.com/github/traces-output.png" width="800"> </img>
</div>

## Supported Integrations

The plugin system supports 50+ tools via OTEL instrumentation wrappers:

| Package | Python | TypeScript |
|---------|--------|------------|
| OpenAI SDK | `respan-instrumentation-openai` | `@respan/instrumentation-openai` |
| OpenAI Agents SDK | `respan-instrumentation-openai-agents` | `@respan/instrumentation-openai-agents` |
| Anthropic SDK | `respan-instrumentation-anthropic` | `@respan/instrumentation-anthropic` |
| OpenInference (Arize) | `respan-instrumentation-openinference` | `@respan/instrumentation-openinference` |
| Any OTEL instrumentor | `OTELInstrumentor(cls)` | `new OTELInstrumentor(cls)` |

Auto-discovery also supports: Azure OpenAI, Cohere, Bedrock, Vertex AI, LangChain, LlamaIndex, Pinecone, ChromaDB, Qdrant, Together AI, and more.

## Star us
Please star us if you found this helpful!

## Examples

- [Python OpenAI SDK examples](python-sdks/examples/openai-sdk/)
- [Python OpenAI Agents SDK examples](python-sdks/examples/openai-agents-sdk/)
- [TypeScript OpenAI SDK examples](javascript-sdks/examples/openai-sdk/)
