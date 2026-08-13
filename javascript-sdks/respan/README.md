# Respan TypeScript SDK

**[respan.ai](https://respan.ai)** | **[Documentation](https://www.respan.ai/docs)** | **[npm](https://www.npmjs.com/package/@respan/respan)**

`@respan/respan` is the unified TypeScript and JavaScript entry point for Respan tracing, lifecycle management, context propagation, and first-party instrumentation plugins. It automatically discovers eligible direct LLM SDK adapters while keeping frameworks, agents, wrappers, and observability bridges explicit to avoid duplicate spans.

## Installation

Install the facade and your provider SDK. For OpenAI:

```bash
npm install @respan/respan openai
```

Replace `openai` with the provider SDK your application uses. A normal `@respan/respan` install includes the supported first-party instrumentation adapters as optional dependencies. Do not use `--omit=optional` when you want automatic discovery.

## Automatic Onboarding

Set your Respan credentials:

```bash
export RESPAN_API_KEY="your-api-key-here"
# Optional; this is also the Respan gateway base URL used below.
export RESPAN_BASE_URL="https://api.respan.ai/api"
```

Construct and initialize `Respan` before the first provider call. TypeScript initialization is asynchronous and must be awaited:

```typescript
import OpenAI from "openai";
import { Respan } from "@respan/respan";

const respanBaseURL =
  process.env.RESPAN_BASE_URL ?? "https://api.respan.ai/api";

// Omit `instrumentations` to use curated direct-LLM automatic discovery.
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

console.table(respan.getInstrumentationStatus());
await respan.shutdown();
```

This example routes inference through the Respan gateway. When calling a provider directly, configure that provider's credentials and endpoint normally; Respan still reads `RESPAN_API_KEY` for telemetry export.

## Automatically Supported Providers

The facade supplies these adapters. Install only the provider SDKs your application actually imports.

| Provider | Application SDK |
|----------|-----------------|
| OpenAI | `openai` |
| Azure OpenAI | `openai` |
| Anthropic | `@anthropic-ai/sdk` |
| Vertex AI | `@google-cloud/vertexai` |
| OpenRouter | `@openrouter/sdk` |
| AWS Bedrock | `@aws-sdk/client-bedrock-runtime` |
| Cohere | `cohere-ai` |
| Together AI | `together-ai` |
| Writer | `writer-sdk` |

Only registry entries classified as direct LLM integrations and enabled by default are activated. A missing provider SDK or adapter is reported as `missing` and does not stop initialization. Respan also disables overlapping generic instrumentation names to prevent duplicate spans.

## Inspect Instrumentation Status

Call the status API after initialization:

```typescript
for (const entry of respan.getInstrumentationStatus()) {
  console.log(entry.id, entry.status, entry.reason);
}
```

Each automatic registry entry includes its ID, category, provider, provider SDK, instrumentation package, instrumentor class, status, and optional reason. Status is one of `enabled`, `disabled`, `missing`, or `failed`.

The status list describes automatic discovery. Explicit plugins are not currently added to this list.

## Control Automatic Discovery

Disable one automatic adapter by selector:

```typescript
const respan = new Respan({
  disabledInstrumentations: ["openrouter"],
});
await respan.initialize();
```

Selectors are case-insensitive and can match a registry ID, provider, SDK package, instrumentation package, class name, or alias. Prefer a provider's registry ID. Use `azure-openai` to target Azure specifically; because OpenAI and Azure OpenAI both use the `openai` SDK package, the selector `openai` matches both. Use `OpenAIInstrumentor` when you need to disable only the non-Azure OpenAI adapter.

Passing an empty explicit list disables all automatic discovery:

```typescript
const respan = new Respan({ instrumentations: [] });
await respan.initialize();
```

## Explicit Framework and Agent Integrations

Frameworks, agents, wrappers, protocol/tooling integrations, observability bridges, and vector databases remain explicit to avoid duplicate provider and framework spans.

For example, OpenAI Agents uses an explicit plugin:

```bash
npm install @respan/respan @respan/instrumentation-openai-agents @openai/agents
```

```typescript
import { Respan } from "@respan/respan";
import { OpenAIAgentsInstrumentor } from "@respan/instrumentation-openai-agents";

const respan = new Respan({
  instrumentations: [new OpenAIAgentsInstrumentor()],
});
await respan.initialize();
```

Supplying `instrumentations`, including an empty array, selects exclusive explicit mode and suppresses automatic discovery. TypeScript currently has no switch for combining explicit plugins with all automatic direct-SDK adapters.

## Lifecycle and Helpers

- `await respan.initialize()` starts telemetry and activates plugins. Call it before the first provider request.
- `await respan.flush()` exports buffered spans without deactivating plugins.
- `await respan.shutdown()` deactivates plugins and shuts down telemetry.
- `getInstrumentationStatus()` returns automatic discovery results.
- `withWorkflow`, `withTask`, `withAgent`, `withTool`, and `propagateAttributes` are re-exported from `@respan/tracing`.

## Examples

- [OpenAI SDK examples](../examples/openai-sdk/)
- [OpenAI Agents SDK example](../examples/openai-agents-sdk/)
- [Claude Agent SDK examples](../examples/claude-agent-sdk/)

## Public API

The package exports `Respan`, registry and status types, `OTELInstrumentor`, `OpenInferenceInstrumentor`, tracing helpers, the Respan client, the span buffer manager, and processor configuration types.

## License

Apache 2.0 — see the repository [LICENSE](../../LICENSE).

## Support

- Email: [team@respan.ai](mailto:team@respan.ai)
- Documentation: [https://www.respan.ai/docs](https://www.respan.ai/docs)
- Issues: [GitHub Issues](https://github.com/respanai/respan/issues)
