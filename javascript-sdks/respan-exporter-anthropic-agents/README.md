# Respan Exporter for Anthropic Agent SDK

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)**

Exporter for Anthropic Agent SDK telemetry to Respan.

## Configuration

### 1. Install

```bash
npm install @anthropic-ai/claude-agent-sdk @respan/exporter-anthropic-agents
npm install -D tsx
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Respan API key used for telemetry export. |
| `RESPAN_BASE_URL` | No | Respan base URL for telemetry export. Defaults to `https://api.respan.ai`. |
| `ANTHROPIC_BASE_URL` | No | Inference/proxy base URL used by the Anthropic SDK. |
| `ANTHROPIC_API_KEY` | Usually | Key used by the Anthropic SDK for inference calls. |
| `ANTHROPIC_AUTH_TOKEN` | Optional | Alternate auth token used by some Anthropic client flows. |

Set both groups together when needed. `RESPAN_*` controls tracing export, while `ANTHROPIC_*` controls where model requests are sent.

```bash
# Tracing export (Respan telemetry)
RESPAN_API_KEY=your_respan_key
RESPAN_BASE_URL=https://api.respan.ai/api

# Inference/proxy routing (Anthropic SDK)
# Optional: set only if you use a custom proxy/gateway base URL
# ANTHROPIC_BASE_URL=https://your-anthropic-base-url
ANTHROPIC_API_KEY=your_inference_key
ANTHROPIC_AUTH_TOKEN=your_inference_key
```

`RESPAN_BASE_URL` controls telemetry export only. In normal usage, instantiate `RespanAnthropicAgentsExporter()` with no arguments and configure via environment variables.
`ANTHROPIC_BASE_URL` is optional. If you use a gateway/proxy, set it to that gateway's Anthropic-compatible base URL.

## Quickstart

### 3. Run Script

Save this as `quickstart.ts`:

```typescript
import { RespanAnthropicAgentsExporter } from "@respan/exporter-anthropic-agents";

const respanApiKey = process.env.RESPAN_API_KEY!;
const anthropicBaseUrl = process.env.ANTHROPIC_BASE_URL;
const anthropicApiKey = process.env.ANTHROPIC_API_KEY ?? respanApiKey;
const anthropicAuthToken = process.env.ANTHROPIC_AUTH_TOKEN ?? anthropicApiKey;

const exporter = new RespanAnthropicAgentsExporter();
const anthropicEnv = {
  ANTHROPIC_API_KEY: anthropicApiKey,
  ANTHROPIC_AUTH_TOKEN: anthropicAuthToken,
  ...(anthropicBaseUrl ? { ANTHROPIC_BASE_URL: anthropicBaseUrl } : {}),
};

for await (const message of exporter.query({
  prompt: "Review this repository and summarize architecture.",
  options: {
    allowedTools: ["Read", "Glob", "Grep"],
    permissionMode: "acceptEdits",
    env: anthropicEnv,
  },
})) {
  console.log(message);
}
```

Run it:

```bash
npx tsx quickstart.ts
```

### 4. View Dashboard

Open:

- `https://platform.respan.ai/platform/traces`

## Further Reading

Runnable examples with full setup instructions:

- **TypeScript examples root:** [typescript/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/typescript/tracing/anthropic-agents-sdk)
- **TypeScript basic scripts:**
  - [hello_world_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/hello_world_test.ts)
  - [wrapped_query_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/wrapped_query_test.ts)
  - [tool_use_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/tool_use_test.ts)
  - [gateway_test.ts](https://github.com/respanai/respan-example-projects/blob/main/typescript/tracing/anthropic-agents-sdk/gateway_test.ts)
- **Python examples root:** [python/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/anthropic-agents-sdk)

## Dev Guide

### Running Tests

```bash
npm test
```
