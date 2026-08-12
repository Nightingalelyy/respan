# @respan/instrumentation-livekit

Respan instrumentation for the LiveKit Agents TypeScript SDK.

The instrumentor registers a span transformer with the active Respan tracing
runtime and points LiveKit's dynamic tracer at that OpenTelemetry provider
during `activate()`. It does not patch OpenTelemetry providers or LiveKit span
methods.

```ts
import { Respan } from "@respan/respan";
import { LiveKitInstrumentor } from "@respan/instrumentation-livekit";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new LiveKitInstrumentor()],
});

await respan.initialize();
```

Captured LiveKit spans include agent sessions, agent turns, LLM nodes, function
tools, user turns, and TTS spans. LiveKit raw `lk.*` attributes are consumed
during translation and removed before export, while Respan canonical fields
such as `respan.entity.log_type`, `traceloop.entity.input`, `gen_ai.prompt.*`,
and `gen_ai.completion.0.content` are added for ingestion.

Useful transport correlation values such as provider request IDs are retained
inside the canonical `respan.metadata` JSON object.

LiveKit emits a transport-level `llm_request` beneath each logical `llm_node`.
The transformer merges request model, provider, token, and timing details into
the node, drops only that correlated wrapper in semantic-name mode, and keeps
retry/error attempt spans visible. Standalone `llm_request` spans remain chats.
