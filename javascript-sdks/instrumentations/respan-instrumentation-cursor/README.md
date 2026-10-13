# Respan Cursor SDK Instrumentation

Trace the native `@cursor/sdk` agent and tool lifecycle through the active Respan OpenTelemetry provider.

```ts
import * as CursorSDK from "@cursor/sdk";
import { Respan } from "@respan/respan";
import { CursorSDKInstrumentor } from "@respan/instrumentation-cursor";

const instrumentor = new CursorSDKInstrumentor({
  sdkModule: { ...CursorSDK },
  agentName: "repository-agent",
});
const respan = new Respan({ instrumentations: [instrumentor] });
await respan.initialize();

const agent = await CursorSDK.Agent.create({
  apiKey: process.env.CURSOR_API_KEY!,
  model: { id: "composer-2.5" },
  cloud: {},
});
const run = await agent.send("Summarize this repository.");
for await (const event of run.stream()) console.log(event.type);
await run.wait();
await agent[Symbol.asyncDispose]();
await respan.flush();
```

The ESM namespace stays immutable. The mutable module copy contains the actual native `Agent` class, whose methods are patched while the plugin is active. `activate()`, `deactivate()`, and `isActive()` support explicit lifecycle management. Multiple owners share one patch; final deactivation restores owned methods and lets existing run handles drain.

## Supported APIs

- `Agent.create()`, `Agent.resume()`, and `SDKAgent.send()` create one live agent span per run.
- `Agent.prompt()` shares that span across its internal create/send/wait calls.
- `Agent.getRun()` records the native detached handle when consumed.
- `Run.stream()`, `wait()`, `conversation()`, and `cancel()` preserve native results and errors. Stream generators retain their identity, `return()`, and `throw()` methods. After an early stream break, call `wait()` or `cancel()` to finish the run.
- Native tool-call events produce tool spans with the actual tool-call ID, arguments, and result. Callback and stream observations of the same call are deduplicated. Pending tools end with an error when their run ends without a result.
- Local custom tools retain input/output schemas, annotations, callback context, results, and thrown errors. Agent-level defaults and per-send overrides are supported.

An agent run can include several model turns. Its model selection and reported aggregate token usage belong in canonical `respan.metadata` as `cursor_model` and `cursor_usage`; this adapter does not fabricate a provider LLM span from that aggregate. Native agent, run, and request identifiers are metadata. Tool schemas use `cursor_tools` in the same metadata object. Model selection identifies the Cursor agent configuration and does not establish its underlying model provider.

Streaming captures legacy cloud tool frames. Wait-only calls capture tool observations when the native SDK provides `onDelta` updates; older cloud streams that provide only legacy frames require consuming `stream()` for individual tool spans. Native schema and event availability determine capture; the adapter does not fetch account billing or infer absent values.

## Versions and privacy

Node.js **22.13 or newer** is required by Cursor. The adapter supports the earliest currently published stable SDK **1.0.7** and is validated against **1.0.37**. SDK 1.0.7 has synchronous `Agent.create()`; the adapter preserves that return shape. Native token usage arrived in 1.0.22, custom-tool output schemas in 1.0.29, annotations in 1.0.31, and custom-tool session IDs in 1.0.35. Fields unavailable on an older native SDK remain absent.

Set `traceContent: false` on the instrumentor, `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, or the canonical Traceloop content context to prevent payload capture. These are immutable ceilings for an existing run. Suppression, DROP, and unsampled RECORD_ONLY spans do not inspect instrumentation payloads. The actual live spans guard late attributes, events, exceptions, and status messages. Native SDK processing of its own request is unchanged.

For an application-wide opt-out, configure the same content choice on the public Respan runtime and this plugin. Existing admitted spans retain their privacy decision when the exporter runs under ambient tracing suppression.

## Validation

The native regression suite uses the released Cursor package against an owned HTTP/SSE endpoint. Local custom-tool checks retain real `SDKAgent` and `Run` objects and replace the public `CursorAgentPlatform.acquireLocalExecutor()` transport with a deterministic executor. This tests SDK event and callback production without model inference, workspace edits, or Cursor cloud access.

The mapped examples in `typescript/tracing/cursor-sdk` cover prompt, multimodal streaming, large tool payloads, resume/detached handles, current local custom-tool features, controlled errors, and content opt-out. Live Cursor service behavior requires a separate authenticated run.
