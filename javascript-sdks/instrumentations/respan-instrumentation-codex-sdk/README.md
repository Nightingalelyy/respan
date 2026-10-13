# Respan Codex SDK Instrumentation

Trace the actual turns and items exposed by `@openai/codex-sdk` 0.141.0 through
0.162.1 with the active Respan OpenTelemetry pipeline.

```ts
import { Codex } from "@openai/codex-sdk";
import { Respan } from "@respan/respan";
import { CodexSDKInstrumentor } from "@respan/instrumentation-codex-sdk";

const instrumentor = new CodexSDKInstrumentor();
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [instrumentor],
});
await respan.initialize();
const thread = new Codex().startThread();
await thread.run("Summarize the repository.");
await respan.shutdown();
```

Each native turn becomes one `agent` span. Actual command, MCP, and web search
items become `tool` children; reasoning summaries, file changes, todo lists,
and nonfatal error items become `task` children. Agent messages provide the
turn output. No additional workflow or LLM request spans are invented.

Codex exposes aggregate turn usage rather than individual provider requests.
The agent's canonical JSON `respan.metadata` preserves `codex_turn_usage`
(input, output, cache-read, reasoning, and, on current releases, cache-write
counts) and the explicitly configured `codex_model`. These values are not
reported as per-request GenAI usage or a resolved response model. Resuming a
thread preserves its native thread identifier; it does not reconstruct prior
conversation history that the SDK does not expose.

Text and local-image input parts retain their order and structure. MCP results
retain their content blocks, structured content, and `_meta`. Tool inputs use
`{name, arguments}`. Canonical metadata preserves each generic `codex_item_id`.
The SDK does not expose an invocation/call ID, so `gen_ai.tool.call.id` is omitted.
Web search items contain a query but no search-result payload, so no result is
fabricated. Buffered and streamed failures preserve native errors and partial
items already observed. Nonfatal error items mark their task without failing
a successful turn.

```ts
const instrumentor = new CodexSDKInstrumentor({
  agentName: "repository-assistant",
  workflowName: "repository-maintenance", // Optional trace group identifier.
  captureItemSpans: true,
  traceContent: false,
});
await instrumentor.activate();
console.log(instrumentor.isActive());
instrumentor.deactivate();
```

`traceContent: false`, `RESPAN_TRACE_CONTENT=false`,
`TRACELOOP_TRACE_CONTENT=false`, the canonical Traceloop content context, or
an observed ancestor's `allow_trace_content=false` prevents adapter payload
capture. Once observed, a veto cannot be restored by a later `true`. Samplers
run before adapter input traversal. DROP and unsampled RECORD_ONLY spans do
not capture content. Actual recording-span attributes, events, and status are
also guarded; safe native identifiers and aggregate numeric usage remain.
The native SDK must still read/serialize the inputs required to execute the
requested operation.

The public `run` and `runStreamed` results, errors, event objects, and iterator
identity are preserved. A tested hook around the SDK's internal
`runStreamedInternal` generator observes buffered partial items without
reimplementing `Thread.run`. This internal hook is verified at both supported
versions; future incompatible SDK versions may require another update.

Concurrent owners share one patch per actual Thread prototype. Any owner's
content opt-out applies to the shared patch. Deactivation restores methods
only while they remain owned by this package, and in-flight turns drain using
the original functions captured at activation. Consume streamed turns fully
or call the iterator's `return()` to finish the admitted span.

## Validation

The tests use the actual released Codex SDK and its public `codexPathOverride`
option with a controlled JSONL subprocess transport. This validates SDK
parsing, options, iteration, and tracing without model inference or execution
of the native Codex CLI. The executable fixture requires a Unix-like platform
and Node on `PATH`. Feature-compatible examples are in
`respan-example-projects/typescript/tracing/codex-sdk`.
