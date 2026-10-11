# Respan Together AI instrumentation

Instrument the actual `together-ai` SDK, verified with 0.40.0 and the current
0.58.0 release. Initialize Respan before making SDK calls.

```ts
import Together from "together-ai";
import { Respan } from "@respan/respan";
import { TogetherAIInstrumentor } from "@respan/instrumentation-together-ai";

const instrumentor = new TogetherAIInstrumentor();
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [instrumentor],
});
await respan.initialize();
const client = new Together({ apiKey: process.env.TOGETHER_API_KEY });
await client.chat.completions.create({
  model: "your-together-model",
  messages: [{ role: "user", content: "Explain distributed tracing." }],
});
await respan.shutdown();
```

| SDK surface                                                | Captured behavior                                                                                                                                   |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chat.completions.create`                                  | Complete messages and multimodal blocks, all response choices, separate reasoning fields, current-turn tool calls and native IDs, full tool schemas |
| Chat/text streams                                          | Native chunks and controller remain available; choices and reasoning remain separate; full consumption or early return finishes one span            |
| `completions.create`                                       | Actual prompts, text choices and reported usage                                                                                                     |
| `embeddings.create`                                        | Actual inputs and full vectors, including 5001 dimensions; reported usage                                                                           |
| `images.generate`, `rerank.create`                         | Actual request and returned payloads                                                                                                                |
| `audio.speech.create`                                      | Native binary `Response` remains unread; response metadata is recorded; streamed chunk payloads are retained when available                         |
| `audio.transcriptions.create`, `audio.translations.create` | Actual upload metadata and returned transcription/translation payloads                                                                              |

Native APIPromise objects and their `then`, `catch`, `finally`, `withResponse`
and `asResponse` helpers remain usable. `asResponse` leaves the HTTP body
unread and does not invent a parsed completion. Native errors retain their
identity and available numeric HTTP status. Missing token totals are left
missing. Native cache/reasoning usage fields are captured when returned.

Historical assistant tool calls and tool results remain input context. They
do not create fictional tool execution spans. Wrap a real application tool
callback with `respan.withTool` when its execution should appear in the tree.
The nine mapped examples demonstrate an actual callback and native call ID.

```ts
const instrumentor = new TogetherAIInstrumentor({
  traceContent: false,
  recordInputs: false,
  recordOutputs: false,
});
console.log(instrumentor.isActive());
await instrumentor.activate();
instrumentor.deactivate();
```

Constructor opt-outs, `RESPAN_TRACE_CONTENT=false`,
`TRACELOOP_TRACE_CONTENT=false`, the canonical Traceloop content context and
an observed ancestor's `allow_trace_content=false` form immutable capture
ceilings. Input and output lanes can be disabled independently. A later
`true` cannot restore denied content. The real tracer sampler runs before
adapter payload traversal; DROP and unsampled RECORD_ONLY paths do not copy
payloads. Actual recording-span attributes, events and status are guarded.
The SDK still serializes the inputs required to execute a request.

Configure the instrumentor and the public runtime consistently when disabling
content. Runtime decorator capture is separate from explicit plugin options;
use the environment flag when a whole workflow must omit content.

Concurrent owners share resource patches and apply the most restrictive
active capture settings. Final deactivation restores only methods still owned
by this package. Already admitted calls and consumed streams drain normally.
Consume streams or call the iterator's `return()` to close their spans.
Canonical bodies, tools, output and usage are assigned before optional indexed
message attributes, so the default OTel attribute budget does not discard
complete payloads. Administrative endpoints, jobs and fine-tuning are outside
this adapter's inference scope.

Tests use the released native SDK with controlled fetch/SSE transports.
Feature-compatible examples are in
`respan-example-projects/typescript/tracing/together-ai` and use a local trace
collector by default. Exporting those traces and live inference are separate
explicit modes.
