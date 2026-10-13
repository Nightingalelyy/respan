# Respan Vertex AI instrumentation

Trace the Vertex backend of the current Google GenAI SDK, verified with
`@google/genai` 1.45.0 and 2.28.0. Existing `@google-cloud/vertexai` APIs remain
supported and tested with 1.1.0 and 1.12.0.

Google's [migration guide](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk)
retires the legacy generative SDK after June 24, 2026 and directs JavaScript
users to `@google/genai`. Keeping legacy API compatibility does not extend
Google's support period or establish live availability of a model.

```ts
import * as genai from "@google/genai";
import { Respan } from "@respan/respan";
import { VertexAIInstrumentor } from "@respan/instrumentation-vertexai";

const instrumentor = new VertexAIInstrumentor({ genaiModule: genai });
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [instrumentor],
});
await respan.initialize();
const client = new genai.GoogleGenAI({
  vertexai: true,
  project: process.env.GOOGLE_CLOUD_PROJECT,
  location: process.env.GOOGLE_CLOUD_LOCATION ?? "us-central1",
});
await client.models.generateContent({
  model: "your-vertex-model",
  contents: "Explain distributed tracing.",
});
await respan.shutdown();
```

| Native SDK surface                                    | Support                                                                                                                                                              |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GenAI `models.generateContent`                        | Complete messages, system instructions, tool schemas/calls and actual IDs; all candidate parts, reasoning/signatures, returned image/audio blocks and reported usage |
| GenAI `models.generateContentStream`                  | Native async generator identity and chunks; complete parts per candidate; early return and native errors                                                             |
| GenAI `chats` and automatic function calling          | Each actual model request is traced; native history, tool arguments/results and IDs survive; callbacks remain owned by the SDK                                       |
| GenAI `models.embedContent`                           | Actual input and complete native embedding values, including 5001 dimensions                                                                                         |
| GenAI image generation/edit/upscale/recontext/segment | Actual inference inputs and native typed responses                                                                                                                   |
| GenAI video generation                                | The actual start-operation response, including a pending operation; no invented completed video or automatic polling                                                 |
| Legacy GA/Preview model generation and streams        | Existing native APIs and response objects remain usable                                                                                                              |
| Legacy GA/Preview chat and chat streams               | Actual exposed chat history, generation result and native stream/response contract                                                                                   |

The Gemini Developer API backend is left untouched. Administrative operations,
model listing, caches, files, batches, tuning, token counting, operation polling,
Live WebSockets and the separate Interactions API are outside this adapter's
scope. This matrix describes tested inference surfaces, not blanket coverage
of every API in the 2.28.0 package.

Legacy setup remains available:

```ts
import * as vertex from "@google-cloud/vertexai";
import { VertexAIInstrumentor } from "@respan/instrumentation-vertexai";

const instrumentor = new VertexAIInstrumentor({ sdkModule: vertex });
await instrumentor.activate();
const client = new vertex.VertexAI({
  project: "your-project",
  location: "us-central1",
});
await client
  .getGenerativeModel({ model: "your-vertex-model" })
  .generateContent("Hello");
instrumentor.deactivate();
```

When modules are omitted, the instrumentor resolves installed compatible
modules. Explicit modules support ESM applications without replacing exported
constructors. The modern bridge observes tested native `Models` internal
request methods because public generation/embedding methods are instance
fields; this preserves native chat and automatic function calling. Future
incompatible internal API changes may require another adapter update.

`traceContent`, `recordInputs` and `recordOutputs` are optional constructor
settings. Constructor false, the Respan/Traceloop content environment flags,
the canonical Traceloop context or an observed ancestor veto cannot be
reversed by later true. Input and output lanes are independent. Real native
sampling precedes adapter payload copies; DROP and unsampled RECORD_ONLY
paths add no serialization beyond the native SDK. Actual span attributes,
events and status are guarded. Native GenAI preprocessing may still transform
request parameters before its internal request hook, and the SDK must still
serialize its HTTP request.

Set plugin and runtime content settings consistently. For a whole workflow,
`RESPAN_TRACE_CONTENT=false` also disables the released runtime's decorator
capture, which is separate from explicit plugin options.

No tool execution is inferred from historical messages or a model function
call. Decorate an actual application callback with `respan.withTool` to trace
its execution; the automatic-function-calling example demonstrates this with
an actual returned native invocation ID. Missing provider totals remain
missing. Provider-reported totals, cache counts and reasoning counts remain
distinct. Canonical complete bodies, schemas, output and usage precede
optional indexed attributes under the default OTel attribute budget.

Owners share only the actual prototypes they activate. Deactivation restores
owned methods without overwriting foreign replacements; in-flight calls drain.
Consume streams or return their iterators to close their spans. Public helper
exports from earlier versions remain available.

Native SDK tests and eleven mapped examples use controlled HTTP/SSE transports
and local collectors. They validate released SDK parsing, iteration, options
and tracing without Google cloud credentials. Live model access is a separate,
explicit example mode and still depends on Google credentials and availability.
