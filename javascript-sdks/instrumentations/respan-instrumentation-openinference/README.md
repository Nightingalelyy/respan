# Respan OpenInference instrumentation

Route spans from released OpenInference helpers and instrumentors through the
Respan tracing runtime. The adapter preserves the producer's spans and parent
links, adds canonical input/output, messages, tools, model and usage fields, and
removes redundant OpenInference attributes from an export clone.

```bash
npm install @respan/respan @respan/instrumentation-openinference @arizeai/openinference-core
```

```typescript
import { Respan } from "@respan/respan";
import { OpenInferenceInstrumentor } from "@respan/instrumentation-openinference";
import { traceAgent, traceTool } from "@arizeai/openinference-core";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new OpenInferenceInstrumentor()],
});
await respan.initialize();

const lookup = traceTool((query: string) => ({ query, found: true }), {
  name: "lookup",
});
const agent = traceAgent((query: string) => lookup(query), { name: "support" });
await respan.withWorkflow({ name: "support-request" }, async () =>
  agent("tracing"),
);
await respan.shutdown();
```

For a released upstream instrumentor, retain the existing constructor interface.
For example, install `openai` and
`@arizeai/openinference-instrumentation-openai`, then use:

```typescript
import * as OpenAI from "openai";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import { OpenInferenceInstrumentor } from "@respan/instrumentation-openinference";

const plugin = new OpenInferenceInstrumentor(OpenAIInstrumentation, OpenAI);
```

Initialize Respan before activating a plugin yourself. `activate()` is
idempotent, `isActive()` reports its state, and the final owner disables the
upstream instrumentor. Instrumentors that return a promise from activation
must be awaited. OpenInference translation for spans already started drains
through the released Respan transformer registry after deactivation.

`OpenInferenceTranslator` remains available as a standalone OpenTelemetry
SpanProcessor. `translateOpenInferenceSpan()` and
`prepareOpenInferenceSpanForExport()` retain their existing signatures.

## Captured fields

LLM spans include request and response models when supplied, input and output
messages, multimodal blocks, historical and current tool calls separately,
complete tool definitions, invocation parameters and actual usage. Embedding
spans include the model, text and complete vectors. Tool execution spans carry
`{name, arguments}` input and the actual tool-call ID when the producer supplies
it. Retriever spans include returned documents. Other kinds retain common
input/output and metadata without being relabeled as LLM calls.

The adapter captures a released `OITracer` setter batch after the native sampler
and before OpenTelemetry's attribute limit. This retains complete canonical
aggregates for large message/tool sets without raising the native limit.
Attributes already discarded by a producer before this boundary cannot be
recovered. Original vendor attributes present on the native span remain
available to producer callbacks; cleanup applies to the export clone.

## Content controls

```typescript
new OpenInferenceInstrumentor({ traceContent: false });
new OpenInferenceInstrumentor({ recordInputs: false });
new OpenInferenceInstrumentor({ recordOutputs: false });
new OpenInferenceInstrumentor(OpenAIInstrumentation, OpenAI, {
  traceContent: false,
});
```

Constructor options, `RESPAN_TRACE_CONTENT=false`,
`TRACELOOP_TRACE_CONTENT=false`, the canonical Traceloop context flag, tracing
suppression at span start, unsampled recording spans and an observed ancestor
`allow_trace_content=false` form capture ceilings. Later `true` values cannot
restore capture. Denied spans retain safe identity and numeric usage/status;
late attributes, events and error text are guarded on the actual recording
span. Input and output controls remain independent, including embedding text and vectors.
If another processor locks a writable native attribute property, the adapter
installs its guard in that property. An immutable attribute bag is scrubbed where
possible and fails closed at export. Data already copied by an earlier processor
cannot be recalled.

Released upstream helpers perform their own input serialization and attribute
construction before calling the native sampler. The adapter cannot prevent
that producer-owned work. OpenInference's own `OITracer` redaction settings
continue to apply before its native setter calls. Exporter ambient suppression
does not revoke content already admitted by the bridge.

## Compatibility

The helper producer baseline is `@arizeai/openinference-core` 2.4.0; the current
suite also exercises 2.8.0. Default helper processing omits a null return value; the examples use the released
`getOutputAttributes()` structured-output API to explicitly record JSON null.
Request/response-model distinctions are available
when the producer emits those newer attributes. The mature adapter suite uses
`@arizeai/openinference-instrumentation-openai` 4.4.1 and OpenAI 7.32.0 with
controlled native HTTP/SSE transport.

Production floors are OpenTelemetry API 1.9.0, core/trace SDK 2.10.0,
semantic-conventions 1.43.0, OpenInference semantic conventions 2.14.0,
Traceloop semantic conventions 0.27.0, Respan SDK 1.3.4 and tracing 1.6.1.

The Arize adapter uses its separately released OpenInference dependency. Its
consumers receive these changes when a compatible OpenInference release is
published; installing this source tree does not update an independently
installed Arize package's released companion.
