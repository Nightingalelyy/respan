# @respan/instrumentation-pydantic-ai

Respan instrumentation for Pydantic AI-compatible TypeScript OpenTelemetry spans.

The repository does not currently document a stable TypeScript Pydantic AI SDK API
to patch. This package therefore installs a conservative span processor instead:
it normalizes Pydantic AI native OTel spans and Pydantic AI-scoped
OpenInference spans into the canonical Respan span contract before export.

## Installation

```bash
npm install @respan/respan @respan/instrumentation-pydantic-ai
```

## Usage

```ts
import { Respan } from "@respan/respan";
import { PydanticAIInstrumentor } from "@respan/instrumentation-pydantic-ai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  appName: "pydantic-ai-typescript",
  instrumentations: [new PydanticAIInstrumentor()],
});

await respan.initialize();
```

The instrumentor expects spans to already be emitted by a Pydantic AI-compatible
runtime or OpenInference bridge. It does not invent or monkey-patch unknown SDK
entrypoints.

## Mapped Span Shapes

- Native Pydantic AI-style GenAI spans such as `gen_ai.operation.name`,
  `gen_ai.input.messages`, `gen_ai.output.messages`,
  `model_request_parameters`, `gen_ai.tool.definitions`, and tool call attrs.
- Pydantic AI-scoped OpenInference spans with `openinference.span.kind` and
  indexed `llm.input_messages.*` / `llm.output_messages.*` attrs.

The package emits canonical Respan fields including:

- `respan.entity.log_type`
- `traceloop.entity.name`
- `traceloop.entity.path`
- `traceloop.entity.input`
- `traceloop.entity.output`
- `llm.request.type`
- `llm.request.functions`
- `gen_ai.prompt.*`
- `gen_ai.completion.*`
- `gen_ai.usage.*`

Raw Pydantic AI/OpenInference attrs and off-contract aliases are stripped before
delegating to the active Respan span processor.
