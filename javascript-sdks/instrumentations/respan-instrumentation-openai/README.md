# OpenAI instrumentation

`@respan/instrumentation-openai` instruments the OpenAI JavaScript SDK and sends
spans through the active OpenTelemetry tracer provider. It works with the Respan
plugin protocol and does not configure a provider or exporter itself.

```sh
npm install @respan/respan @respan/instrumentation-openai openai
```

```ts
import OpenAI from "openai";
import { Respan } from "@respan/respan";
import { OpenAIInstrumentor } from "@respan/instrumentation-openai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new OpenAIInstrumentor()],
});
await respan.initialize();

const client = new OpenAI();
await client.responses.create({ model: "gpt-4o-mini", input: "Say hello" });
await respan.flush();
```

## Compatibility

The supported SDK range is `openai >=4.0.0 <8`. The package test suite exercises
real SDK releases **4.0.0, 4.104.0, 5.23.2, 6.49.0, 7.0.0, and 7.23.0** against
deterministic HTTP/SSE responses. SDK 7.23.0 was the latest npm release checked
on September 27, 2026. Use a Node.js version supported by the SDK you install.

| API                                             | Coverage                                                                                                 |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `chat.completions.create()`                     | Messages, function tools, usage, non-streaming and streaming                                             |
| `completions.create()`                          | Text input/output, usage, non-streaming and streaming                                                    |
| `responses.create()`                            | Instructions, message/function-call input, text/function-call output, usage, non-streaming and streaming |
| `embeddings.create()`                           | Input, returned vectors, and provider token usage                                                        |
| Chat/Responses `parse()` and `stream()` helpers | Delegated requests are traced once; helpers retain native return values                                  |

Responses and helper methods are available only when the installed SDK provides
them; they do not exist in every 4.x release. Function-tool definitions and calls
are captured, but this plugin does not create spans for your tool implementation.
Use `respan.withTool()` around tool execution when you need those spans.

Azure OpenAI uses its separate instrumentation package and is skipped here.
Images, audio, Realtime/WebSocket, Assistants, batch job execution, and Responses
retrieval/background polling are outside this package's tracing coverage.

The adapter now patches SDK resources directly. `@traceloop/instrumentation-openai`
is no longer required. Avoid also instrumenting the same client with a separate
OpenAI adapter, which would create duplicate spans.

## SDK behavior and lifecycle

The default activation resolves the application's installed SDK and patches both
its CommonJS and ESM resource classes. For an application with multiple SDK copies,
pass the actual constructor or imported module:

```ts
const instrumentor = new OpenAIInstrumentor({ openAI: OpenAI });
await instrumentor.activate();
// Make SDK calls after the tracer provider and instrumentor are active.
instrumentor.deactivate();
```

Activation is idempotent and shared between active plugin instances. The last
deactivation restores the patched methods. Requests already in progress finish
their existing spans.

`APIPromise`, `.withResponse()`, and SDK stream objects retain their methods.
Calling `.asResponse()` alone records request metadata and ends the span without
reading the response body. Stream spans finish on exhaustion, early iterator
return, or explicit controller abort, retaining the output observed so far.
Consume or abort streams before flushing. Streaming usage is recorded only when
the provider sends usage; Chat Completions may require
`stream_options: { include_usage: true }`.

## Content controls

Set `traceContent: false` on the instrumentor or `TRACELOOP_TRACE_CONTENT=false`
to omit prompts, completions, tool payloads, and embedding vectors. An active
`CONTEXT_KEY_ALLOW_TRACE_CONTENT` value takes precedence over the option and
environment default. OpenTelemetry tracing suppression is respected. Model and
provider token usage remain available when content is disabled.

## Development

From the JavaScript workspace, install dependencies with Yarn, then run:

```sh
cd instrumentations/respan-instrumentation-openai
npm test
npm pack --dry-run
```

The tests cover the SDK matrix, native promise/stream helpers, raw access, tool
payloads, lifecycle, Azure exclusion, context propagation, content controls,
stream cancellation, and transport/parse failures. Paired examples live in
[`respan-example-projects/typescript/tracing/openai-sdk`](https://github.com/respanai/respan-example-projects/tree/main/typescript/tracing/openai-sdk).
