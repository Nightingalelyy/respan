# @respan/instrumentation-braintrust

Translate real Braintrust TypeScript logger and span activity into the active
Respan OpenTelemetry pipeline. Supports Braintrust 3.9.0 and later; validated
with 3.9.0 and 3.37.2 on Node.js 24.

```ts
import { Respan } from "@respan/respan";
import { BraintrustInstrumentor } from "@respan/instrumentation-braintrust";
import { initLogger } from "braintrust";

const instrumentor = new BraintrustInstrumentor();
const respan = new Respan({ instrumentations: [instrumentor] });
await respan.initialize();

const logger = initLogger({
  projectName: "my-project",
  apiKey: process.env.BRAINTRUST_API_KEY,
});
await logger.traced(
  async (span) => {
    span.log({ input: { city: "Kyoto" } });
    const result = { city: "Kyoto", temperature: 21 };
    span.log({ output: result });
    return result;
  },
  { name: "lookup-weather", type: "tool" },
);

await logger.flush();
await instrumentor.flush();
await respan.flush();
await instrumentor.deactivate();
await respan.shutdown();
```

The bridge replaces the global Braintrust background logger while active, so
native rows are sent through Respan instead of the Braintrust log destination.
Braintrust can still contact its configured service for login and project
metadata. Configure Braintrust normally for an application; the mapped examples
use a loopback service and fixture keys, with no hosted Braintrust requests.

`logger.log`, `logger.startSpan`, `logger.traced`, child spans, `span.log`,
`span.setAttributes`, `span.end`, and Braintrust provider wrappers retain their
native return values, errors, promises, stream objects and controllers. Updates
merge across flushes. A span is exported only when native metrics include its
end time. Flush does not invent completion for an unfinished native span.

The real OTel sampler runs before the bridge captures payloads. `traceContent:
false`, either `RESPAN_TRACE_CONTENT=false` or `TRACELOOP_TRACE_CONTENT=false`,
Traceloop context denial, tracing suppression at admission, and observed parent
content denial prevent payload capture. Denial stays in effect for that span.
DROP and RECORD_ONLY skip bridge-owned lazy row resolution. Native Braintrust
may still copy application data as part of its own unmodified logging behavior.
Late attributes, events and status text on the actual recording span are guarded.

```ts
const instrumentor = new BraintrustInstrumentor({ traceContent: false });
console.log(instrumentor.isActive()); // method, available after activation
```

Canonical JSON preserves full native inputs, outputs, multimodal message parts,
tool definitions and tool-call IDs, vectors, zero/false/empty/null values,
metadata, scores, tags and explicit usage. Native IDs remain in `respan.metadata`
for correlation; OTel generates its own IDs and retains native parent links.
Provider/model names and total tokens are never inferred. Workflow, agent and
task spans carry common fields; tool execution input uses `{name, arguments}`.
Optional indexed message attributes follow complete canonical fields so default
OTel attribute limits do not remove full input, output, tools or usage.

Braintrust's OpenAI wrapper currently logs embedding length rather than vector
values. The bridge preserves that native output; full vectors are captured when
supplied through the native span API. Wrapper model metadata may arrive after
native span construction and therefore after sampling. Initial native metadata
is available to the sampler when supplied in `startSpan`/`traced` arguments.
Feedback and remote update rows without a local native span lifecycle do not
create new spans. Multiple instrumentor owners share one bridge, final
deactivation drains in-flight spans, and foreign logger/prototype overrides
are preserved.

`includeBraintrustRawMetadata: true` additionally retains the original native
`span_attributes` inside canonical metadata. The existing `exportRecord` and
`exportRecords` APIs remain available for explicit imported rows; normal native
applications should use Braintrust's logger/span APIs.

Braintrust's official `@braintrust/otel` integration sends OTel spans to
Braintrust and shares context. This package handles the opposite producer
path: native Braintrust spans into Respan.
