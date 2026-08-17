# Respan OpenLIT instrumentation

`respan-instrumentation-openlit` sends OpenLIT's native OpenTelemetry spans
through the active Respan tracer provider and normalizes them to the Respan span
contract. It does not wrap provider calls a second time, so enabling this plugin
does not create a duplicate Respan span around each OpenLIT span.

## Install

```bash
pip install respan-ai respan-instrumentation-openlit
```

## Use

```python
from respan import Respan
from respan_instrumentation_openlit import OpenLITInstrumentor

respan = Respan(
    api_key="...",
    instrumentations=[OpenLITInstrumentor(capture_content=True)],
)
```

Activation calls `openlit.init()` with the existing OpenTelemetry provider,
metrics and events disabled, and an offline empty pricing file. OpenLIT therefore
owns provider/framework patching while Respan owns export. Pass a custom
`pricing_json` if OpenLIT cost calculation is required.

`capture_content=False` disables OpenLIT message capture, strips content-bearing
attributes and events, and replaces error status descriptions with a generic
message before export. OpenLIT events are removed in both modes so provider or
retry integrations cannot bypass the bounded canonical attributes.
`max_content_length` bounds redacted OpenAI request content by UTF-8 bytes.
`disabled_instrumentors` is forwarded to OpenLIT and can exclude selected
libraries.

Generic HTTP client instrumentors are disabled by default so a provider call
does not also export nested transport noise. Set `capture_transport_spans=True`
to opt in. Calls made while multiple adapter instances are active share one
OpenLIT activation; every instance must use the same configuration and be
deactivated before the owned patches are removed.

For current OpenAI Chat Completions and Responses calls, the adapter retains
canonical `gen_ai.provider.name` / `gen_ai.system`, maps request tools to
`llm.request.functions`, maps current-turn calls to indexed
`gen_ai.completion.0.tool_calls`, preserves provider HTTP errors, and only
retains token usage when the OpenAI response actually supplied it. OpenLIT
spans from other providers retain their native usage because OpenLIT does not
expose a cross-provider provenance hook that distinguishes provider counts from
upstream estimates.

For OpenAI sync and async embedding calls, the adapter enriches OpenLIT native
spans with every returned vector element in canonical `traceloop.entity.output`.
The corresponding `traceloop.entity.input` contains the raw embedded value(s),
not a chat-message envelope. The enrichment does not create another span and is
disabled together with all other payload capture by `capture_content=False`.

Do not also enable a Respan provider instrumentation for the same client in the
same process unless two nested provider spans are intentional. Deactivation only
removes OpenLIT wrappers that this plugin activated; pre-existing and later
foreign wrappers/processors are preserved.
