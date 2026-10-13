# Respan Helicone instrumentation

Trace calls made through Helicone's Python manual logger with Respan. This
package instruments the published `helicone-helpers` `HeliconeManualLogger`
surface without replacing or bypassing Helicone's own logging behavior.

The instrumentation covers:

- `HeliconeManualLogger.log_request()` and direct `send_log()` calls
- `HeliconeLogBuilder`, including streamed chunks and error logs
- OpenAI-, Anthropic-, and custom-model request/response payloads
- Helicone `_type=tool`, `_type=vector_db`, and `_type=data` custom logs
- sync manual logging and the builder's async `send_log()` lifecycle

## Install

```bash
pip install respan-ai respan-instrumentation-helicone "helicone-helpers~=1.2.1"
```

## Usage

```python
import os

from helicone_helpers import HeliconeManualLogger
from respan import Respan, workflow
from respan_instrumentation_helicone import HeliconeInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[HeliconeInstrumentor()],
)
helicone = HeliconeManualLogger(api_key=os.environ["HELICONE_API_KEY"])


@workflow(name="helicone_manual_log")
def run(prompt: str) -> str:
    request = {
        "model": "custom-chat-model",
        "messages": [{"role": "user", "content": prompt}],
    }

    def operation(recorder):
        response = {
            "model": "custom-chat-model",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        }
        recorder.append_results(response)
        return "Hello!"

    return helicone.log_request(request, operation, provider="openai")


try:
    print(run("Say hello."))
finally:
    respan.flush()
    respan.shutdown()
```

`HeliconeInstrumentor(capture_content=False)` keeps span identity, model,
provider, timing, usage, and error type/status while omitting request and response
content and exception text. The published `helicone-helpers` 1.2.1 is both the
tested current release and supported minimum.
Authentication and Helicone transport headers are never copied into spans.
Safe constructor/request correlation headers are normalized into canonical
customer/session fields and one JSON `respan.metadata` attribute.

Spans use the active OpenTelemetry tracer and sampler. Dropped calls and
record-only calls do not normalize or serialize request/response bodies.
Instrumentation suppression and a disabled Respan runtime prevent new spans.
`TRACELOOP_TRACE_CONTENT=false`, `RESPAN_TRACE_CONTENT=false`, the runtime's
`ENABLE_CONTENT_TRACING_KEY=False` context value, and an ancestor span's
`allow_trace_content=False` attribute each prevent content capture. A later
`True` context value cannot override a constructor, environment, or ancestor
veto.

The native logger, result recorder, builder, return values, exceptions, and
iterator results retain their identities. Privacy is checked at native request,
recorder append, builder mutation, and send boundaries. Once observed for a
pending log, a content veto survives context detachment. Request and response
JSON, tool definitions, and embedding vectors retain complete native content;
indexed convenience attributes remain subject to OpenTelemetry attribute
limits. Canonical bodies, identity, model, and usage are set after those indexed
attributes so they survive the default attribute budget.

Builders capture their creation context and privacy bounds, then start a span
when sent. Creating or abandoning a builder starts no span. A creation-time
veto and send-time veto both apply, including after the creation parent ends.
An admitted `log_request()` finishes once if the final owner deactivates inside
its callback; its remaining content is omitted. A builder from a deactivated
generation still sends its native log but emits no span under a later owner.
Builders created before instrumentation activation have no creation snapshot
and use their send context.

## Activation and overlap policy

Helicone manual logging is explicit instrumentation. Activate
`HeliconeInstrumentor()` only in applications that use `HeliconeManualLogger`;
it is not a direct-provider auto-instrumentor. The package patches Helicone's
logger and builder methods only—it does not patch OpenAI, Anthropic, or another
provider SDK.

Do not combine this package with instrumentation for the provider operation
that the same manual log describes unless you intentionally want both records.
For example, wrapping an OpenAI call in `HeliconeManualLogger.log_request()`
while also activating `OpenAIInstrumentor()` produces one provider span and one
Helicone manual-log span for the same logical model call.

The instrumentor is reference-counted and restores the exact SDK methods it
patched when the final owner deactivates. It also exposes `instrument()` and
`uninstrument()` aliases for compatibility with OpenTelemetry-style lifecycle
code.
Concurrent owners must use the same `capture_content` setting; a conflicting
owner is rejected without changing the active policy. Foreign patches installed
after activation are preserved at shutdown.

See the Helicone example set in `respan-example-projects/python/tracing/helicone`
for deterministic sync, async-builder, streaming, error, tool, vector database,
and custom data coverage.
