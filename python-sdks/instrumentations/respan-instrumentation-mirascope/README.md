# Respan instrumentation for Mirascope

This package traces Mirascope 2.x model calls, context-aware calls, sync and
async streams, and toolkit execution. Model calls become canonical Respan chat
spans; toolkit executions become canonical tool spans. Streaming spans finish
when the underlying stream is consumed, emit `llm.is_streaming=true`, and
preserve real iterator errors without treating normal iterator completion as a
failure.

```python
from respan_instrumentation_mirascope import MirascopeInstrumentor

instrumentor = MirascopeInstrumentor(capture_content=True)
instrumentor.activate()
```

`capture_content=False` omits messages, tool definitions, arguments, and model
outputs while retaining models, providers, usage, status, and errors. Activation
is reference-counted and safe to repeat.

Mirascope `ToolCall.args` JSON text is decoded for tool-span input as
`{"name": ..., "arguments": ...}`. Chat spans retain canonical request tool
definitions and current-turn calls, while tool outputs store the executed
function result rather than Mirascope's transport envelope.

Do not combine this adapter with `mirascope.ops.instrument_llm()` unless you
intentionally want two independent telemetry pipelines for each operation.
