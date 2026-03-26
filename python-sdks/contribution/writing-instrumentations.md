# Writing Respan Instrumentation Plugins — SOP

## Overview

This guide covers how to build a `respan-instrumentation-*` package that plugs into `Respan(instrumentations=[...])`. There are two paths depending on what OTEL support exists for the SDK you're instrumenting.

**Decision tree:**

```
Does the SDK have an existing OTEL instrumentor?
│
├── YES (e.g. opentelemetry-instrumentation-openai,
│        opentelemetry-instrumentation-anthropic,
│        openinference-instrumentation-*)
│   └── Path A: Thin OTEL Wrapper
│       Just activate the existing instrumentor.
│       Spans are already ReadableSpan — they flow through the pipeline natively.
│       Example: respan-instrumentation-openai
│
└── NO (the SDK has its own tracing/callback system,
│       e.g. OpenAI Agents SDK, Claude Agent SDK, Google ADK)
    └── Path B: Custom Emitter
        Register a hook with the SDK, convert its events to ReadableSpan,
        inject into the OTEL pipeline.
        Example: respan-instrumentation-openai-agents
```

---

## The Protocol

Every plugin implements this (defined in `respan/_types.py`):

```python
class Instrumentation(Protocol):
    name: str
    def activate(self) -> None: ...
    def deactivate(self) -> None: ...
```

| Method | When called | What to do |
|--------|-------------|------------|
| `activate()` | `Respan.__init__()` | Register hooks/processors with the vendor SDK |
| `deactivate()` | `Respan.shutdown()` | Clean up hooks/processors |

No exporter argument is passed. Spans enter the pipeline via one of:
- **OTEL instrumentor** → spans flow through TracerProvider automatically
- **`inject_span()`** → manually push a ReadableSpan into the processor chain

---

## Package Structure

All instrumentation packages follow the same layout:

```
respan-instrumentation-{name}/
├── pyproject.toml
├── README.md
├── src/
│   └── respan_instrumentation_{name}/
│       ├── __init__.py              # Exports the instrumentor class
│       ├── _instrumentation.py      # The instrumentor (activate/deactivate)
│       ├── _otel_emitter.py         # [Path B only] Converts SDK events → ReadableSpan
│       └── _utils.py                # [Path B only] Serialization/formatting helpers
└── tests/
    └── ...
```

### pyproject.toml template

```toml
[tool.poetry]
name = "respan-instrumentation-{name}"
version = "0.1.0"
description = "Respan instrumentation plugin for {SDK Name}."
packages = [{include = "respan_instrumentation_{name}", from = "src"}]

[tool.poetry.dependencies]
python = ">=3.11,<3.14"
respan-tracing = ">=2.3.0"
respan-sdk = ">=0.4.0"
# Path A: add the OTEL instrumentor
# opentelemetry-instrumentation-{name} = ">=0.x.0"
# Path B: add the vendor SDK
# {vendor-sdk} = ">=x.y.z"

[tool.poetry.plugins."respan.instrumentations"]
{name} = "respan_instrumentation_{name}:XxxInstrumentor"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### `__init__.py` template

```python
"""Respan instrumentation plugin for {SDK Name}."""

from ._instrumentation import XxxInstrumentor

__all__ = ["XxxInstrumentor"]
```

---

## Path A: Thin OTEL Wrapper

Use this when an OTEL instrumentor already exists. Your plugin just activates it.

### OTEL instrumentor sources

1. **Traceloop OpenLLMetry** — `opentelemetry-instrumentation-{name}`
   - https://github.com/traceloop/openllmetry
   - Covers: OpenAI, Anthropic, Cohere, Bedrock, etc.

2. **Arize OpenInference** — `openinference-instrumentation-{name}`
   - https://github.com/Arize-ai/openinference
   - Covers: OpenAI, Anthropic, LangChain, LlamaIndex, Google ADK, Claude Agent SDK, etc.

Both produce standard OTEL `ReadableSpan` objects that flow through the Respan pipeline natively.

**Prefer Traceloop instrumentors** when available — they use `traceloop.*` attributes that our pipeline already understands. OpenInference instrumentors use `openinference.*` attributes which may need enrichment in the exporter.

### `_instrumentation.py` template (Path A)

```python
"""Thin OTEL wrapper for {SDK Name}."""

import logging

logger = logging.getLogger(__name__)


class XxxInstrumentor:
    """Respan instrumentor for {SDK Name}.

    Activates OTEL auto-instrumentation so that all {SDK} calls
    are traced automatically.

    Usage::

        from respan import Respan
        from respan_instrumentation_{name} import XxxInstrumentor

        respan = Respan(instrumentations=[XxxInstrumentor()])
    """

    name = "{name}"

    def __init__(self) -> None:
        self._instrumented = False

    def activate(self) -> None:
        try:
            from opentelemetry.instrumentation.{name} import (
                XxxInstrumentor as OTELInstrumentor,
            )

            instrumentor = OTELInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            self._instrumented = True
            logger.info("{SDK Name} instrumentation activated")
        except ImportError as exc:
            logger.warning(
                "Failed to activate {SDK Name} instrumentation — "
                "missing dependency: %s",
                exc,
            )

    def deactivate(self) -> None:
        if self._instrumented:
            try:
                from opentelemetry.instrumentation.{name} import (
                    XxxInstrumentor as OTELInstrumentor,
                )
                OTELInstrumentor().uninstrument()
            except Exception:
                pass
            self._instrumented = False
        logger.info("{SDK Name} instrumentation deactivated")
```

That's it. The OTEL library handles span creation. The spans flow through:

```
OTEL instrumentor creates span → TracerProvider → RespanSpanProcessor →
is_processable_span() → RespanSpanExporter → /v2/traces
```

### When to add patches (like OpenAI)

Sometimes the upstream OTEL library has bugs. The OpenAI plugin adds a sync prompt-capture patch because the upstream library loses prompts in sync contexts. If you find similar issues:

1. Add the patch in `_instrumentation.py` as a private function
2. Call it after `instrumentor.instrument()` in `activate()`
3. Wrap it in try/except so the plugin still works if the patch fails
4. Document WHY the patch exists (not just what it does)

---

## Path B: Custom Emitter

Use this when the SDK has its own tracing/callback system that is NOT OTEL-based. You need to:

1. Hook into the SDK's callback/processor system
2. Convert SDK events to OTEL `ReadableSpan` objects
3. Inject them into the pipeline

### Key imports

```python
from respan_tracing.utils.span_factory import build_readable_span, inject_span
```

| Function | What it does |
|----------|-------------|
| `build_readable_span(name, trace_id=, span_id=, parent_id=, attributes=, ...)` | Creates a `ReadableSpan` with proper IDs and attributes. Auto-merges propagated attributes from `propagate_attributes()` context. |
| `inject_span(span)` | Pushes the span through the active TracerProvider's processor chain → exporter → `/v2/traces`. |

### `_instrumentation.py` template (Path B)

```python
"""{SDK Name} instrumentation plugin for Respan."""

import logging
from typing import Optional

# Import the SDK's callback/processor interface
from {sdk}.tracing import SomeTracingProcessor, SomeSpan, SomeTrace

from ._otel_emitter import emit_sdk_item

logger = logging.getLogger(__name__)


class _RespanProcessor(SomeTracingProcessor):
    """SDK processor that converts events to OTEL spans."""

    def on_trace_end(self, trace: SomeTrace) -> None:
        emit_sdk_item(trace)

    def on_span_end(self, span: SomeSpan) -> None:
        emit_sdk_item(span)

    # Implement other required methods as no-ops
    def on_trace_start(self, trace): pass
    def on_span_start(self, span): pass
    def shutdown(self): pass
    def force_flush(self): pass


class XxxInstrumentor:
    """Respan instrumentor for {SDK Name}."""

    name = "{name}"

    def __init__(self) -> None:
        self._processor: Optional[_RespanProcessor] = None

    def activate(self) -> None:
        from {sdk}.tracing import set_processors

        self._processor = _RespanProcessor()
        set_processors([self._processor])
        logger.info("{SDK Name} instrumentation activated")

    def deactivate(self) -> None:
        self._processor = None
        logger.info("{SDK Name} instrumentation deactivated")
```

### `_otel_emitter.py` template (Path B)

This is where the conversion happens. Each SDK span type gets its own emitter function.

```python
"""Convert {SDK Name} spans to OTEL ReadableSpan objects."""

import json
import logging
from typing import Any, Dict

from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)

# Attribute key constants
_SPAN_KIND = "traceloop.span.kind"
_ENTITY_NAME = "traceloop.entity.name"
_ENTITY_PATH = "traceloop.entity.path"
_ENTITY_INPUT = "traceloop.entity.input"
_ENTITY_OUTPUT = "traceloop.entity.output"
_WORKFLOW_NAME = "traceloop.workflow.name"
_LLM_REQUEST_TYPE = "llm.request.type"
_GEN_AI_SYSTEM = "gen_ai.system"
_GEN_AI_MODEL = "gen_ai.request.model"
_GEN_AI_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
_GEN_AI_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
_LOG_TYPE = "respan.entity.log_type"


def _base_attrs(span_kind: str, entity_name: str, entity_path: str, log_type: str) -> Dict[str, Any]:
    """Common attributes shared by all emitters."""
    return {
        _SPAN_KIND: span_kind,
        _ENTITY_NAME: entity_name,
        _ENTITY_PATH: entity_path,
        _LOG_TYPE: log_type,
    }


# ---------------------------------------------------------------------------
# Per-type emitters — add one for each SDK span type
# ---------------------------------------------------------------------------


def emit_trace(trace_obj) -> None:
    """Emit the root workflow span."""
    attrs = _base_attrs(
        span_kind="workflow",
        entity_name=trace_obj.name or "trace",
        entity_path="",  # root — empty path
        log_type="workflow",
    )
    attrs[_WORKFLOW_NAME] = trace_obj.name or "trace"

    span = build_readable_span(
        name=f"{trace_obj.name}.workflow",
        trace_id=trace_obj.trace_id,
        span_id=trace_obj.trace_id,  # root uses trace_id as span_id
        attributes=attrs,
    )
    inject_span(span)


def emit_llm_call(item) -> None:
    """Emit an LLM call span."""
    attrs = _base_attrs(
        span_kind="task",
        entity_name="response",
        entity_path="response",  # non-empty to prevent root promotion
        log_type="generation",
    )
    attrs[_LLM_REQUEST_TYPE] = "chat"
    attrs[_GEN_AI_SYSTEM] = "openai"  # or "anthropic", etc.
    attrs[_GEN_AI_MODEL] = item.model or ""
    attrs[_GEN_AI_PROMPT_TOKENS] = item.usage.input_tokens or 0
    attrs[_GEN_AI_COMPLETION_TOKENS] = item.usage.output_tokens or 0
    attrs[_ENTITY_INPUT] = json.dumps(item.messages, default=str)
    attrs[_ENTITY_OUTPUT] = json.dumps(item.output, default=str)

    span = build_readable_span(
        name="openai.chat",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


def emit_tool_call(item) -> None:
    """Emit a tool/function call span."""
    attrs = _base_attrs(
        span_kind="tool",
        entity_name=item.name,
        entity_path=item.name,
        log_type="tool",
    )
    attrs[_ENTITY_INPUT] = json.dumps(item.input, default=str)
    attrs[_ENTITY_OUTPUT] = json.dumps(item.output, default=str)

    span = build_readable_span(
        name=f"{item.name}.tool",
        trace_id=item.trace_id,
        span_id=item.span_id,
        parent_id=item.parent_id or item.trace_id,
        attributes=attrs,
        status_code=400 if item.error else 200,
        error_message=str(item.error) if item.error else None,
    )
    inject_span(span)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def emit_sdk_item(item) -> None:
    """Route an SDK item to the correct emitter."""
    if isinstance(item, SomeTrace):
        emit_trace(item)
        return

    # Dispatch by span data type
    span_data = item.span_data
    if isinstance(span_data, LLMSpanData):
        emit_llm_call(item)
    elif isinstance(span_data, ToolSpanData):
        emit_tool_call(item)
    # ... add more types as needed
    else:
        logger.warning("Unknown span type: %s", type(span_data).__name__)
```

---

## Required Attributes Reference

For spans to be processed and displayed correctly by the Respan backend:

### Must-have (all spans)

| Attribute | Values | Why |
|-----------|--------|-----|
| `traceloop.span.kind` | `"workflow"`, `"agent"`, `"task"`, `"tool"` | Identifies span type. Required by `is_processable_span()`. |
| `traceloop.entity.name` | Any string | Display name in the dashboard. |
| `respan.entity.log_type` | `"workflow"`, `"agent"`, `"generation"`, `"tool"`, `"handoff"`, `"guardrail"`, `"custom"` | Backend uses this for span categorization. |

### Must-have (LLM call spans)

| Attribute | Values | Why |
|-----------|--------|-----|
| `llm.request.type` | `"chat"` | Triggers prompt/completion/model/token parsing on the backend. |
| `gen_ai.request.model` | Model name string | Displayed in dashboard. |
| `traceloop.entity.input` | JSON string of messages | The input messages. Format: `[{"role": "user", "content": "..."}]` |
| `traceloop.entity.output` | Plain text string | The output text content. The backend wraps this in `{"role": "assistant", "content": "..."}` automatically — do **not** pre-wrap it or it will be double-wrapped. |

### Should-have (LLM call spans)

| Attribute | Values | Why |
|-----------|--------|-----|
| `gen_ai.system` | `"openai"`, `"anthropic"`, etc. | Provider identification. |
| `gen_ai.usage.prompt_tokens` | int | Token usage tracking. |
| `gen_ai.usage.completion_tokens` | int | Token usage tracking. |

### Critical: `traceloop.entity.path`

| Span type | `entity_path` value | Why |
|-----------|-------------------|-----|
| Root (workflow/trace) | `""` (empty string) | Marks it as the root. |
| All children | Non-empty (e.g. `"response"`, `"search"`) | **Prevents accidental root promotion** by `is_root_span_candidate()`. If a child span has an empty path, it may be promoted to root and break the trace hierarchy. |

---

## ID Handling

`build_readable_span()` accepts string IDs. Use the IDs from the vendor SDK directly:

```python
span = build_readable_span(
    name="openai.chat",
    trace_id=sdk_item.trace_id,       # string — auto-converted to 128-bit int
    span_id=sdk_item.span_id,         # string — auto-converted to 64-bit int
    parent_id=sdk_item.parent_id,     # string or None (None = root span)
)
```

- Hex strings are parsed directly
- Non-hex strings (UUIDs with hyphens, arbitrary IDs) are hashed via MD5 to produce a stable numeric ID
- `None` trace_id/span_id = auto-generated

**Important:** Preserve the SDK's original IDs. This maintains the parent-child hierarchy from the original trace.

---

## Propagated Attributes

`build_readable_span()` automatically merges attributes from `propagate_attributes()` context (customer_identifier, thread_identifier, metadata, etc.). You don't need to handle this — it's built in.

If a user writes:
```python
with propagate_attributes(customer_identifier="user_123"):
    result = await Runner.run(agent, "Hello")
```

All spans created by your plugin within that scope will automatically carry `respan.customer_params.customer_identifier = "user_123"`.

---

## Using OpenInference Instrumentors

For SDKs where no Traceloop OTEL instrumentor exists, check if [OpenInference](https://github.com/Arize-ai/openinference) has one. OpenInference instrumentors produce standard OTEL spans — they work with Path A.

```python
# Example: using OpenInference for Google ADK
class GoogleADKInstrumentor:
    name = "google-adk"

    def activate(self) -> None:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor as OIInstrumentor
        self._instrumentor = OIInstrumentor()
        self._instrumentor.instrument()

    def deactivate(self) -> None:
        self._instrumentor.uninstrument()
```

**Caveats with OpenInference:**
- Uses `openinference.*` semantic attributes instead of `traceloop.*`
- Spans will still pass `is_processable_span()` if they have `gen_ai.system` or `llm.request.type`
- If the backend doesn't parse prompts/completions correctly, you may need to add attribute mapping in the emitter or enrichment in `EnrichedSpan`
- Test thoroughly — attribute naming differences can cause empty fields on the dashboard

**Available OpenInference instrumentors** (that have no Traceloop equivalent):
- `openinference-instrumentation-google-adk` — Google ADK
- `openinference-instrumentation-claude-agent-sdk` — Claude Agent SDK
- `openinference-instrumentation-pydantic-ai` — PydanticAI
- `openinference-instrumentation-smolagents` — smolagents
- `openinference-instrumentation-autogen-agentchat` — Microsoft Autogen
- `openinference-instrumentation-crewai` — CrewAI
- `openinference-instrumentation-instructor` — Instructor

---

## Testing Checklist

Before submitting a new instrumentation package:

- [ ] `activate()` succeeds when the vendor SDK is installed
- [ ] `activate()` logs a warning (not error) when the vendor SDK is missing
- [ ] `deactivate()` cleans up without errors
- [ ] Spans appear on the Respan dashboard with correct:
  - [ ] Trace hierarchy (parent-child relationships intact)
  - [ ] Span names and types
  - [ ] Input/output messages (for LLM calls)
  - [ ] Model name and token usage (for LLM calls)
- [ ] `propagate_attributes()` works — customer_identifier etc. appear on all spans
- [ ] No duplicate spans (if both auto-instrumentation and plugin are active, one should be disabled)
- [ ] Error spans show error status and message
