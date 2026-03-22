# Respan Python SDK — Architecture Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Package Hierarchy](#2-package-hierarchy)
3. [How It All Connects](#3-how-it-all-connects)
4. [The OTEL Pipeline (Data Flow)](#4-the-otel-pipeline-data-flow)
5. [File-by-File Reference](#5-file-by-file-reference)
6. [Architecture Rating & Issues](#6-architecture-rating--issues)
7. [Target Architecture](#7-target-architecture)
8. [Migration Plan](#8-migration-plan)

---

## 1. Overview

Respan is an observability SDK for AI/LLM applications. It records what your AI code does — which models were called, what inputs/outputs, how long things took — and sends that data to Respan's servers.

It's built on **OpenTelemetry (OTEL)**, the industry standard for application tracing.

**Core concepts:**

| Term | What it is | Example |
|------|-----------|---------|
| **Span** | One recorded operation | "called GPT-4", "ran tool X" |
| **Trace** | A group of related spans | "the entire chatbot request" |
| **Processor** | A layer that filters/enriches spans before export | "only export LLM spans, not HTTP noise" |
| **Exporter** | Ships spans to the Respan API | Converts OTEL spans → OTLP JSON → POST /v2/traces |
| **Instrumentor** | Hooks into a library to auto-create spans | Monkey-patches `openai.chat.completions.create()` |

---

## 2. Package Hierarchy

Three core packages, layered like a cake:

```
┌─────────────────────────────────────────────────┐
│  respan-ai  (pip install respan-ai)             │
│  "The entry point"                               │
│                                                  │
│  - Respan() class                               │
│  - Plugin activation (instrumentations=[...])    │
│  - Re-exports decorators, client, etc.           │
│  - log_batch_results() helper                    │
│                                                  │
│  depends on: respan-tracing                      │
├─────────────────────────────────────────────────┤
│  respan-tracing  (pip install respan-tracing)   │
│  "The engine"                                    │
│                                                  │
│  - OTEL TracerProvider setup                     │
│  - Processor chain (filter, enrich, buffer)      │
│  - RespanSpanExporter (→ /v2/traces)             │
│  - @workflow, @task, @agent, @tool decorators    │
│  - RespanClient (trace/span API)                 │
│  - span_factory (build_readable_span, inject)    │
│  - 32 built-in auto-instrumentations (*)         │
│                                                  │
│  depends on: respan-sdk                          │
├─────────────────────────────────────────────────┤
│  respan-sdk  (pip install respan-sdk)            │
│  "The types"                                     │
│                                                  │
│  - RespanParams, FilterParamDict                 │
│  - OTLP constants                                │
│  - RespanSpanAttributes enum                     │
│  - Validation utilities                          │
│                                                  │
│  depends on: pydantic, requests                  │
└─────────────────────────────────────────────────┘

(*) Deprecated — being extracted to separate packages. See "Migration Plan".
```

### Instrumentation Packages (new pattern)

These implement the `Instrumentation` protocol and plug into `Respan(instrumentations=[...])`:

| Package | What it instruments | Pattern |
|---------|-------------------|---------|
| `respan-instrumentation-anthropic-agents` | Claude Agent SDK | Monkey-patches `claude_agent_sdk.query` |
| `respan-instrumentation-openai-agents` | OpenAI Agents SDK | Registers tracing processor |
| `respan-instrumentation-google-adk` | Google ADK | Intercepts OTEL span processors |
| `respan-instrumentation-openai` | Direct OpenAI SDK | Thin OTEL wrapper |
| `respan-instrumentation-langfuse` | Langfuse | Integration bridge |

Each plugin:
1. Implements `activate()` / `deactivate()`
2. Converts SDK events → `ReadableSpan` via `build_readable_span()`
3. Injects spans into the OTEL pipeline via `inject_span()`
4. Everything exits through the single `/v2/traces` endpoint

### Legacy Exporter Packages (old pattern — to be deprecated)

These predate the plugin architecture. They use dict-based export to `/v1/traces/ingest`:

| Package | Status | Superseded by |
|---------|--------|---------------|
| `respan-exporter-anthropic-agents` | Legacy | `respan-instrumentation-anthropic-agents` |
| `respan-exporter-openai-agents` | Legacy | `respan-instrumentation-openai-agents` |
| `respan-exporter-litellm` | Standalone (no respan deps) | Needs `respan-instrumentation-litellm` |
| `respan-exporter-agno` | Standalone (has OTEL deps) | Needs `respan-instrumentation-agno` |
| `respan-exporter-haystack` | Standalone (no respan deps) | Needs `respan-instrumentation-haystack` |
| `respan-exporter-braintrust` | Standalone | Needs `respan-instrumentation-braintrust` |
| `respan-exporter-superagent` | Standalone | Needs `respan-instrumentation-superagent` |

---

## 3. How It All Connects

### Initialization Flow

When a user writes:
```python
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

respan = Respan(
    api_key="xxx",
    instrumentations=[OpenAIAgentsInstrumentor()]
)
```

This is what happens:

```
Respan.__init__()                              # respan/_core.py
│
├── 1. RespanTelemetry(auto_instrument=False)  # respan_tracing/main.py
│      └── RespanTracer (singleton)            # respan_tracing/core/tracer.py
│            ├── Creates TracerProvider
│            ├── Adds RespanSpanExporter → /v2/traces
│            └── Skips auto-instrumentation (auto_instrument=False)
│
├── 2. Seeds _PROPAGATED_ATTRIBUTES ContextVar # respan_tracing/utils/span_factory.py
│      (customer_identifier, thread_id, etc.)
│
└── 3. Calls inst.activate() for each plugin   # Plugin's own code
       └── Plugin hooks into its SDK
           (monkey-patch, register processor, etc.)
```

### The Instrumentation Protocol

Every plugin implements this (defined in `respan/_types.py`):

```python
class Instrumentation(Protocol):
    name: str
    def activate(self) -> None: ...    # No exporter arg — uses inject_span()
    def deactivate(self) -> None: ...
```

---

## 4. The OTEL Pipeline (Data Flow)

There is **one single pipeline**. All spans — whether from decorators, auto-instrumentation, or plugins — flow through it.

```
                    THREE ENTRY POINTS
                    ==================

  @workflow/@task        Auto-instrumented         Plugin instrumentors
  decorators             LLM calls (OpenAI,        (Anthropic Agents,
                         Anthropic, etc.)            OpenAI Agents, ADK)
       │                      │                          │
       │                      │                          │
       ▼                      ▼                          ▼
  tracer.start_span()    OTEL instrumentor         build_readable_span()
  + span.end()           creates span               + inject_span()
       │                 automatically                    │
       │                      │                          │
       └──────────┬───────────┘                          │
                  │                                      │
                  ▼                                      │
     ┌─────────────────────────┐                         │
     │  TracerProvider         │◄────────────────────────┘
     │  _active_span_processor │
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │  BufferingSpanProcessor │  → If SpanBuffer is active, save to buffer
     │                         │  → Otherwise, pass through ↓
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │  FilteringSpanProcessor │  → If filter_fn exists, check it
     │                         │  → No filter = all spans pass
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │  RespanSpanProcessor    │  → on_start(): enriches with workflow name,
     │                         │    entity path, propagated attributes
     │                         │  → on_end(): is_processable_span()?
     │                         │    YES → forward to exporter
     │                         │    NO  → drop (auto-instrumentation noise)
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │  BatchSpanProcessor     │  → OTEL built-in, queues spans
     │  (OTEL built-in)        │  → Flushes batch on background thread
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │  RespanSpanExporter     │  → is_root_span_candidate()? wrap in ModifiedSpan
     │                         │  → Convert ReadableSpan → OTLP JSON
     │                         │  → POST /v2/traces
     │                         │  → Anti-recursion: _SUPPRESS_INSTRUMENTATION_KEY
     └─────────────────────────┘
```

### What `is_processable_span()` accepts:

| Has this attribute? | Meaning | Processed? |
|--------------------|---------|-----------:|
| `traceloop.span.kind` | User-decorated span (`@workflow`, `@task`, etc.) | YES |
| `traceloop.entity.path` | Child span within an entity context | YES |
| `llm.request.type` | Auto-instrumented LLM call | YES |
| Google ADK scope | From ADK instrumentation | YES |
| None of the above | HTTP noise, DB calls, etc. | NO — dropped |

### What `is_root_span_candidate()` promotes:

Standalone `@workflow`/`@task` spans or standalone LLM calls that have no parent entity get their parent cleared, making them root spans in the trace.

---

## 5. File-by-File Reference

### respan-tracing/src/respan_tracing/

| File | What it does | Key exports |
|------|-------------|-------------|
| **`__init__.py`** | Package front door. Re-exports everything users need. | `RespanTelemetry`, `get_client`, `workflow`, `task`, `agent`, `tool` |
| **`main.py`** | Entry point. Creates `RespanTracer`, sets up logging. | `RespanTelemetry` class, `get_client()` |
| **`instruments.py`** | Enum of 32 supported auto-instrumentation libraries. *(deprecated — to be removed)* | `Instruments` enum |
| **`core/tracer.py`** | Singleton OTEL engine. Sets up TracerProvider, manages processors. | `RespanTracer` |
| **`core/client.py`** | User-facing API for trace/span operations. | `RespanClient` |
| **`decorators/__init__.py`** | The 4 decorator functions — thin wrappers around `create_entity_method()`. | `workflow()`, `task()`, `agent()`, `tool()` |
| **`decorators/base.py`** | Decorator implementation. Creates spans, records input/output, handles sync/async/generators. | `create_entity_method()`, `_setup_span()` |
| **`processors/base.py`** | The 3-layer processor chain + SpanBuffer. | `RespanSpanProcessor`, `FilteringSpanProcessor`, `BufferingSpanProcessor`, `SpanBuffer` |
| **`exporters/respan.py`** | Converts OTEL spans → OTLP JSON, POSTs to `/v2/traces`. Anti-recursion. Root-span promotion. | `RespanSpanExporter`, `ModifiedSpan` |
| **`utils/span_factory.py`** | Shared utilities for constructing + injecting spans into the pipeline. Also holds `propagate_attributes()`. | `build_readable_span()`, `inject_span()`, `propagate_attributes()`, `read_propagated_attributes()` |
| **`utils/instrumentation.py`** | 800 lines: 32 `_init_*()` functions + OpenAI monkey-patch. *(deprecated — to be extracted)* | `init_instrumentations()` |
| **`utils/preprocessing/span_processing.py`** | Decides which spans to keep vs filter out. | `is_processable_span()`, `is_root_span_candidate()` |
| **`filters/evaluator.py`** | Evaluates export filter conditions on span attributes. | `evaluate_export_filter()` |
| **`contexts/span.py`** | Context manager for setting Respan-specific span attributes. | `respan_span_attributes()` |
| **`constants/`** | String constants: tracer name, attribute keys, config keys. | `TRACER_NAME`, `EXPORT_FILTER_ATTR`, etc. |
| **`testing/exporters.py`** | In-memory exporter for tests. | `InMemorySpanExporter` |
| **`utils/logging.py`** | Logging setup + span export preview. | `get_respan_logger()` |
| **`utils/context.py`** | OTEL context helper. | `get_entity_path()` |
| **`utils/imports.py`** | Dynamic import from string. | `import_from_string()` |

### respan/ (respan-ai)

| File | What it does | Key exports |
|------|-------------|-------------|
| **`_core.py`** | `Respan` class: sets up telemetry + activates plugins + `log_batch_results()`. | `Respan` |
| **`_types.py`** | The `Instrumentation` protocol definition. | `Instrumentation` |
| **`__init__.py`** | Re-exports everything from both `_core` and `respan-tracing`. | All public API |

---

## 6. Architecture Rating & Issues

### Rating: 7.5/10

Good bones. The unified OTEL pipeline is clean. The plugin protocol is simple. The span_factory is well-designed. One structural problem remains (built-in instrumentations), but the path forward is clear.

### What's Good

| Aspect | Why |
|--------|-----|
| **Single pipeline** | Everything flows through ReadableSpan → processors → /v2/traces. No dual-pipeline confusion. |
| **Singleton tracer** | `RespanTracer.__new__` + Lock guarantees one TracerProvider. |
| **Plugin protocol** | Simple `activate()`/`deactivate()`. Adding a new integration = new package, not modifying core. |
| **span_factory.py** | `build_readable_span()` + `inject_span()` is a clean interface for plugins to emit spans. |
| **propagate_attributes()** | Elegant ContextVar-based context propagation. Works with async. Supports nesting. |
| **Anti-recursion** | The exporter uses `_SUPPRESS_INSTRUMENTATION_KEY` to prevent infinite span loops. |
| **Processor composability** | Buffering → Filtering → Respan → Batch → Export chain is OTEL-idiomatic. |

### What Needs Work

#### 32 Instrumentations Still Baked Into respan-tracing

`utils/instrumentation.py` is **800 lines** with 32 `_init_*()` functions + a 150-line OpenAI monkey-patch. Plus `instruments.py` has a 32-value enum. Plus a 71-line `if/elif` dispatcher.

**Why this is a problem:**
- `respan-tracing` has a hard dependency on `opentelemetry-instrumentation-openai` in `pyproject.toml`
- Adding a new library means modifying the core package
- The `Instruments` enum and dispatcher must be manually kept in sync
- This contradicts the plugin pattern already established by `respan-instrumentation-*` packages

#### processors/base.py Has Too Many Responsibilities

Four unrelated classes in one 455-line file:
- `RespanSpanProcessor` (metadata enrichment + span filtering)
- `BufferingSpanProcessor` (optional buffering)
- `FilteringSpanProcessor` (routing to specific exporters)
- `SpanBuffer` (manual span collection)

Could be split into separate files for clarity.

#### Legacy Exporter Packages

5 of 7 `respan-exporter-*` packages have no `respan-instrumentation-*` counterpart yet. They use the old dict-based `/v1/traces/ingest` pattern and should be migrated to the OTEL pipeline.

---

## 7. Target Architecture

### The Goal

**Zero** built-in instrumentations in `respan-tracing`. Every integration is a separate `respan-instrumentation-*` package. Legacy exporters are replaced. The core only provides the engine.

### Target Dependency Graph

```
┌─────────────────────────────────────────────────┐
│                Application Code                  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  respan-ai                                       │
│                                                  │
│  - Respan() class                               │
│  - Auto-discovers installed                      │
│    respan-instrumentation-* via entry_points     │
│  - Activates plugins                             │
│  - Re-exports decorators + client                │
│                                                  │
│  depends on: respan-tracing (only)               │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────────┐
│  respan-tracing  │  │  respan-instrumentation-  │
│                  │  │  openai                    │
│  PURE ENGINE:    │  │  anthropic                 │
│  - TracerProvider│  │  langchain                 │
│  - Processors    │  │  litellm                   │
│  - Exporter      │  │  agno                      │
│  - Decorators    │  │  haystack                  │
│  - Client        │  │  ...etc (one per library)  │
│  - span_factory  │  │                            │
│                  │  │  Each:                     │
│  NO instruments! │  │  - activate()/deactivate() │
│  NO library deps!│  │  - depends on:             │
│                  │  │    respan-tracing          │
└──────────────────┘  │    + its own OTEL library  │
          │           └──────────────────────────┘
          ▼
┌──────────────────┐
│  respan-sdk      │
│  (types only)    │
└──────────────────┘
```

### How Auto-Discovery Would Work

Instead of a hardcoded enum, `respan-ai` discovers installed plugins via Python entry points:

```python
# respan/_core.py (target)
import importlib.metadata

def _discover_instrumentations():
    """Find all installed respan-instrumentation-* packages."""
    discovered = []
    for ep in importlib.metadata.entry_points(group="respan.instrumentations"):
        instrumentor_cls = ep.load()
        discovered.append(instrumentor_cls())
    return discovered
```

Each instrumentation package registers itself:

```toml
# respan-instrumentation-openai/pyproject.toml
[tool.poetry.plugins."respan.instrumentations"]
openai = "respan_instrumentation_openai:OpenAIInstrumentor"
```

Then `Respan()` with no arguments auto-discovers everything installed:

```python
respan = Respan()  # Finds and activates all installed instrumentations
```

### What respan-tracing Looks Like After (Target)

```
respan_tracing/
├── __init__.py
├── main.py                              # RespanTelemetry (no auto_instrument)
├── core/
│   ├── tracer.py                        # RespanTracer (no instrumentation logic)
│   └── client.py                        # RespanClient
├── decorators/
│   ├── __init__.py                      # workflow, task, agent, tool
│   └── base.py
├── processors/
│   ├── respan_processor.py              # RespanSpanProcessor
│   ├── filtering_processor.py           # FilteringSpanProcessor
│   ├── buffering_processor.py           # BufferingSpanProcessor
│   └── span_buffer.py                   # SpanBuffer
├── exporters/
│   └── respan.py                        # RespanSpanExporter → /v2/traces
├── utils/
│   ├── span_factory.py                  # build_readable_span, inject_span, propagate_attributes
│   ├── preprocessing/span_processing.py
│   ├── context.py
│   ├── logging.py
│   └── imports.py
├── filters/
│   └── evaluator.py
├── contexts/
│   └── span.py
├── constants/
│   └── ...
└── testing/
    └── exporters.py
```

Zero library-specific knowledge. Zero auto-instrumentation code. Just the engine.

---

## 8. Migration Plan

### Overview

The migration extracts 32 built-in instrumentations from `respan-tracing` into separate packages, deprecates 7 legacy exporters, and adds auto-discovery — all **without breaking existing users**.

```
Phase 1  ──── Clean dead code (done)
Phase 1b ──── Structural refactoring (processors, minor cleanup)
Phase 2  ──── Backward-compatible extraction (per-instrument, gradual)
Phase 3  ──── Auto-discovery in respan-ai
Phase 4  ──── Deprecate legacy exporters
Phase 5  ──── Final cleanup (remove old code)
```

### Phase 1: Clean Dead Code ✅ DONE

Already completed:
- Deleted `contexts/stdio.py` (unused stdout/stderr suppression)
- Deleted `utils/notebook.py` (unused Jupyter detection)
- Removed stale import of `is_notebook` from `core/tracer.py`
- Removed unused functions from `utils/imports.py`
- Removed unused constants `WORKFLOW_NAME_KEY`, `ENTITY_PATH_KEY`
- Cleaned stale imports in `decorators/base.py` and `utils/context.py`

### Phase 1b: Structural Refactoring

Non-breaking changes that improve code clarity before the big extraction begins.

#### Split `processors/base.py` into separate files (-0.75 gap)

Currently 4 unrelated classes in one 455-line file. They don't share private state or helpers — just co-located.

**Before:**
```
processors/
└── base.py   # RespanSpanProcessor, BufferingSpanProcessor,
              # FilteringSpanProcessor, SpanBuffer (455 lines)
```

**After:**
```
processors/
├── __init__.py                # Re-exports all 4 classes (backward compatible)
├── respan_processor.py        # RespanSpanProcessor (enrichment + filtering)
├── filtering_processor.py     # FilteringSpanProcessor (routing to exporters)
├── buffering_processor.py     # BufferingSpanProcessor (optional buffering)
└── span_buffer.py             # SpanBuffer (manual span collection)
```

Keep `from respan_tracing.processors.base import RespanSpanProcessor` working via re-export in `base.py` or `__init__.py` for backward compat.

#### Simplify `auto_instrument` defaulting logic (-0.25 gap)

Current logic in `respan/_core.py` flips `auto_instrument` based on whether `instrumentations` is provided — this is confusing. Make it explicit:

```python
# Current (confusing)
auto_instrument = not bool(instrumentations) if auto_instrument is None else auto_instrument

# Target (clear)
auto_instrument = auto_instrument or False  # Default: off. User must opt in.
```

#### Add type hints to processor chain internals (-0.25 gap)

Functions in the processor chain (`on_start`, `on_end`, internal helpers) are missing type annotations. Add them so new developers can follow the data flow:

```python
# Before
def on_end(self, span):
    if not is_processable_span(span):
        return

# After
def on_end(self, span: ReadableSpan) -> None:
    if not is_processable_span(span):
        return
```

#### Document the `ModifiedSpan` / `EnrichedSpan` proxy pattern (-0.25 gap)

The exporter wraps spans in `ModifiedSpan` (in `exporters/respan.py`) to inject `llm.request.type="chat"` and clear parent for root promotion. This is a non-obvious proxy pattern — add a docstring explaining *why* it exists:

```python
class ModifiedSpan:
    """Proxy around ReadableSpan that allows attribute/parent overrides at export time.

    Why: OTEL ReadableSpan is immutable after creation. But the exporter needs to:
    1. Inject llm.request.type="chat" for GenAI spans that are missing it
       (required by Respan backend to trigger prompt/completion parsing)
    2. Clear parent_span_id for root-promoted spans

    This proxy delegates all attributes to the wrapped span but overrides
    specific fields without mutating the original.
    """
```

### Phase 2: Extract Built-in Instrumentations (Backward Compatible)

**Strategy:** Keep the old code working as a fallback. Extract one instrument at a time. Users who install the new package get the new code; users who don't are unaffected.

#### 2a. For each instrument, modify the `_init_*()` function:

```python
# BEFORE (current)
def _init_anthropic() -> bool:
    if not is_package_installed("anthropic"):
        return False
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
    instrumentor = AnthropicInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()
    return True

# AFTER (backward compatible)
def _init_anthropic() -> bool:
    # Prefer the extracted package if installed
    try:
        from respan_instrumentation_anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().activate()
        return True
    except ImportError:
        pass

    # Fallback: inline code (deprecated)
    if not is_package_installed("anthropic"):
        return False
    import warnings
    warnings.warn(
        "Built-in Anthropic instrumentation is deprecated. "
        "Install respan-instrumentation-anthropic instead: "
        "pip install respan-instrumentation-anthropic",
        DeprecationWarning,
        stacklevel=3,
    )
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
    instrumentor = AnthropicInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()
    return True
```

#### 2b. Create the new package:

```
respan-instrumentation-anthropic/
├── pyproject.toml
└── src/
    └── respan_instrumentation_anthropic/
        ├── __init__.py          # exports AnthropicInstrumentor
        └── instrumentor.py      # activate() / deactivate()
```

```toml
# pyproject.toml
[tool.poetry]
name = "respan-instrumentation-anthropic"
version = "1.0.0"

[tool.poetry.dependencies]
python = ">=3.11,<3.14"
respan-tracing = ">=2.3.0"
opentelemetry-instrumentation-anthropic = ">=0.48.0"

[tool.poetry.plugins."respan.instrumentations"]
anthropic = "respan_instrumentation_anthropic:AnthropicInstrumentor"
```

```python
# instrumentor.py
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor as _OTELAnthropicInstrumentor

class AnthropicInstrumentor:
    name = "anthropic"

    def activate(self) -> None:
        instrumentor = _OTELAnthropicInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()

    def deactivate(self) -> None:
        try:
            _OTELAnthropicInstrumentor().uninstrument()
        except Exception:
            pass
```

#### 2c. Extraction order:

| Priority | Instruments | Why this order |
|----------|------------|----------------|
| 1st batch | marqo, lancedb, alephalpha, weaviate, milvus | Simplest (identical 13-line functions), lowest usage, lowest risk |
| 2nd batch | cohere, mistral, ollama, groq, together, replicate | Same simple pattern, slightly more popular |
| 3rd batch | anthropic, bedrock, sagemaker, vertexai, google_generativeai, watsonx | Cloud AI services |
| 4th batch | pinecone, qdrant, chroma | Vector databases |
| 5th batch | langchain, llama_index, haystack, crew, mcp | Frameworks |
| 6th batch | redis, requests, urllib3, pymysql | Infrastructure |
| 7th batch | transformers | ML library |
| 8th (last) | openai | Most complex — includes the 150-line `_patch_chat_prompt_capture` |
| Special | threading | Keep as built-in OR extract — it's for context propagation, not a library integration |

### Phase 3: Auto-Discovery in respan-ai

Once the first few packages are extracted, add entry-point discovery to `respan-ai`:

```python
# respan/_core.py
import importlib.metadata
import logging

logger = logging.getLogger(__name__)

def _discover_instrumentations():
    """Auto-discover installed respan-instrumentation-* packages."""
    discovered = []
    for ep in importlib.metadata.entry_points(group="respan.instrumentations"):
        try:
            instrumentor_cls = ep.load()
            discovered.append(instrumentor_cls())
            logger.debug("Discovered instrumentation: %s", ep.name)
        except Exception as exc:
            logger.warning("Failed to load instrumentation %s: %s", ep.name, exc)
    return discovered
```

Update `Respan.__init__()`:

```python
def __init__(self, ..., instrumentations=None, auto_instrument=None, ...):
    # If no explicit instrumentations, auto-discover
    if instrumentations is None:
        instrumentations = _discover_instrumentations()

    # auto_instrument=False by default (plugins handle it)
    if auto_instrument is None:
        auto_instrument = not bool(instrumentations)

    # ... rest unchanged
```

**User experience:**
```python
# Before (explicit)
from respan_instrumentation_openai import OpenAIInstrumentor
respan = Respan(instrumentations=[OpenAIInstrumentor()])

# After (auto-discovery — just install the package)
# pip install respan-instrumentation-openai
respan = Respan()  # Finds and activates OpenAIInstrumentor automatically
```

### Phase 4: Deprecate Legacy Exporters

For the 7 `respan-exporter-*` packages:

| Package | Action |
|---------|--------|
| `respan-exporter-anthropic-agents` | Already superseded. Add deprecation notice in README, point to `respan-instrumentation-anthropic-agents`. |
| `respan-exporter-openai-agents` | Already superseded. Add deprecation notice in README, point to `respan-instrumentation-openai-agents`. |
| `respan-exporter-litellm` | Create `respan-instrumentation-litellm` using the OTEL pipeline. Deprecate old package. |
| `respan-exporter-agno` | Create `respan-instrumentation-agno` using the OTEL pipeline. Deprecate old package. |
| `respan-exporter-haystack` | Create `respan-instrumentation-haystack` using the OTEL pipeline. Deprecate old package. |
| `respan-exporter-braintrust` | Create `respan-instrumentation-braintrust` using the OTEL pipeline. Deprecate old package. |
| `respan-exporter-superagent` | Create `respan-instrumentation-superagent` using the OTEL pipeline. Deprecate old package. |

Each new package follows the same pattern as Phase 2 — implement `activate()`/`deactivate()`, use `build_readable_span()` + `inject_span()`, register an entry point.

### Phase 5: Final Cleanup

Once all 32 built-in instrumentations are extracted and the deprecation period has passed (2–3 minor releases):

1. **Delete from respan-tracing:**
   - `instruments.py` (the 32-value enum)
   - `utils/instrumentation.py` (the 800-line file)
   - Remove `opentelemetry-instrumentation-openai` from `pyproject.toml`
   - Remove `opentelemetry-instrumentation-threading` from `pyproject.toml`
   - Remove `auto_instrument` parameter from `RespanTelemetry` / `RespanTracer`
   - Remove `instruments` and `block_instruments` parameters

2. **Archive legacy exporters:**
   - Mark all `respan-exporter-*` packages as deprecated on PyPI
   - Add final release with deprecation warning that auto-fires on import

3. **Bump major version:**
   - `respan-tracing` → 3.0.0 (breaking: removed built-in instrumentations)
   - `respan-ai` → 4.0.0 (breaking: requires instrumentation packages to be installed separately)

### Version Timeline

```
v2.3.x  ── Current: built-in instrumentations work, no warnings
v2.4.x  ── Phase 2 begins: first batch extracted, fallbacks emit DeprecationWarning
v2.5-8  ── Phase 2 continues: more batches extracted
v2.9.x  ── Phase 2 complete: all 32 extracted, all fallbacks warn
         ── Phase 3: auto-discovery added to respan-ai
v3.0.0  ── Phase 5: built-in instrumentations removed (breaking change)
```
