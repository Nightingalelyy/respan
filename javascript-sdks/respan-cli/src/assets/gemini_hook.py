#!/usr/bin/env python3
"""
Respan Hook for Gemini CLI

Sends Gemini CLI LLM call data to Respan after each model response.
Uses Gemini CLI's AfterModel hook to capture request/response and forward
to Respan's trace ingest API.

Handles streaming: Gemini fires AfterModel per chunk. We accumulate text
and only send on the final chunk (empty text or finishReason=STOP).

Handles tool calls: When the model calls a tool (file read, shell command,
web search), the turn ends with finishReason=STOP, then Gemini CLI executes
the tool and starts a new model turn. The hook detects this via message count
changes (the model's pre-tool response is added to the messages array) and
carries the accumulator across turns, so a single span captures the complete
response including text from before and after tool execution.

Detection strategy:
  - If the response contains functionCall/toolCall parts: immediate detection
    (safety net for future Gemini CLI versions that may include this data).
  - If message count increases with a model-role message after a send: the
    model resumed after a tool call. Bump send_version to cancel any pending
    delayed sender and continue accumulating.
  - text + STOP (method b): send immediately — this is never a tool-call
    boundary, the model produced text and said STOP on the same chunk.
  - empty + STOP or empty after text (method a): delay the send by SEND_DELAY
    seconds. If tool-call text arrives before the delay fires, the pending
    send is canceled and the accumulator continues.

Configuration:
    Auth:    ~/.respan/credentials.json  (from `respan auth login`)
    Config:  ~/.gemini/respan.json       (from `respan integrate gemini-cli`)
    Debug:   GEMINI_RESPAN_DEBUG=true    (check ~/.gemini/state/respan_hook.log)
    Delay:   GEMINI_RESPAN_SEND_DELAY=10 (seconds to wait before sending on
             empty chunks; increase for slow tools like web search)
"""

import contextlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # Not available on Windows

# ── SDK constants (import with fallback for standalone deployment) ─
# When the hook is pip-installed alongside respan-sdk, these come from the
# canonical source.  When deployed as a standalone script (the common case for
# `respan integrate gemini-cli`), the except branch provides local copies that
# mirror the SDK values exactly.
try:
    from respan_sdk.constants.api_constants import (
        DEFAULT_RESPAN_API_BASE_URL,
        TRACES_INGEST_PATH,
    )
    from respan_sdk.constants.tracing_constants import (
        RESPAN_DOGFOOD_HEADER,
        resolve_tracing_ingest_endpoint,
    )
except ImportError:
    DEFAULT_RESPAN_API_BASE_URL = "https://api.respan.ai/api"
    TRACES_INGEST_PATH = "v1/traces/ingest"
    RESPAN_DOGFOOD_HEADER = "X-Respan-Dogfood"

    def resolve_tracing_ingest_endpoint(base_url=None):
        if not base_url:
            return f"{DEFAULT_RESPAN_API_BASE_URL}/{TRACES_INGEST_PATH}"
        normalized = base_url.rstrip("/")
        if normalized.endswith("/api"):
            return f"{normalized}/{TRACES_INGEST_PATH}"
        return f"{normalized}/api/{TRACES_INGEST_PATH}"

# Map Gemini CLI built-in tool function names to friendly display names,
# matching the pattern used by the Codex hook (_tool_display_name).
# Source: base-declarations.ts (canonical names) + tool-names.ts (legacy aliases)
#   https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/definitions/base-declarations.ts
#   https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/tool-names.ts
GEMINI_TOOL_DISPLAY_NAMES = {
    "read_file": "File Read",
    "read_many_files": "File Read",
    "write_file": "File Write",
    "list_directory": "Directory List",
    "run_shell_command": "Shell",
    "google_web_search": "Web Search",
    "web_fetch": "Web Fetch",
    "glob": "Find Files",
    "grep_search": "Search Text",
    "search_file_content": "Search Text",  # legacy alias for grep_search
    "replace": "File Edit",
    "save_memory": "Memory",
    "write_todos": "Todos",
    "get_internal_docs": "Docs",
}


# Configuration
STATE_DIR = Path.home() / ".gemini" / "state"
LOG_FILE = STATE_DIR / "respan_hook.log"
LOCK_FILE = STATE_DIR / "respan_hook.lock"
DEBUG = os.environ.get("GEMINI_RESPAN_DEBUG", "").lower() == "true"

try:
    MAX_CHARS = int(os.environ.get("GEMINI_RESPAN_MAX_CHARS", "4000"))
except (ValueError, TypeError):
    MAX_CHARS = 4000

# Gemini-specific: delay before sending on empty chunks, to allow tool-call
# resumptions to cancel the pending send. Not needed by Claude/Codex hooks
# since they process complete transcripts rather than per-chunk streaming.
try:
    SEND_DELAY = int(os.environ.get("GEMINI_RESPAN_SEND_DELAY", "10"))
except (ValueError, TypeError):
    SEND_DELAY = 10


def log(level: str, message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} [{level}] {message}\n")


def debug(message: str) -> None:
    if DEBUG:
        log("DEBUG", message)


def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


# ── Credentials ──────────────────────────────────────────────────

def resolve_credentials() -> Tuple[Optional[str], str]:
    """Resolve API key and base URL from env vars or ~/.respan/credentials.json.

    Matches the credential resolution used by the Claude Code and Codex CLI hooks.
    """
    api_key = os.getenv("RESPAN_API_KEY")
    base_url = os.getenv("RESPAN_BASE_URL", DEFAULT_RESPAN_API_BASE_URL)

    if not api_key:
        creds_file = Path.home() / ".respan" / "credentials.json"
        if creds_file.exists():
            try:
                creds = json.loads(creds_file.read_text(encoding="utf-8"))
                config_file = Path.home() / ".respan" / "config.json"
                profile = "default"
                if config_file.exists():
                    cfg = json.loads(config_file.read_text(encoding="utf-8"))
                    profile = cfg.get("activeProfile", "default")
                cred = creds.get(profile, {})
                api_key = cred.get("apiKey") or cred.get("accessToken")
                if not base_url or base_url == DEFAULT_RESPAN_API_BASE_URL:
                    base_url = cred.get("baseUrl", base_url)
                if api_key:
                    debug(f"Using API key from credentials.json (profile: {profile})")
            except (json.JSONDecodeError, IOError) as e:
                debug(f"Failed to read credentials.json: {e}")

    # Also check respan.json for base_url (written by `respan integrate gemini-cli --base-url`)
    if not base_url or base_url == DEFAULT_RESPAN_API_BASE_URL:
        config = load_respan_config()
        cfg_base = config.get("fields", {}).get("base_url", "")
        if cfg_base:
            base_url = cfg_base

    return api_key, base_url


def load_respan_config() -> Dict[str, Any]:
    """Load ~/.gemini/respan.json for span field overrides.

    Returns a dict with two keys:
      - "fields": known span fields (customer_id, span_name, workflow_name)
      - "properties": everything else (custom properties -> metadata)
    """
    config_path = Path.home() / ".gemini" / "respan.json"
    if not config_path.exists():
        return {"fields": {}, "properties": {}}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"fields": {}, "properties": {}}
        fields = {}
        properties = {}
        for k, v in raw.items():
            if k in {"customer_id", "span_name", "workflow_name", "base_url"}:
                fields[k] = v
            else:
                properties[k] = v
        return {"fields": fields, "properties": properties}
    except (json.JSONDecodeError, IOError) as e:
        debug(f"Failed to load respan.json: {e}")
        return {"fields": {}, "properties": {}}


# ── Streaming accumulator ────────────────────────────────────────

def _state_path(session_id: str) -> Path:
    """Temp file to accumulate streamed text across hook invocations."""
    safe_id = session_id.replace("/", "_").replace("\\", "_")[:64]
    return STATE_DIR / f"respan_stream_{safe_id}.json"


def load_stream_state(session_id: str) -> Dict[str, Any]:
    p = _state_path(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"accumulated_text": "", "last_tokens": 0, "first_chunk_time": ""}


def save_stream_state(session_id: str, state: Dict[str, Any]) -> None:
    """Save state atomically via write-to-temp + rename."""
    p = _state_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.rename(tmp_path, str(p))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
    except OSError as e:
        log("ERROR", f"Failed to save state atomically, falling back: {e}")
        p.write_text(json.dumps(state), encoding="utf-8")


def clear_stream_state(session_id: str) -> None:
    p = _state_path(session_id)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


@contextlib.contextmanager
def state_lock(timeout: float = 5.0):
    """Acquire an advisory file lock around state operations."""
    if fcntl is None:
        yield
        return

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = open(LOCK_FILE, "w")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (IOError, OSError):
                if time.monotonic() >= deadline:
                    debug("Could not acquire state lock within timeout, proceeding without lock")
                    lock_fd.close()
                    lock_fd = None
                    break
                time.sleep(0.1)
    except Exception as e:
        debug(f"Lock error, proceeding without lock: {e}")
        if lock_fd is not None:
            with contextlib.suppress(Exception):
                lock_fd.close()
        lock_fd = None

    try:
        yield
    finally:
        if lock_fd is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                lock_fd.close()


# ── Data extraction ───────────────────────────────────────────────

def extract_messages(
    hook_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Extract prompt_messages from hook data.

    Preserves all content including Gemini CLI's <session_context> block
    for full trace fidelity.
    """
    llm_req = hook_data.get("llm_request", {})
    messages = llm_req.get("messages", [])
    formatted = []

    for msg in messages:
        # Gemini uses "model" for assistant; Respan API only accepts "assistant"
        raw_role = msg.get("role", "user")
        role = "assistant" if raw_role == "model" else raw_role
        content = msg.get("content", "")
        formatted.append({
            "role": role,
            "content": truncate(content),
        })

    return formatted


def detect_model(hook_data: Dict[str, Any]) -> str:
    """Detect model from hook data or environment."""
    override = os.environ.get("RESPAN_GEMINI_MODEL", "")
    if override:
        return override
    llm_req = hook_data.get("llm_request", {})
    model = llm_req.get("model", "")
    if model:
        return model
    return "gemini-cli"


# ── Tool formatting (matches Codex hook pattern) ─────────────────

def _tool_display_name(name: str) -> str:
    """Map Gemini CLI tool function names to friendly display names."""
    return GEMINI_TOOL_DISPLAY_NAMES.get(name, name or "Unknown")


def _format_tool_input(tool_name: str, args: Any) -> str:
    """Format tool call arguments for display in the span input field."""
    if not args:
        return ""
    if tool_name == "run_shell_command" and isinstance(args, dict):
        cmd = args.get("command", "")
        dir_path = args.get("dir_path", "")
        result = f"Command: {cmd}"
        if dir_path:
            result = f"[{dir_path}] {result}"
        return truncate(result)
    if tool_name in ("read_file", "write_file") and isinstance(args, dict):
        return truncate(args.get("file_path", json.dumps(args, default=str)))
    if tool_name == "read_many_files" and isinstance(args, dict):
        return truncate(args.get("include", json.dumps(args, default=str)))
    if tool_name == "google_web_search" and isinstance(args, dict):
        return truncate(f"Query: {args.get('query', str(args))}")
    if tool_name == "web_fetch" and isinstance(args, dict):
        return truncate(args.get("prompt", json.dumps(args, default=str)))
    if tool_name in ("glob", "grep_search", "search_file_content") and isinstance(args, dict):
        return truncate(args.get("pattern", json.dumps(args, default=str)))
    if tool_name == "replace" and isinstance(args, dict):
        path = args.get("file_path", "")
        old = args.get("old_string", "")
        if path and old:
            return truncate(f"{path}: {old!r} → ...")
        return truncate(json.dumps(args, default=str))
    if isinstance(args, dict):
        try:
            return truncate(json.dumps(args, indent=2))
        except (TypeError, ValueError):
            pass
    return truncate(str(args))


# ── Span construction ────────────────────────────────────────────

def build_spans(
    hook_data: Dict[str, Any],
    output_text: str,
    tokens: Dict[str, int],
    config: Optional[Dict[str, Any]] = None,
    start_time_iso: Optional[str] = None,
    tool_turns: int = 0,
    tool_details: Optional[List[Dict[str, Any]]] = None,
    thoughts_tokens: int = 0,
) -> List[Dict[str, Any]]:
    """Build Respan spans for a Gemini CLI LLM call.

    Produces a span tree matching the Claude Code / Codex hook structure:
        Root: gemini-cli (agent container — metadata, latency)
          ├── gemini.chat (generation — model, tokens, messages)
          ├── Reasoning (if thinking tokens > 0)
          └── Tool: {name} (one per tool turn, with formatted input)
    """
    spans: List[Dict[str, Any]] = []

    session_id = hook_data.get("session_id", "")
    model = detect_model(hook_data)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    end_time = hook_data.get("timestamp") or now_str
    begin_time = start_time_iso or end_time

    # Compute latency from tracked chunk times
    latency: Optional[float] = None
    try:
        t_start = datetime.fromisoformat(begin_time.replace("Z", "+00:00"))
        t_end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        latency = max((t_end - t_start).total_seconds(), 0.0)
    except (ValueError, TypeError):
        pass

    # Messages — session context preserved for trace fidelity
    prompt_messages = extract_messages(hook_data)
    completion_message: Dict[str, str] = {"role": "assistant", "content": truncate(output_text)}

    # Config overrides from respan.json
    cfg_fields = (config or {}).get("fields", {})
    cfg_props = (config or {}).get("properties", {})

    # IDs — keep under 64 chars to avoid silent drops on ingest.
    # Longest suffix is "_tool_99" (8 chars) + "gcli_" (5 chars) = 13.
    safe_id = session_id.replace("/", "_").replace("\\", "_")[:50]
    trace_unique_id = f"gcli_{safe_id}"
    root_span_id = f"gcli_{safe_id}_root"
    gen_span_id = f"gcli_{safe_id}_gen"
    workflow_name = os.environ.get("RESPAN_WORKFLOW_NAME") or cfg_fields.get("workflow_name") or "gemini-cli"
    root_span_name = os.environ.get("RESPAN_SPAN_NAME") or cfg_fields.get("span_name") or "gemini-cli"
    thread_id = f"gcli_{session_id}"
    customer_id = os.environ.get("RESPAN_CUSTOMER_ID") or cfg_fields.get("customer_id") or ""

    # LLM config
    llm_req = hook_data.get("llm_request", {})
    req_config = llm_req.get("config", {})

    # Metadata — custom properties from respan.json, then env overrides
    metadata: Dict[str, Any] = {"source": "gemini-cli"}
    if cfg_props:
        metadata.update(cfg_props)
    env_metadata = os.environ.get("RESPAN_METADATA")
    if env_metadata:
        try:
            extra = json.loads(env_metadata)
            if isinstance(extra, dict):
                metadata.update(extra)
        except json.JSONDecodeError:
            pass
    if tool_turns > 0:
        metadata["tool_turns"] = tool_turns
    if thoughts_tokens > 0:
        metadata["reasoning_tokens"] = thoughts_tokens

    # Token counts
    prompt_tokens = tokens.get("prompt_tokens", 0)
    completion_tokens = tokens.get("completion_tokens", 0)
    total_tokens = tokens.get("total_tokens", 0) or (prompt_tokens + completion_tokens)

    # ------------------------------------------------------------------
    # Root span — agent container (matches Claude Code / Codex pattern)
    # ------------------------------------------------------------------
    root_span: Dict[str, Any] = {
        "trace_unique_id": trace_unique_id,
        "thread_identifier": thread_id,
        "customer_identifier": customer_id,
        "span_unique_id": root_span_id,
        "span_name": root_span_name,
        "span_workflow_name": workflow_name,
        "log_type": "agent",
        "model": model,
        "provider_id": "",
        "span_path": "",
        "input": json.dumps(prompt_messages) if prompt_messages else "",
        "output": json.dumps(completion_message),
        "timestamp": end_time,
        "start_time": begin_time,
        "metadata": metadata,
    }
    if latency is not None:
        root_span["latency"] = latency
    spans.append(root_span)

    # ------------------------------------------------------------------
    # Generation child span — carries model, tokens, messages
    # ------------------------------------------------------------------
    gen_span: Dict[str, Any] = {
        "trace_unique_id": trace_unique_id,
        "span_unique_id": gen_span_id,
        "span_parent_id": root_span_id,
        "span_name": "gemini.chat",
        "span_workflow_name": workflow_name,
        "span_path": "gemini_chat",
        "model": model,
        "provider_id": "google",
        "log_type": "chat",
        "metadata": {},
        # Uses the current `input`/`output` fields (JSON-stringified messages).
        # The codex and claude-code hooks still use the legacy approach:
        #   "prompt_messages": prompt_messages,        # list of dicts
        #   "completion_message": completion_message,   # dict
        # See: https://respan.ai/docs/documentation/resources/reference/span-fields#prompt-messages
        "input": json.dumps(prompt_messages) if prompt_messages else "",
        "output": json.dumps(completion_message),
        "timestamp": end_time,
        "start_time": begin_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if latency is not None:
        gen_span["latency"] = latency
    # Optional LLM config fields
    if req_config.get("temperature") is not None:
        gen_span["temperature"] = req_config["temperature"]
    if req_config.get("maxOutputTokens") is not None:
        gen_span["max_tokens"] = req_config["maxOutputTokens"]
    spans.append(gen_span)

    # ------------------------------------------------------------------
    # Reasoning child span (if thinking/reasoning tokens detected)
    # Matches the Codex hook's dedicated reasoning span pattern.
    # ------------------------------------------------------------------
    if thoughts_tokens > 0:
        spans.append({
            "trace_unique_id": trace_unique_id,
            "span_unique_id": f"gcli_{safe_id}_reasoning",
            "span_parent_id": root_span_id,
            "span_name": "Reasoning",
            "span_workflow_name": workflow_name,
            "span_path": "reasoning",
            "provider_id": "",
            "metadata": {"reasoning_tokens": thoughts_tokens},
            "input": "",
            "output": f"[Reasoning: {thoughts_tokens} tokens]",
            "timestamp": end_time,
            "start_time": begin_time,
        })

    # ------------------------------------------------------------------
    # Tool child spans (one per detected tool turn)
    # Tool details come from BeforeTool/AfterTool hooks (primary) or
    # from functionCall parts in AfterModel candidates (fallback).
    # ------------------------------------------------------------------
    _details = tool_details or []
    for i in range(1, tool_turns + 1):
        detail = _details[i - 1] if i <= len(_details) else None
        tool_name = (detail.get("name", "") if detail else "") or ""
        # BeforeTool/AfterTool use "input", functionCall parts use "args"
        tool_args = (detail.get("args") or detail.get("input", {}) if detail else {}) or {}
        tool_output = (detail.get("output", "") if detail else "") or ""
        display_name = _tool_display_name(tool_name) if tool_name else f"Call {i}"
        tool_input_str = _format_tool_input(tool_name, tool_args) if tool_name else ""
        tool_meta: Dict[str, Any] = {}
        if tool_name:
            tool_meta["tool_name"] = tool_name
        if detail and detail.get("error"):
            tool_meta["error"] = detail["error"]

        # Use individual timing from BeforeTool/AfterTool if available,
        # otherwise fall back to the parent span's timestamps.
        tool_start = (detail.get("start_time") if detail else None) or begin_time
        tool_end = (detail.get("end_time") if detail else None) or end_time
        tool_latency: Optional[float] = None
        try:
            t0 = datetime.fromisoformat(tool_start.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(tool_end.replace("Z", "+00:00"))
            tool_latency = max((t1 - t0).total_seconds(), 0.0)
        except (ValueError, TypeError):
            pass

        tool_span: Dict[str, Any] = {
            "trace_unique_id": trace_unique_id,
            "span_unique_id": f"gcli_{safe_id}_tool_{i}",
            "span_parent_id": root_span_id,
            "span_name": f"Tool: {display_name}",
            "span_workflow_name": workflow_name,
            "span_path": f"tool_{tool_name}" if tool_name else "tool_call",
            "log_type": "tool",
            "provider_id": "",
            "metadata": tool_meta,
            "input": tool_input_str,
            "output": truncate(tool_output),
            "timestamp": tool_end,
            "start_time": tool_start,
        }
        if tool_latency is not None:
            tool_span["latency"] = tool_latency
        spans.append(tool_span)

    # Apply platform defaults (matches Claude Code / Codex hooks).
    # Most of these are gateway/proxy parameters (encoding_format, field_name,
    # delimiter, disable_fallback, disable_log, request_breakdown) that have no
    # effect on trace ingestion. Kept for consistency with the other hooks.
    respan_defaults = {
        "warnings": "",
        "encoding_format": "float",
        "disable_fallback": False,
        "field_name": "data: ",
        "delimiter": "\n\n",
        "disable_log": False,
        "request_breakdown": False,
    }
    for span in spans:
        for key, value in respan_defaults.items():
            if key not in span:
                span[key] = value
        if "respan_params" not in span:
            span["respan_params"] = {
                "has_webhook": False,
                "environment": os.environ.get("RESPAN_ENVIRONMENT", "prod"),
            }

    return spans


# ── Send to Respan ────────────────────────────────────────────────

def send_spans(
    spans: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
) -> None:
    """Send spans to Respan as a single batch via /v1/traces/ingest (array format).

    Writes the payload to a temp file and launches an inline Python script
    via ``subprocess.Popen`` so the HTTP request runs in a fully independent
    process (Gemini CLI may kill the hook after reading ``{}``).
    Uses ``urllib`` (stdlib) — no dependency on ``requests`` or external scripts.

    Note: respan_sdk's RetryHandler uses exponential backoff with jitter, but
    we just do flat 1s retries here to keep the inline subprocess script simple.
    """
    url = resolve_tracing_ingest_endpoint(base_url)

    span_names = [s.get("span_name", "?") for s in spans]
    debug(f"Sending {len(spans)} span(s) to {url}: {span_names}")

    if DEBUG:
        debug_file = STATE_DIR / "respan_last_payload.json"
        debug_file.write_text(json.dumps(spans, indent=2), encoding="utf-8")

    # Write payload to temp file and launch a fully independent sender process.
    # This avoids Gemini CLI killing the HTTP request mid-flight.
    import subprocess
    payload_file = STATE_DIR / f"respan_send_{os.getpid()}.json"
    payload_file.write_text(json.dumps(spans), encoding="utf-8")

    # Inline script uses urllib (stdlib) — no dependency on requests or
    # external send_spans.py.  API key is passed via environment variable
    # to avoid exposure in process listings.
    sender_script = (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "from urllib.request import Request, urlopen\n"
        "from urllib.error import URLError, HTTPError\n"
        f"pf = Path({str(payload_file)!r})\n"
        "try:\n"
        "    data = pf.read_bytes()\n"
        "    for attempt in range(3):\n"
        "        try:\n"
        f"            req = Request({url!r}, data=data, headers={{\n"
        '                "Content-Type": "application/json",\n'
        f'                "{RESPAN_DOGFOOD_HEADER}": "1",\n'
        '                "Authorization": "Bearer " + os.environ["RESPAN_API_KEY"],\n'
        "            })\n"
        "            urlopen(req, timeout=30)\n"
        "            break\n"
        "        except HTTPError as e:\n"
        "            if e.code < 500:\n"
        "                break\n"
        "            if attempt < 2:\n"
        "                time.sleep(1)\n"
        "        except (URLError, OSError):\n"
        "            if attempt < 2:\n"
        "                time.sleep(1)\n"
        "finally:\n"
        "    pf.unlink(missing_ok=True)\n"
    )

    env = os.environ.copy()
    env["RESPAN_API_KEY"] = api_key

    try:
        subprocess.Popen(
            ["python3", "-c", sender_script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        debug("Launched sender subprocess")
    except Exception as e:
        log("ERROR", f"Failed to launch sender: {e}")
        payload_file.unlink(missing_ok=True)


def launch_delayed_send(
    session_id: str,
    send_version: int,
    spans: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
) -> None:
    """Launch a background process that waits SEND_DELAY seconds, then sends.

    Before sending, the process reads the state file and checks if
    ``send_version`` still matches.  If the version changed (because new text
    arrived from a tool-call resumption), it skips the send and cleans up.
    """
    import subprocess

    payload_file = STATE_DIR / f"respan_delayed_{os.getpid()}.json"
    payload_file.write_text(json.dumps(spans), encoding="utf-8")

    state_file_path = str(_state_path(session_id))
    log_path = str(LOG_FILE)
    debug_flag = "1" if DEBUG else "0"
    url = resolve_tracing_ingest_endpoint(base_url)

    # Inline script: sleep, check version, then POST with urllib (stdlib).
    # API key is received via RESPAN_API_KEY env var — never interpolated
    # into the script string.
    script = f"""
import json, time, os, sys
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

def _log(msg):
    if {debug_flag!r} == "1":
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open({log_path!r}, "a") as f:
            f.write(f"{{ts}} [DEBUG] delayed_send: {{msg}}\\n")

time.sleep({SEND_DELAY})

state_file = Path({state_file_path!r})
payload_file = Path({str(payload_file)!r})

try:
    if not state_file.exists():
        _log("state file gone, skipping")
        payload_file.unlink(missing_ok=True)
        sys.exit(0)

    state = json.loads(state_file.read_text())
    current_version = state.get("send_version", 0)

    if current_version != {send_version}:
        _log(f"version mismatch (expected={send_version}, current={{current_version}}), skipping")
        payload_file.unlink(missing_ok=True)
        sys.exit(0)

    _log(f"version matches ({send_version}), sending")

    data = payload_file.read_bytes()
    for attempt in range(3):
        try:
            req = Request({url!r}, data=data, headers={{
                "Content-Type": "application/json",
                "{RESPAN_DOGFOOD_HEADER}": "1",
                "Authorization": "Bearer " + os.environ.get("RESPAN_API_KEY", ""),
            }})
            urlopen(req, timeout=30)
            _log("sent successfully")
            break
        except HTTPError as e:
            if e.code < 500:
                _log(f"client error {{e.code}}, not retrying")
                break
            if attempt < 2:
                _log(f"server error {{e.code}}, retrying in 1s")
                time.sleep(1)
        except (URLError, OSError) as e:
            if attempt < 2:
                _log(f"connection error {{e}}, retrying in 1s")
                time.sleep(1)

    # Clear state and payload now that we've sent
    state_file.unlink(missing_ok=True)
    payload_file.unlink(missing_ok=True)

except Exception as e:
    _log(f"error: {{e}}")
    payload_file.unlink(missing_ok=True)
"""

    env = os.environ.copy()
    env["RESPAN_API_KEY"] = api_key

    try:
        subprocess.Popen(
            ["python3", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        debug(f"Launched delayed sender (version={send_version}, delay={SEND_DELAY}s)")
    except Exception as e:
        log("ERROR", f"Failed to launch delayed sender: {e}")
        payload_file.unlink(missing_ok=True)


# ── BeforeTool / AfterTool handlers ──────────────────────────────

def _process_before_tool(hook_data: Dict[str, Any]) -> None:
    """Store tool name and input from BeforeTool hook into state."""
    session_id = hook_data.get("session_id", "unknown")
    tool_name = hook_data.get("tool_name", "")
    tool_input = hook_data.get("tool_input", {})

    debug(f"BeforeTool: {tool_name}")

    state = load_stream_state(session_id)
    pending = state.get("pending_tools", [])
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    pending.append({"name": tool_name, "input": tool_input, "start_time": now_str})
    state["pending_tools"] = pending
    save_stream_state(session_id, state)

    print("{}")
    sys.stdout.flush()


def _process_after_tool(hook_data: Dict[str, Any]) -> None:
    """Match pending tool and store output from AfterTool hook into state."""
    session_id = hook_data.get("session_id", "unknown")
    tool_name = hook_data.get("tool_name", "")
    tool_response = hook_data.get("tool_response", {})

    output = tool_response.get("llmContent", "")
    error = tool_response.get("error")
    debug(f"AfterTool: {tool_name}, output_len={len(output)}, error={error}")

    state = load_stream_state(session_id)
    pending = state.get("pending_tools", [])
    completed = state.get("tool_details", [])

    # Match last pending tool with this name
    for i in range(len(pending) - 1, -1, -1):
        if pending[i]["name"] == tool_name:
            detail = pending.pop(i)
            detail["output"] = output
            detail["end_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            if error:
                detail["error"] = error
            completed.append(detail)
            break

    state["pending_tools"] = pending
    state["tool_details"] = completed
    save_stream_state(session_id, state)

    print("{}")
    sys.stdout.flush()


def _process_chunk(hook_data: Dict[str, Any]) -> None:
    """Process a single streaming chunk under advisory file lock.

    This function contains all state read/modify/write logic. It is called
    from ``main()`` inside ``with state_lock():`` to prevent concurrent
    hook invocations from corrupting the accumulator state.
    """
    session_id = hook_data.get("session_id", "unknown")

    # Extract current chunk data
    llm_resp = hook_data.get("llm_response", {})
    chunk_text = llm_resp.get("text", "") or ""
    usage = llm_resp.get("usageMetadata", {})
    completion_tokens = usage.get("candidatesTokenCount", 0) or 0

    # Track thinking/reasoning tokens (e.g., Gemini 2.5 thinking mode)
    thoughts_tokens = usage.get("thoughtsTokenCount", 0) or 0

    # Check for finish signal and extract tool call details from candidates.
    # Gemini CLI currently filters out functionCall parts, but we extract
    # them as a safety net for future versions (and for rich tool spans).
    candidates = llm_resp.get("candidates", [])
    finish_reason = ""
    has_tool_call = False
    chunk_tool_details: List[Dict[str, Any]] = []
    if candidates and isinstance(candidates, list) and isinstance(candidates[0], dict):
        finish_reason = candidates[0].get("finishReason", "")
        content = candidates[0].get("content", {})
        if isinstance(content, dict):
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    continue
                fc = part.get("functionCall") or part.get("toolCall")
                if fc:
                    has_tool_call = True
                    if isinstance(fc, dict):
                        chunk_tool_details.append({
                            "name": fc.get("name", ""),
                            "args": fc.get("args", {}),
                        })

    # Message count for detecting tool-call resumptions across turns.
    messages = hook_data.get("llm_request", {}).get("messages", [])
    current_msg_count = len(messages)

    # Load accumulated state
    state = load_stream_state(session_id)

    is_finished = finish_reason in ("STOP", "MAX_TOKENS", "SAFETY")

    # ── Step 0: Detect tool-call resumption via message count ────
    saved_msg_count = state.get("msg_count", 0)
    tool_call_detected = False

    if saved_msg_count > 0 and current_msg_count > saved_msg_count:
        new_msgs = messages[saved_msg_count:]
        has_new_user_msg = any(
            m.get("role") == "user" for m in new_msgs
        )
        if has_new_user_msg:
            debug(
                f"New user message detected "
                f"(msgs {saved_msg_count} → {current_msg_count}), "
                f"starting fresh turn"
            )
            clear_stream_state(session_id)
            state = {
                "accumulated_text": "", "last_tokens": 0,
                "first_chunk_time": "",
            }
        else:
            state["tool_turns"] = state.get("tool_turns", 0) + 1
            state["send_version"] = state.get("send_version", 0) + 1
            tool_call_detected = True
            debug(
                f"Tool call detected via msg_count "
                f"({saved_msg_count} → {current_msg_count}), "
                f"keeping {len(state['accumulated_text'])} chars buffered, "
                f"tool_turns={state['tool_turns']}, "
                f"send_version={state['send_version']}"
            )

    state["msg_count"] = current_msg_count

    # ── Step 1: Accumulate text chunks ───────────────────────────
    if chunk_text:
        if not state.get("first_chunk_time"):
            state["first_chunk_time"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z"
        state["accumulated_text"] += chunk_text
        state["last_tokens"] = completion_tokens or state.get("last_tokens", 0)
        if thoughts_tokens > 0:
            state["thoughts_tokens"] = thoughts_tokens
        save_stream_state(session_id, state)
        debug(
            f"Accumulated chunk: +{len(chunk_text)} chars, "
            f"total={len(state['accumulated_text'])}"
        )

    # ── Step 1.5: functionCall/toolCall in response parts ────────
    is_tool_turn = has_tool_call or finish_reason in (
        "TOOL_CALLS", "FUNCTION_CALL", "TOOL_USE",
    )
    if is_tool_turn:
        state["tool_turns"] = state.get("tool_turns", 0) + 1
        state["send_version"] = state.get("send_version", 0) + 1
        # Store extracted tool details for rich span construction
        if chunk_tool_details:
            existing = state.get("tool_details", [])
            existing.extend(chunk_tool_details)
            state["tool_details"] = existing
        save_stream_state(session_id, state)
        debug(
            f"Tool call detected via response parts "
            f"(finish_reason={finish_reason}, has_tool_call={has_tool_call}), "
            f"carrying accumulator, tool_turns={state['tool_turns']}"
        )
        print("{}")
        sys.stdout.flush()
        return

    # ── Step 2: Detect completion and send ───────────────────────
    has_new_text = (
        len(state.get("accumulated_text", ""))
        > state.get("last_send_text_len", 0)
    )
    # Allow sending when is_finished even if tool_call_detected on the same
    # chunk (e.g. tool-call resumption completes in a single text+STOP chunk).
    should_send = (
        (not tool_call_detected or is_finished)
        and has_new_text
        and state["accumulated_text"]
        and ((not chunk_text) or is_finished)
    )

    # Print response immediately so Gemini CLI can proceed
    print("{}")
    sys.stdout.flush()

    if not should_send:
        if tool_call_detected:
            save_stream_state(session_id, state)
        return

    # Resolve credentials before deciding send strategy
    api_key, base_url = resolve_credentials()
    if not api_key:
        log("ERROR", "No API key found. Run: respan auth login")
        clear_stream_state(session_id)
        return

    final_prompt = usage.get("promptTokenCount", 0) or 0
    final_completion = completion_tokens or state.get("last_tokens", 0)
    final_total = usage.get("totalTokenCount", 0) or 0
    tok = {
        "prompt_tokens": final_prompt,
        "completion_tokens": final_completion,
        "total_tokens": final_total or (final_prompt + final_completion),
    }
    config = load_respan_config()
    spans = build_spans(
        hook_data,
        state["accumulated_text"],
        tok,
        config,
        start_time_iso=state.get("first_chunk_time"),
        tool_turns=state.get("tool_turns", 0),
        tool_details=state.get("tool_details", []),
        thoughts_tokens=state.get("thoughts_tokens", 0),
    )

    n_tool_turns = state.get("tool_turns", 0)

    # Method b: text + STOP → send immediately.
    if is_finished and chunk_text:
        debug(
            f"Immediate send (text+STOP, tool_turns={n_tool_turns}), "
            f"sending {len(state['accumulated_text'])} chars"
        )
        send_spans(spans, api_key, base_url)
        clear_stream_state(session_id)
        return

    # Method a: empty chunk after accumulated text — delayed send.
    state["send_version"] = state.get("send_version", 0) + 1
    state["last_send_text_len"] = len(state["accumulated_text"])
    save_stream_state(session_id, state)
    debug(
        f"Delayed send (version={state['send_version']}, "
        f"tool_turns={n_tool_turns}, delay={SEND_DELAY}s), "
        f"buffered {len(state['accumulated_text'])} chars"
    )
    launch_delayed_send(
        session_id, state["send_version"],
        spans, api_key, base_url,
    )


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print("{}")
            return

        hook_data = json.loads(raw)
        event = hook_data.get("hook_event_name", "")

        with state_lock():
            if event == "BeforeTool":
                _process_before_tool(hook_data)
            elif event == "AfterTool":
                _process_after_tool(hook_data)
            else:
                _process_chunk(hook_data)

    except json.JSONDecodeError as e:
        log("ERROR", f"Invalid JSON from stdin: {e}")
        print("{}")
    except Exception as e:
        log("ERROR", f"Hook error: {e}")
        print("{}")


if __name__ == "__main__":
    main()
