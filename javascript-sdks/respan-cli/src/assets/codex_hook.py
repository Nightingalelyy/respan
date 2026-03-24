#!/usr/bin/env python3
"""
Respan Hook for Codex CLI

Sends Codex CLI conversation traces to Respan after each agent turn.
Uses Codex CLI's notify hook to capture session JSONL files and convert
them to Respan spans.

Usage:
    Add to ~/.codex/config.toml:
        notify = ["python3", "~/.respan/codex_hook.py"]
    Run: respan integrate codex-cli
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
    import requests
except ImportError:
    print("Error: Python 'requests' package is required. Install: pip3 install requests", file=sys.stderr)
    sys.exit(1)

try:
    import fcntl
except ImportError:
    fcntl = None  # Not available on Windows

# Configuration
LOG_FILE = Path.home() / ".codex" / "state" / "respan_hook.log"
STATE_FILE = Path.home() / ".codex" / "state" / "respan_state.json"
LOCK_FILE = Path.home() / ".codex" / "state" / "respan_hook.lock"
DEBUG = os.environ.get("CODEX_RESPAN_DEBUG", "").lower() == "true"

try:
    MAX_CHARS = int(os.environ.get("CODEX_RESPAN_MAX_CHARS", "4000"))
except (ValueError, TypeError):
    MAX_CHARS = 4000


def log(level: str, message: str) -> None:
    """Log a message to the log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} [{level}] {message}\n")


def debug(message: str) -> None:
    """Log a debug message (only if DEBUG is enabled)."""
    if DEBUG:
        log("DEBUG", message)


def load_state() -> Dict[str, Any]:
    """Load the state file containing session tracking info."""
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return {}


def save_state(state: Dict[str, Any]) -> None:
    """Save the state file atomically via write-to-temp + rename."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.rename(tmp_path, STATE_FILE)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
    except OSError as e:
        log("ERROR", f"Failed to save state atomically, falling back: {e}")
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


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


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


# Known config keys in respan.json that map to span fields.
KNOWN_CONFIG_KEYS = {"customer_id", "span_name", "workflow_name"}


def load_respan_config(cwd: str) -> Dict[str, Any]:
    """Load .codex/respan.json from the project directory.

    Returns a dict with two keys:
      - "fields": known span fields (customer_id, span_name, workflow_name)
      - "properties": everything else (custom properties -> metadata)
    """
    config_path = Path(cwd) / ".codex" / "respan.json"
    if not config_path.exists():
        return {"fields": {}, "properties": {}}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"fields": {}, "properties": {}}
        fields = {}
        properties = {}
        for k, v in raw.items():
            if k in KNOWN_CONFIG_KEYS:
                fields[k] = v
            else:
                properties[k] = v
        return {"fields": fields, "properties": properties}
    except (json.JSONDecodeError, IOError) as e:
        debug(f"Failed to load respan.json from {config_path}: {e}")
        return {"fields": {}, "properties": {}}


def truncate(text: str, max_length: int = MAX_CHARS) -> str:
    """Truncate text to max_length."""
    if len(text) > max_length:
        return text[:max_length] + "\n... (truncated)"
    return text


def find_session_file(session_id: str) -> Optional[Path]:
    """Find the session JSONL file for a given session ID.

    Codex CLI stores sessions at:
      ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<session-id>.jsonl

    The session ID appears in the filename after the timestamp prefix.
    """
    sessions_dir = Path.home() / ".codex" / "sessions"
    if not sessions_dir.exists():
        debug(f"Sessions directory not found: {sessions_dir}")
        return None

    # Search date directories in reverse order (newest first)
    for year_dir in sorted(sessions_dir.iterdir(), reverse=True):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue
            for day_dir in sorted(month_dir.iterdir(), reverse=True):
                if not day_dir.is_dir():
                    continue
                for f in day_dir.glob("*.jsonl"):
                    if session_id in f.name:
                        debug(f"Found session file: {f}")
                        return f

    debug(f"No session file found for session ID: {session_id}")
    return None


def find_latest_session_file() -> Optional[Tuple[str, Path]]:
    """Find the most recently modified session JSONL file.

    Returns (session_id, path) or None.
    """
    sessions_dir = Path.home() / ".codex" / "sessions"
    if not sessions_dir.exists():
        return None

    latest_file = None
    latest_mtime = 0

    for year_dir in sessions_dir.iterdir():
        if not year_dir.is_dir():
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            for day_dir in month_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                for f in day_dir.glob("*.jsonl"):
                    mtime = f.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_file = f

    if latest_file:
        # Extract session ID from first line
        try:
            first_line = latest_file.read_text(encoding="utf-8").split("\n")[0]
            if first_line:
                first_msg = json.loads(first_line)
                payload = first_msg.get("payload", {})
                session_id = payload.get("id", latest_file.stem)
                return (session_id, latest_file)
        except (json.JSONDecodeError, IOError, IndexError) as e:
            debug(f"Error reading session file {latest_file}: {e}")

    return None


def parse_session(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse JSONL lines into a list of events."""
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def extract_turns(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract turns from session events.

    A turn is bounded by task_started and task_complete events.
    Returns a list of turn dicts, each containing:
      - turn_id: str
      - start_time: str (ISO timestamp)
      - end_time: str (ISO timestamp)
      - model: str
      - cwd: str
      - user_message: str
      - assistant_message: str
      - commentary: List[str]
      - tool_calls: List[dict]
      - tool_outputs: List[dict]
      - reasoning: bool (whether reasoning was present)
      - token_usage: dict (from last_token_usage)
      - events: List[dict] (raw events in this turn)
    """
    turns = []
    current_turn = None

    for event in events:
        evt_type = event.get("type")
        payload = event.get("payload") or {}
        timestamp = event.get("timestamp", "")

        if evt_type == "event_msg":
            msg_type = payload.get("type", "")

            if msg_type == "task_started":
                current_turn = {
                    "turn_id": payload.get("turn_id", ""),
                    "start_time": timestamp,
                    "end_time": "",
                    "model": "",
                    "cwd": "",
                    "user_message": "",
                    "assistant_message": "",
                    "commentary": [],
                    "tool_calls": [],
                    "tool_outputs": [],
                    "reasoning": False,
                    "token_usage": {},
                    "events": [],
                }

            elif msg_type == "task_complete" and current_turn:
                current_turn["end_time"] = timestamp
                current_turn["_complete_payload"] = payload
                turns.append(current_turn)
                current_turn = None

            elif msg_type == "user_message" and current_turn:
                current_turn["user_message"] = payload.get("message", "")

            elif msg_type == "agent_message" and current_turn:
                phase = payload.get("phase", "")
                message = payload.get("message", "")
                if phase == "final_answer":
                    current_turn["assistant_message"] = message
                elif phase == "commentary":
                    current_turn["commentary"].append(message)

            elif msg_type == "token_count" and current_turn:
                info = payload.get("info") or {}
                last_usage = info.get("last_token_usage") or {}
                if last_usage:
                    current_turn["token_usage"] = last_usage

        elif evt_type == "turn_context" and current_turn:
            current_turn["model"] = payload.get("model", "")
            current_turn["cwd"] = payload.get("cwd", "")

        elif evt_type == "response_item" and current_turn:
            item_type = payload.get("type", "")

            if item_type == "function_call":
                current_turn["tool_calls"].append({
                    "name": payload.get("name", "unknown"),
                    "arguments": payload.get("arguments", ""),
                    "call_id": payload.get("call_id", ""),
                    "timestamp": timestamp,
                })

            elif item_type == "custom_tool_call":
                current_turn["tool_calls"].append({
                    "name": payload.get("name", "unknown"),
                    "arguments": payload.get("input", ""),
                    "call_id": payload.get("call_id", ""),
                    "timestamp": timestamp,
                })

            elif item_type == "function_call_output":
                current_turn["tool_outputs"].append({
                    "call_id": payload.get("call_id", ""),
                    "output": payload.get("output", ""),
                    "timestamp": timestamp,
                })

            elif item_type == "custom_tool_call_output":
                current_turn["tool_outputs"].append({
                    "call_id": payload.get("call_id", ""),
                    "output": payload.get("output", ""),
                    "timestamp": timestamp,
                })

            elif item_type == "reasoning":
                current_turn["reasoning"] = True

            elif item_type == "web_search_call":
                action = payload.get("action", {})
                query = action.get("query", "")
                current_turn["tool_calls"].append({
                    "name": "web_search",
                    "arguments": json.dumps({"query": query}),
                    "call_id": f"web_search_{timestamp}",
                    "timestamp": timestamp,
                })

        if current_turn is not None:
            current_turn["events"].append(event)

    return turns


def create_respan_spans(
    session_id: str,
    turn_num: int,
    turn: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create Respan span logs for a single Codex CLI turn.

    Produces a span tree:
        Root: codex-cli (agent container, latency, metadata)
          +-- openai.chat (generation - model, tokens, messages)
          +-- Reasoning (if reasoning_output_tokens > 0)
          +-- Tool: Shell (if exec_command)
          +-- Tool: File Edit (if apply_patch)
          +-- Tool: Web Search (if web_search_call)
    """
    spans = []

    # Extract data from the turn
    user_text = turn.get("user_message", "")
    assistant_text = turn.get("assistant_message", "")
    commentary = turn.get("commentary", [])
    model = turn.get("model", "gpt-5.4")
    cwd = turn.get("cwd", "")
    token_usage = turn.get("token_usage", {})
    tool_calls = turn.get("tool_calls", [])
    tool_outputs = turn.get("tool_outputs", [])
    has_reasoning = turn.get("reasoning", False)

    # Timing
    start_time_str = turn.get("start_time", "")
    end_time_str = turn.get("end_time", "")
    now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not start_time_str:
        start_time_str = now_str
    if not end_time_str:
        end_time_str = now_str

    latency = None
    start_dt = parse_timestamp(start_time_str)
    end_dt = parse_timestamp(end_time_str)
    if start_dt and end_dt:
        latency = (end_dt - start_dt).total_seconds()

    # Messages for input/output
    prompt_messages: List[Dict[str, Any]] = []
    if user_text:
        prompt_messages.append({"role": "user", "content": user_text})
    completion_message: Optional[Dict[str, Any]] = None
    if assistant_text:
        completion_message = {"role": "assistant", "content": assistant_text}

    # IDs from config
    cfg_fields = (config or {}).get("fields", {})
    cfg_props = (config or {}).get("properties", {})

    trace_unique_id = f"{session_id}_turn_{turn_num}"
    workflow_name = os.environ.get("RESPAN_WORKFLOW_NAME") or cfg_fields.get("workflow_name") or "codex-cli"
    root_span_name = os.environ.get("RESPAN_SPAN_NAME") or cfg_fields.get("span_name") or "codex-cli"
    thread_id = f"codexcli_{session_id}"
    customer_id = os.environ.get("RESPAN_CUSTOMER_ID") or cfg_fields.get("customer_id") or ""

    # Metadata
    metadata: Dict[str, Any] = {
        "codex_cli_turn": turn_num,
    }
    if cwd:
        metadata["cwd"] = cwd
    if commentary:
        metadata["commentary"] = "\n".join(commentary)
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

    # Token usage mapping
    usage_fields: Dict[str, Any] = {}
    if token_usage:
        prompt_tokens = token_usage.get("input_tokens", 0)
        completion_tokens = token_usage.get("output_tokens", 0)
        usage_fields["prompt_tokens"] = prompt_tokens
        usage_fields["completion_tokens"] = completion_tokens
        total = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
        if total > 0:
            usage_fields["total_tokens"] = total
        cached = token_usage.get("cached_input_tokens", 0)
        if cached > 0:
            usage_fields["prompt_tokens_details"] = {"cached_tokens": cached}
        reasoning_tokens = token_usage.get("reasoning_output_tokens", 0)
        if reasoning_tokens > 0:
            metadata["reasoning_tokens"] = reasoning_tokens

    # ------------------------------------------------------------------
    # Root span - agent container
    # ------------------------------------------------------------------
    root_span_id = f"codexcli_{trace_unique_id}_root"
    root_span: Dict[str, Any] = {
        "trace_unique_id": trace_unique_id,
        "thread_identifier": thread_id,
        "customer_identifier": customer_id,
        "span_unique_id": root_span_id,
        "span_name": root_span_name,
        "span_workflow_name": workflow_name,
        "model": model,
        "provider_id": "",
        "span_path": "",
        "input": json.dumps(prompt_messages) if prompt_messages else "",
        "output": json.dumps(completion_message) if completion_message else "",
        "timestamp": end_time_str,
        "start_time": start_time_str,
        "metadata": metadata,
    }
    if latency is not None:
        root_span["latency"] = latency
    spans.append(root_span)

    # ------------------------------------------------------------------
    # LLM generation child span
    # ------------------------------------------------------------------
    gen_span_id = f"codexcli_{trace_unique_id}_gen"
    gen_span: Dict[str, Any] = {
        "trace_unique_id": trace_unique_id,
        "span_unique_id": gen_span_id,
        "span_parent_id": root_span_id,
        "span_name": "openai.chat",
        "span_workflow_name": workflow_name,
        "span_path": "openai_chat",
        "model": model,
        "provider_id": "openai",
        "metadata": {},
        "input": json.dumps(prompt_messages) if prompt_messages else "",
        "output": json.dumps(completion_message) if completion_message else "",
        "prompt_messages": prompt_messages,
        "completion_message": completion_message,
        "timestamp": end_time_str,
        "start_time": start_time_str,
    }
    if latency is not None:
        gen_span["latency"] = latency
    gen_span.update(usage_fields)
    spans.append(gen_span)

    # ------------------------------------------------------------------
    # Reasoning child span (if reasoning_output_tokens > 0)
    # ------------------------------------------------------------------
    reasoning_tokens = token_usage.get("reasoning_output_tokens", 0)
    if has_reasoning or reasoning_tokens > 0:
        spans.append({
            "trace_unique_id": trace_unique_id,
            "span_unique_id": f"codexcli_{trace_unique_id}_reasoning",
            "span_parent_id": root_span_id,
            "span_name": "Reasoning",
            "span_workflow_name": workflow_name,
            "span_path": "reasoning",
            "provider_id": "",
            "metadata": {"reasoning_tokens": reasoning_tokens} if reasoning_tokens > 0 else {},
            "input": "",
            "output": f"[Reasoning: {reasoning_tokens} tokens]" if reasoning_tokens > 0 else "[Reasoning]",
            "timestamp": end_time_str,
            "start_time": start_time_str,
        })

    # ------------------------------------------------------------------
    # Tool child spans
    # ------------------------------------------------------------------
    # Build output lookup by call_id
    output_map: Dict[str, Dict[str, Any]] = {}
    for to in tool_outputs:
        call_id = to.get("call_id", "")
        if call_id:
            output_map[call_id] = to

    tool_num = 0
    for tc in tool_calls:
        tool_num += 1
        tool_name = tc.get("name", "unknown")
        call_id = tc.get("call_id", "")
        arguments = tc.get("arguments", "")
        tool_ts = tc.get("timestamp", start_time_str)

        # Map Codex tool names to friendly display names
        display_name = _tool_display_name(tool_name)

        # Format input
        tool_input = _format_tool_input(tool_name, arguments)

        # Format output
        tool_output_data = output_map.get(call_id, {})
        tool_output = _format_tool_output(tool_output_data.get("output", ""))
        tool_end = tool_output_data.get("timestamp", end_time_str)

        # Calculate tool latency
        tool_latency = None
        tool_start_dt = parse_timestamp(tool_ts)
        tool_end_dt = parse_timestamp(tool_end)
        if tool_start_dt and tool_end_dt:
            tool_latency = (tool_end_dt - tool_start_dt).total_seconds()

        tool_span: Dict[str, Any] = {
            "trace_unique_id": trace_unique_id,
            "span_unique_id": f"codexcli_{trace_unique_id}_tool_{tool_num}",
            "span_parent_id": root_span_id,
            "span_name": f"Tool: {display_name}",
            "span_workflow_name": workflow_name,
            "span_path": f"tool_{tool_name.lower()}",
            "provider_id": "",
            "metadata": {},
            "input": tool_input,
            "output": tool_output,
            "timestamp": tool_end,
            "start_time": tool_ts,
        }
        if tool_latency is not None:
            tool_span["latency"] = tool_latency
        spans.append(tool_span)

    # Add required Respan platform fields to every span.
    respan_defaults = {
        "warnings": "",
        "encoding_format": "float",
        "disable_fallback": False,
        "respan_params": {
            "has_webhook": False,
            "environment": os.environ.get("RESPAN_ENVIRONMENT", "prod"),
        },
        "field_name": "data: ",
        "delimiter": "\n\n",
        "disable_log": False,
        "request_breakdown": False,
    }
    for span in spans:
        for key, value in respan_defaults.items():
            if key not in span:
                span[key] = value

    return spans


def _tool_display_name(name: str) -> str:
    """Map Codex CLI tool names to display names."""
    mapping = {
        "exec_command": "Shell",
        "apply_patch": "File Edit",
        "web_search": "Web Search",
    }
    return mapping.get(name, name)


def _format_tool_input(tool_name: str, arguments: str) -> str:
    """Format tool input for display."""
    if not arguments:
        return ""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return truncate(str(arguments))

    if tool_name == "exec_command" and isinstance(args, dict):
        cmd = args.get("cmd", "")
        workdir = args.get("workdir", "")
        result = f"Command: {cmd}"
        if workdir:
            result = f"[{workdir}] {result}"
        return truncate(result)

    if tool_name == "apply_patch" and isinstance(arguments, str):
        return truncate(arguments)

    if isinstance(args, dict):
        try:
            return truncate(json.dumps(args, indent=2))
        except (TypeError, ValueError):
            pass

    return truncate(str(arguments))


def _format_tool_output(output: str) -> str:
    """Format tool output for display."""
    if not output:
        return ""
    # Try to parse JSON output (custom_tool_call_output wraps in JSON)
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict) and "output" in parsed:
            return truncate(parsed["output"])
    except (json.JSONDecodeError, TypeError):
        pass
    return truncate(output)


def send_spans(
    spans: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    turn_num: int,
) -> None:
    """Send spans to Respan as a single batch."""
    url = f"{base_url}/v1/traces/ingest"
    headers = {"Authorization": f"Bearer {api_key}"}

    span_names = [s.get("span_name", "?") for s in spans]
    payload_json = json.dumps(spans)
    payload_size = len(payload_json)
    debug(f"Sending {len(spans)} spans ({payload_size} bytes) for turn {turn_num}: {span_names}")
    if DEBUG:
        debug_file = LOG_FILE.parent / f"respan_codex_spans_turn_{turn_num}.json"
        debug_file.write_text(payload_json, encoding="utf-8")
        debug(f"Dumped spans to {debug_file}")

    for attempt in range(3):
        try:
            response = requests.post(url, json=spans, headers=headers, timeout=30)
            if response.status_code < 400:
                resp_text = response.text[:300] if response.text else ""
                debug(f"Sent {len(spans)} spans for turn {turn_num} "
                      f"(attempt {attempt + 1}): {resp_text}")
                return
            if response.status_code < 500:
                log("ERROR", f"Spans rejected for turn {turn_num}: "
                    f"HTTP {response.status_code} - {response.text[:200]}")
                return
            debug(f"Server error for turn {turn_num} "
                  f"(attempt {attempt + 1}), retrying...")
            time.sleep(1.0)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(1.0)
        except Exception as e:
            log("ERROR", f"Failed to send spans for turn {turn_num}: {e}")
            return

    log("ERROR", f"Failed to send {len(spans)} spans for turn {turn_num} after 3 attempts")


def process_session(
    session_id: str,
    session_file: Path,
    state: Dict[str, Any],
    api_key: str,
    base_url: str,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Process a session JSONL file and create traces for new turns."""
    session_state = state.get(session_id, {})
    last_turn_count = session_state.get("turn_count", 0)

    # Read and parse the full session file
    lines = session_file.read_text(encoding="utf-8").strip().split("\n")
    events = parse_session(lines)

    if not events:
        debug("No events in session file")
        return 0

    # Extract all turns from the session
    all_turns = extract_turns(events)
    total_turns = len(all_turns)

    if total_turns <= last_turn_count:
        debug(f"No new turns (total: {total_turns}, processed: {last_turn_count})")
        return 0

    # Process only new turns
    new_turns = all_turns[last_turn_count:]
    turns_processed = 0

    for turn in new_turns:
        turns_processed += 1
        turn_num = last_turn_count + turns_processed
        spans = create_respan_spans(session_id, turn_num, turn, config=config)
        send_spans(spans, api_key, base_url, turn_num)

    # Update state
    state[session_id] = {
        "turn_count": last_turn_count + turns_processed,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)

    return turns_processed


def resolve_credentials() -> Tuple[Optional[str], str]:
    """Resolve API key and base URL from env vars or credentials file.

    Returns (api_key, base_url).
    """
    api_key = os.getenv("RESPAN_API_KEY")
    base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

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
                if not base_url or base_url == "https://api.respan.ai/api":
                    base_url = cred.get("baseUrl", base_url)
                if base_url and not base_url.rstrip("/").endswith("/api"):
                    base_url = base_url.rstrip("/") + "/api"
                if api_key:
                    debug(f"Using API key from credentials.json (profile: {profile})")
            except (json.JSONDecodeError, IOError) as e:
                debug(f"Failed to read credentials.json: {e}")

    return api_key, base_url


def main():
    script_start = datetime.now()
    debug("Codex hook started")

    # Parse notify payload from sys.argv[1]
    if len(sys.argv) < 2:
        debug("No argument provided (expected JSON payload in sys.argv[1])")
        sys.exit(0)

    try:
        payload = json.loads(sys.argv[1])
    except (json.JSONDecodeError, TypeError) as e:
        debug(f"Invalid JSON in sys.argv[1]: {e}")
        sys.exit(0)

    # Only process agent-turn-complete events
    event_type = payload.get("type", "")
    if event_type != "agent-turn-complete":
        debug(f"Ignoring event type: {event_type}")
        sys.exit(0)

    # Extract session info from the payload
    session_id = payload.get("thread-id", "")
    if not session_id:
        debug("No thread-id in notify payload")
        sys.exit(0)

    debug(f"Processing notify: type={event_type}, session={session_id}")

    # Resolve credentials
    api_key, base_url = resolve_credentials()
    if not api_key:
        log("ERROR", "No API key found. Run: respan auth login")
        sys.exit(0)

    # Find the session file
    session_file = find_session_file(session_id)
    if not session_file:
        # Fall back to latest session file
        result = find_latest_session_file()
        if result:
            session_id, session_file = result
        else:
            debug("No session file found")
            sys.exit(0)

    # Load respan.json config from the project directory
    config: Dict[str, Any] = {"fields": {}, "properties": {}}
    cwd = payload.get("cwd", "")
    if cwd:
        config = load_respan_config(cwd)
        debug(f"Loaded respan.json config from {cwd}: {config}")

    # Process the session with retry logic
    max_attempts = 3
    turns = 0
    try:
        for attempt in range(max_attempts):
            with state_lock():
                state = load_state()
                turns = process_session(
                    session_id, session_file, state, api_key, base_url, config=config
                )

            if turns > 0:
                break

            if attempt < max_attempts - 1:
                delay = 0.5 * (attempt + 1)
                debug(f"No turns processed (attempt {attempt + 1}/{max_attempts}), "
                      f"retrying in {delay}s...")
                time.sleep(delay)

        duration = (datetime.now() - script_start).total_seconds()
        log("INFO", f"Processed {turns} turns in {duration:.1f}s")

        if duration > 180:
            log("WARN", f"Hook took {duration:.1f}s (>3min), consider optimizing")

    except Exception as e:
        log("ERROR", f"Failed to process session: {e}")
        import traceback
        debug(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
