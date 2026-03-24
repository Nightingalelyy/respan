#!/usr/bin/env python3
"""
Respan Hook for Claude Code

Sends Claude Code conversation traces to Respan after each response.
Uses Claude Code's Stop hook to capture transcripts and convert them to Respan spans.

Usage:
    Copy this file to ~/.claude/hooks/respan_hook.py
    Configure in ~/.claude/settings.json (see .claude/settings.json.example)
    Enable per-project in .claude/settings.local.json (see .claude/settings.local.json.example)
"""

import contextlib
import json
import os
import sys
import tempfile
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # Not available on Windows

# Configuration
LOG_FILE = Path.home() / ".claude" / "state" / "respan_hook.log"
STATE_FILE = Path.home() / ".claude" / "state" / "respan_state.json"
LOCK_FILE = Path.home() / ".claude" / "state" / "respan_hook.lock"
DEBUG = os.environ.get("CC_RESPAN_DEBUG", "").lower() == "true"

try:
    MAX_CHARS = int(os.environ.get("CC_RESPAN_MAX_CHARS", "4000"))
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


# Known config keys in respan.json that map to span fields.
# Anything else is treated as a custom property (goes into metadata).
KNOWN_CONFIG_KEYS = {"customer_id", "span_name", "workflow_name"}


def load_respan_config(cwd: str) -> Dict[str, Any]:
    """Load .claude/respan.json from the project directory.

    Returns a dict with two keys:
      - "fields": known span fields (customer_id, span_name, workflow_name)
      - "properties": everything else (custom properties → metadata)
    """
    config_path = Path(cwd) / ".claude" / "respan.json"
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


def get_content(msg: Dict[str, Any]) -> Any:
    """Extract content from a message."""
    if isinstance(msg, dict):
        if "message" in msg:
            return msg["message"].get("content")
        return msg.get("content")
    return None


def is_tool_result(msg: Dict[str, Any]) -> bool:
    """Check if a message contains tool results."""
    content = get_content(msg)
    if isinstance(content, list):
        return any(
            isinstance(item, dict) and item.get("type") == "tool_result"
            for item in content
        )
    return False


def get_tool_calls(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool use blocks from a message."""
    content = get_content(msg)
    if isinstance(content, list):
        return [
            item for item in content
            if isinstance(item, dict) and item.get("type") == "tool_use"
        ]
    return []


def get_text_content(msg: Dict[str, Any]) -> str:
    """Extract text content from a message."""
    content = get_content(msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    return ""


def format_tool_input(tool_name: str, tool_input: Any, max_length: int = MAX_CHARS) -> str:
    """Format tool input for better readability."""
    if not tool_input:
        return ""
    
    # Handle Write/Edit tool - show file path and content preview
    if tool_name in ("Write", "Edit", "MultiEdit"):
        if isinstance(tool_input, dict):
            file_path = tool_input.get("file_path", tool_input.get("path", ""))
            content = tool_input.get("content", "")
            
            result = f"File: {file_path}\n"
            if content:
                content_preview = content[:2000] + "..." if len(content) > 2000 else content
                result += f"Content:\n{content_preview}"
            return result[:max_length]
    
    # Handle Read tool
    if tool_name == "Read":
        if isinstance(tool_input, dict):
            file_path = tool_input.get("file_path", tool_input.get("path", ""))
            return f"File: {file_path}"
    
    # Handle Bash/Shell tool
    if tool_name in ("Bash", "Shell"):
        if isinstance(tool_input, dict):
            command = tool_input.get("command", "")
            return f"Command: {command}"
    
    # Default: JSON dump with truncation
    try:
        result = json.dumps(tool_input, indent=2)
        if len(result) > max_length:
            result = result[:max_length] + "\n... (truncated)"
        return result
    except (TypeError, ValueError):
        return str(tool_input)[:max_length]


def format_tool_output(tool_name: str, tool_output: Any, max_length: int = MAX_CHARS) -> str:
    """Format tool output for better readability."""
    if not tool_output:
        return ""
    
    # Handle string output directly
    if isinstance(tool_output, str):
        if len(tool_output) > max_length:
            return tool_output[:max_length] + "\n... (truncated)"
        return tool_output
    
    # Handle list of content blocks (common in Claude Code tool results)
    if isinstance(tool_output, list):
        parts = []
        total_length = 0
        
        for item in tool_output:
            if isinstance(item, dict):
                # Text content block
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if total_length + len(text) > max_length:
                        remaining = max_length - total_length
                        if remaining > 100:
                            parts.append(text[:remaining] + "... (truncated)")
                        break
                    parts.append(text)
                    total_length += len(text)
                # Image or other type
                elif item.get("type") == "image":
                    parts.append("[Image output]")
                else:
                    # Try to extract any text-like content
                    text = str(item)[:500]
                    parts.append(text)
                    total_length += len(text)
            elif isinstance(item, str):
                if total_length + len(item) > max_length:
                    remaining = max_length - total_length
                    if remaining > 100:
                        parts.append(item[:remaining] + "... (truncated)")
                    break
                parts.append(item)
                total_length += len(item)
        
        return "\n".join(parts)
    
    # Handle dict output
    if isinstance(tool_output, dict):
        # Special handling for Write tool success/error
        if "success" in tool_output:
            return f"Success: {tool_output.get('success')}\n{tool_output.get('message', '')}"
        
        # Default JSON formatting
        try:
            result = json.dumps(tool_output, indent=2)
            if len(result) > max_length:
                result = result[:max_length] + "\n... (truncated)"
            return result
        except (TypeError, ValueError):
            return str(tool_output)[:max_length]
    
    return str(tool_output)[:max_length]


def merge_assistant_parts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple assistant message parts into one."""
    if not parts:
        return {}
    
    merged_content = []
    for part in parts:
        content = get_content(part)
        if isinstance(content, list):
            merged_content.extend(content)
        elif content:
            merged_content.append({"type": "text", "text": str(content)})
    
    # Use the structure from the first part
    result = parts[0].copy()
    if "message" in result:
        result["message"] = result["message"].copy()
        result["message"]["content"] = merged_content
    else:
        result["content"] = merged_content
    
    return result


def find_latest_transcript() -> Optional[Tuple[str, Path]]:
    """Find the most recently modified transcript file.
    
    Claude Code stores transcripts as *.jsonl files directly in the project directory.
    Main conversation files have UUID names, agent files have agent-*.jsonl names.
    The session ID is stored inside each JSON line.
    """
    projects_dir = Path.home() / ".claude" / "projects"
    
    if not projects_dir.exists():
        debug(f"Projects directory not found: {projects_dir}")
        return None
    
    latest_file = None
    latest_mtime = 0
    
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        
        # Look for all .jsonl files directly in the project directory
        for transcript_file in project_dir.glob("*.jsonl"):
            mtime = transcript_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = transcript_file
    
    if latest_file:
        # Extract session ID from the first line of the file
        try:
            first_line = latest_file.read_text(encoding="utf-8").split("\n")[0]
            if first_line:
                first_msg = json.loads(first_line)
                session_id = first_msg.get("sessionId", latest_file.stem)
                debug(f"Found transcript: {latest_file}, session: {session_id}")
                return (session_id, latest_file)
        except (json.JSONDecodeError, IOError, IndexError, UnicodeDecodeError) as e:
            debug(f"Error reading transcript {latest_file}: {e}")
            return None
    
    debug("No transcript files found")
    return None


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    try:
        # Handle both with and without timezone
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def create_respan_spans(
    session_id: str,
    turn_num: int,
    user_msg: Dict[str, Any],
    assistant_msgs: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create Respan span logs for a single turn with all available metadata.

    Produces a proper span tree so that the Respan UI renders nested children:
        Root (agent container)
          ├── claude.chat  (generation – carries model, tokens, messages)
          ├── Thinking 1   (generation, if extended thinking is present)
          ├── Tool: Read   (tool, if tool use occurred)
          └── Tool: Write  (tool, if tool use occurred)
    """
    spans = []

    # ------------------------------------------------------------------
    # 1. Extract data from the transcript messages
    # ------------------------------------------------------------------
    user_text = get_text_content(user_msg)
    user_timestamp = user_msg.get("timestamp")
    user_time = parse_timestamp(user_timestamp) if user_timestamp else None

    # Collect assistant text across all messages in the turn
    final_output = ""
    if assistant_msgs:
        text_parts = [get_text_content(m) for m in assistant_msgs]
        final_output = "\n".join(p for p in text_parts if p)

    # Aggregate model, usage, timing from (possibly multiple) API calls
    model = "claude"
    usage = None
    request_id = None
    stop_reason = None
    first_assistant_timestamp = None
    last_assistant_timestamp = None
    last_assistant_time = None

    for a_msg in assistant_msgs:
        if not (isinstance(a_msg, dict) and "message" in a_msg):
            continue
        msg_obj = a_msg["message"]
        model = msg_obj.get("model", model)
        request_id = a_msg.get("requestId", request_id)
        stop_reason = msg_obj.get("stop_reason") or stop_reason
        ts = a_msg.get("timestamp")
        if ts:
            if first_assistant_timestamp is None:
                first_assistant_timestamp = ts
            last_assistant_timestamp = ts
            last_assistant_time = parse_timestamp(ts)

        msg_usage = msg_obj.get("usage")
        if msg_usage:
            if usage is None:
                usage = dict(msg_usage)
            else:
                for key in ("input_tokens", "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens"):
                    if key in msg_usage:
                        usage[key] = usage.get(key, 0) + msg_usage[key]
                if "service_tier" in msg_usage:
                    usage["service_tier"] = msg_usage["service_tier"]

    # Timing
    now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    start_time_str = user_timestamp or first_assistant_timestamp or now_str
    timestamp_str = last_assistant_timestamp or first_assistant_timestamp or now_str

    latency = None
    if user_time and last_assistant_time:
        latency = (last_assistant_time - user_time).total_seconds()

    # Messages
    prompt_messages: List[Dict[str, Any]] = []
    if user_text:
        prompt_messages.append({"role": "user", "content": user_text})
    completion_message: Optional[Dict[str, Any]] = None
    if final_output:
        completion_message = {"role": "assistant", "content": final_output}

    # IDs — respan.json fields, then env var overrides
    cfg_fields = (config or {}).get("fields", {})
    cfg_props = (config or {}).get("properties", {})

    trace_unique_id = f"{session_id}_turn_{turn_num}"
    workflow_name = os.environ.get("RESPAN_WORKFLOW_NAME") or cfg_fields.get("workflow_name") or "claude-code"
    root_span_name = os.environ.get("RESPAN_SPAN_NAME") or cfg_fields.get("span_name") or "claude-code"
    thread_id = f"claudecode_{session_id}"
    customer_id = os.environ.get("RESPAN_CUSTOMER_ID") or cfg_fields.get("customer_id") or ""

    # Metadata — custom properties from respan.json, then env overrides
    metadata: Dict[str, Any] = {"claude_code_turn": turn_num}
    if cfg_props:
        metadata.update(cfg_props)
    if request_id:
        metadata["request_id"] = request_id
    if stop_reason:
        metadata["stop_reason"] = stop_reason
    env_metadata = os.environ.get("RESPAN_METADATA")
    if env_metadata:
        try:
            extra = json.loads(env_metadata)
            if isinstance(extra, dict):
                metadata.update(extra)
        except json.JSONDecodeError:
            pass

    # Usage
    usage_fields: Dict[str, Any] = {}
    if usage:
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        usage_fields["prompt_tokens"] = prompt_tokens
        usage_fields["completion_tokens"] = completion_tokens
        total = prompt_tokens + completion_tokens
        if total > 0:
            usage_fields["total_tokens"] = total
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        if cache_creation > 0:
            usage_fields["cache_creation_prompt_tokens"] = cache_creation
        prompt_tokens_details: Dict[str, int] = {}
        if cache_creation > 0:
            prompt_tokens_details["cache_creation_tokens"] = cache_creation
        if cache_read > 0:
            prompt_tokens_details["cached_tokens"] = cache_read
        if prompt_tokens_details:
            usage_fields["prompt_tokens_details"] = prompt_tokens_details
        service_tier = usage.get("service_tier")
        if service_tier:
            metadata["service_tier"] = service_tier

    # ------------------------------------------------------------------
    # 2. Root span – pure agent container (no model / token info)
    # ------------------------------------------------------------------
    root_span_id = f"claudecode_{trace_unique_id}_root"
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
        "timestamp": timestamp_str,
        "start_time": start_time_str,
        "metadata": metadata,
    }
    if latency is not None:
        root_span["latency"] = latency
    spans.append(root_span)

    # ------------------------------------------------------------------
    # 3. LLM generation child span (always created → every turn has ≥1 child)
    # ------------------------------------------------------------------
    gen_span_id = f"claudecode_{trace_unique_id}_gen"
    gen_start = first_assistant_timestamp or start_time_str
    gen_end = last_assistant_timestamp or timestamp_str
    gen_latency = None
    gen_start_dt = parse_timestamp(gen_start) if gen_start else None
    gen_end_dt = parse_timestamp(gen_end) if gen_end else None
    if gen_start_dt and gen_end_dt:
        gen_latency = (gen_end_dt - gen_start_dt).total_seconds()

    gen_span: Dict[str, Any] = {
        "trace_unique_id": trace_unique_id,
        "span_unique_id": gen_span_id,
        "span_parent_id": root_span_id,
        "span_name": "claude.chat",
        "span_workflow_name": workflow_name,
        "span_path": "claude_chat",
        "model": model,
        "provider_id": "anthropic",
        "metadata": {},
        "input": json.dumps(prompt_messages) if prompt_messages else "",
        "output": json.dumps(completion_message) if completion_message else "",
        "prompt_messages": prompt_messages,
        "completion_message": completion_message,
        "timestamp": gen_end,
        "start_time": gen_start,
    }
    if gen_latency is not None:
        gen_span["latency"] = gen_latency
    gen_span.update(usage_fields)
    spans.append(gen_span)

    # ------------------------------------------------------------------
    # 4. Thinking child spans
    # ------------------------------------------------------------------
    thinking_num = 0
    for assistant_msg in assistant_msgs:
        if not (isinstance(assistant_msg, dict) and "message" in assistant_msg):
            continue
        content = assistant_msg["message"].get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "thinking":
                thinking_text = item.get("thinking", "")
                if not thinking_text:
                    continue
                thinking_num += 1
                thinking_ts = assistant_msg.get("timestamp", timestamp_str)
                spans.append({
                    "trace_unique_id": trace_unique_id,
                    "span_unique_id": f"claudecode_{trace_unique_id}_thinking_{thinking_num}",
                    "span_parent_id": root_span_id,
                    "span_name": f"Thinking {thinking_num}",
                    "span_workflow_name": workflow_name,
                    "span_path": "thinking",
                    "provider_id": "",
                    "metadata": {},
                    "input": "",
                    "output": thinking_text,
                    "timestamp": thinking_ts,
                    "start_time": thinking_ts,
                })

    # ------------------------------------------------------------------
    # 5. Tool child spans
    # ------------------------------------------------------------------
    tool_call_map: Dict[str, Dict[str, Any]] = {}
    for assistant_msg in assistant_msgs:
        for tool_call in get_tool_calls(assistant_msg):
            tool_id = tool_call.get("id", "")
            tool_call_map[tool_id] = {
                "name": tool_call.get("name", "unknown"),
                "input": tool_call.get("input", {}),
                "id": tool_id,
                "timestamp": assistant_msg.get("timestamp") if isinstance(assistant_msg, dict) else None,
            }

    for tr in tool_results:
        tr_content = get_content(tr)
        tool_result_metadata: Dict[str, Any] = {}
        if isinstance(tr, dict):
            tur = tr.get("toolUseResult") or {}
            for src, dst in [("durationMs", "duration_ms"), ("numFiles", "num_files"),
                             ("filenames", "filenames"), ("truncated", "truncated")]:
                if src in tur:
                    tool_result_metadata[dst] = tur[src]
        if isinstance(tr_content, list):
            for item in tr_content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    tool_use_id = item.get("tool_use_id")
                    if tool_use_id in tool_call_map:
                        tool_call_map[tool_use_id]["output"] = item.get("content")
                        tool_call_map[tool_use_id]["result_metadata"] = tool_result_metadata
                        tool_call_map[tool_use_id]["result_timestamp"] = tr.get("timestamp")

    tool_num = 0
    for tool_id, td in tool_call_map.items():
        tool_num += 1
        tool_ts = td.get("result_timestamp") or td.get("timestamp") or timestamp_str
        tool_start = td.get("timestamp") or start_time_str
        tool_span: Dict[str, Any] = {
            "trace_unique_id": trace_unique_id,
            "span_unique_id": f"claudecode_{trace_unique_id}_tool_{tool_num}",
            "span_parent_id": root_span_id,
            "span_name": f"Tool: {td['name']}",
            "span_workflow_name": workflow_name,
            "span_path": f"tool_{td['name'].lower()}",
            "provider_id": "",
            "metadata": td.get("result_metadata") or {},
            "input": format_tool_input(td["name"], td["input"]),
            "output": format_tool_output(td["name"], td.get("output")),
            "timestamp": tool_ts,
            "start_time": tool_start,
        }
        if td.get("result_metadata"):
            duration_ms = td["result_metadata"].get("duration_ms")
            if duration_ms:
                tool_span["latency"] = duration_ms / 1000.0
        spans.append(tool_span)

    # Add required Respan platform fields to every span.
    # The backend expects these on all spans (per official SDK examples).
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


def send_spans(
    spans: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    turn_num: int,
) -> None:
    """Send spans to Respan as a single batch (matches official SDK behaviour).

    The official Respan tracing SDK sends all spans for a trace in one
    POST request to ``/v1/traces/ingest``.  We do the same here, with
    simple retry logic for transient server errors.
    """
    url = f"{base_url}/v1/traces/ingest"
    headers = {"Authorization": f"Bearer {api_key}"}

    span_names = [s.get("span_name", "?") for s in spans]
    payload_json = json.dumps(spans)
    payload_size = len(payload_json)
    debug(f"Sending {len(spans)} spans ({payload_size} bytes) for turn {turn_num}: {span_names}")
    if DEBUG:
        debug_file = LOG_FILE.parent / f"respan_spans_turn_{turn_num}.json"
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
            # 5xx — retry after short delay
            debug(f"Server error for turn {turn_num} "
                  f"(attempt {attempt + 1}), retrying...")
            time.sleep(1.0)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(1.0)
        except Exception as e:
            log("ERROR", f"Failed to send spans for turn {turn_num}: {e}")
            return

    log("ERROR", f"Failed to send {len(spans)} spans for turn {turn_num} "
        f"after 3 attempts")


def process_transcript(
    session_id: str,
    transcript_file: Path,
    state: Dict[str, Any],
    api_key: str,
    base_url: str,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Process a transcript file and create traces for new turns."""
    # Get previous state for this session
    session_state = state.get(session_id, {})
    last_line = session_state.get("last_line", 0)
    turn_count = session_state.get("turn_count", 0)
    
    # Read transcript - need ALL messages to build conversation history
    lines = transcript_file.read_text(encoding="utf-8").strip().split("\n")
    total_lines = len(lines)
    
    if last_line >= total_lines:
        debug(f"No new lines to process (last: {last_line}, total: {total_lines})")
        return 0
    
    # Parse new messages, tracking their line indices
    new_messages = []
    for i in range(last_line, total_lines):
        try:
            if lines[i].strip():
                msg = json.loads(lines[i])
                msg["_line_idx"] = i
                new_messages.append(msg)
        except json.JSONDecodeError:
            continue

    if not new_messages:
        return 0

    debug(f"Processing {len(new_messages)} new messages")

    # Group messages into turns (user -> assistant(s) -> tool_results)
    turns_processed = 0
    # Track the line after the last fully-processed turn so we can
    # re-read incomplete turns on the next invocation.
    last_committed_line = last_line
    current_user = None
    current_user_line = last_line
    current_assistants = []
    current_assistant_parts = []
    current_msg_id = None
    current_tool_results = []

    def _commit_turn():
        """Send the current turn and update last_committed_line."""
        nonlocal turns_processed, last_committed_line
        turns_processed += 1
        turn_num = turn_count + turns_processed
        spans = create_respan_spans(
            session_id, turn_num, current_user, current_assistants, current_tool_results,
            config=config,
        )
        send_spans(spans, api_key, base_url, turn_num)
        last_committed_line = total_lines  # safe default, refined below

    for msg in new_messages:
        line_idx = msg.pop("_line_idx", last_line)
        role = msg.get("type") or (msg.get("message", {}).get("role"))

        if role == "user":
            # Check if this is a tool result
            if is_tool_result(msg):
                current_tool_results.append(msg)
                continue

            # New user message - finalize previous turn
            if current_msg_id and current_assistant_parts:
                merged = merge_assistant_parts(current_assistant_parts)
                current_assistants.append(merged)
                current_assistant_parts = []
                current_msg_id = None

            if current_user and current_assistants:
                _commit_turn()
                # Advance committed line to just before this new user msg
                last_committed_line = line_idx

            # Start new turn
            current_user = msg
            current_user_line = line_idx
            current_assistants = []
            current_assistant_parts = []
            current_msg_id = None
            current_tool_results = []

        elif role == "assistant":
            msg_id = None
            if isinstance(msg, dict) and "message" in msg:
                msg_id = msg["message"].get("id")

            if not msg_id:
                # No message ID, treat as continuation
                current_assistant_parts.append(msg)
            elif msg_id == current_msg_id:
                # Same message ID, add to current parts
                current_assistant_parts.append(msg)
            else:
                # New message ID - finalize previous message
                if current_msg_id and current_assistant_parts:
                    merged = merge_assistant_parts(current_assistant_parts)
                    current_assistants.append(merged)

                # Start new assistant message
                current_msg_id = msg_id
                current_assistant_parts = [msg]

    # Process final turn
    if current_msg_id and current_assistant_parts:
        merged = merge_assistant_parts(current_assistant_parts)
        current_assistants.append(merged)

    if current_user and current_assistants:
        # Check if the turn has actual text output.  The Stop hook can fire
        # before the final assistant text block is flushed to disk, leaving
        # only thinking/tool_use blocks.  If no text content is found, treat
        # the turn as incomplete so the retry logic re-reads it.
        has_text = any(get_text_content(m) for m in current_assistants)
        if has_text:
            _commit_turn()
            last_committed_line = total_lines
        else:
            last_committed_line = current_user_line
            debug(f"Turn has assistant msgs but no text output yet (likely not flushed), will retry")
    else:
        # Incomplete turn — rewind so the next run re-reads from the
        # unmatched user message (or from where we left off if no user).
        if current_user:
            last_committed_line = current_user_line
            debug(f"Incomplete turn at line {current_user_line}, will retry next run")
        # else: no pending user, advance past non-turn lines
        elif last_committed_line == last_line:
            last_committed_line = total_lines

    # Update state
    state[session_id] = {
        "last_line": last_committed_line,
        "turn_count": turn_count + turns_processed,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)
    
    return turns_processed


def read_stdin_payload() -> Optional[Tuple[str, Path]]:
    """Read session_id and transcript_path from stdin JSON payload.

    Claude Code hooks pipe a JSON object on stdin with at least
    ``session_id`` and ``transcript_path``.  Returns ``None`` when
    stdin is a TTY, empty, or contains invalid data.
    """
    if sys.stdin.isatty():
        debug("stdin is a TTY, skipping stdin payload")
        return None

    try:
        raw = sys.stdin.read()
    except Exception as e:
        debug(f"Failed to read stdin: {e}")
        return None

    if not raw or not raw.strip():
        debug("stdin is empty")
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        debug(f"Invalid JSON on stdin: {e}")
        return None

    session_id = payload.get("session_id")
    transcript_path_str = payload.get("transcript_path")
    if not session_id or not transcript_path_str:
        debug("stdin payload missing session_id or transcript_path")
        return None

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        debug(f"transcript_path from stdin does not exist: {transcript_path}")
        return None

    debug(f"Got transcript from stdin: session={session_id}, path={transcript_path}")
    return (session_id, transcript_path)


@contextlib.contextmanager
def state_lock(timeout: float = 5.0):
    """Acquire an advisory file lock around state operations.

    Falls back to no-lock when fcntl is unavailable (Windows) or on errors.
    """
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
                    yield
                    return
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    except Exception as e:
        debug(f"Lock error, proceeding without lock: {e}")
        if lock_fd is not None:
            with contextlib.suppress(Exception):
                lock_fd.close()
        yield


def main():
    script_start = datetime.now()
    debug("Hook started")

    # Check if tracing is enabled
    if os.environ.get("TRACE_TO_RESPAN", "").lower() != "true":
        debug("Tracing disabled (TRACE_TO_RESPAN != true)")
        sys.exit(0)

    # Resolve API key: env var > ~/.respan/credentials.json
    api_key = os.getenv("RESPAN_API_KEY")
    base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

    if not api_key:
        creds_file = Path.home() / ".respan" / "credentials.json"
        if creds_file.exists():
            try:
                creds = json.loads(creds_file.read_text(encoding="utf-8"))
                # Find the active profile's credential
                config_file = Path.home() / ".respan" / "config.json"
                profile = "default"
                if config_file.exists():
                    cfg = json.loads(config_file.read_text(encoding="utf-8"))
                    profile = cfg.get("activeProfile", "default")
                cred = creds.get(profile, {})
                api_key = cred.get("apiKey") or cred.get("accessToken")
                if not base_url or base_url == "https://api.respan.ai/api":
                    base_url = cred.get("baseUrl", base_url)
                # Ensure base_url ends with /api (credentials store the host only)
                if base_url and not base_url.rstrip("/").endswith("/api"):
                    base_url = base_url.rstrip("/") + "/api"
                if api_key:
                    debug(f"Using API key from credentials.json (profile: {profile})")
            except (json.JSONDecodeError, IOError) as e:
                debug(f"Failed to read credentials.json: {e}")

    if not api_key:
        log("ERROR", "No API key found. Run: respan auth login")
        sys.exit(0)

    # Try stdin payload first, fall back to filesystem scan
    result = read_stdin_payload()
    if not result:
        result = find_latest_transcript()
    if not result:
        debug("No transcript file found")
        sys.exit(0)

    session_id, transcript_file = result

    if not transcript_file:
        debug("No transcript file found")
        sys.exit(0)

    debug(f"Processing session: {session_id}")

    # Load respan.json config from the project directory.
    # Extract the project CWD from the first user message in the transcript.
    config: Dict[str, Any] = {"fields": {}, "properties": {}}
    try:
        first_line = transcript_file.read_text(encoding="utf-8").split("\n")[0]
        if first_line:
            first_msg = json.loads(first_line)
            cwd = first_msg.get("cwd")
            if not cwd:
                # Try second line (first is often file-history-snapshot)
                lines = transcript_file.read_text(encoding="utf-8").split("\n")
                for line in lines[:5]:
                    if line.strip():
                        msg = json.loads(line)
                        cwd = msg.get("cwd")
                        if cwd:
                            break
            if cwd:
                config = load_respan_config(cwd)
                debug(f"Loaded respan.json config from {cwd}: {config}")
    except Exception as e:
        debug(f"Failed to extract CWD or load config: {e}")

    # Process the transcript under file lock.
    # Retry up to 3 times with a short delay — the Stop hook can fire
    # before Claude Code finishes flushing the assistant response to
    # the transcript file, causing an incomplete turn on the first read.
    max_attempts = 3
    turns = 0
    try:
        for attempt in range(max_attempts):
            with state_lock():
                state = load_state()
                turns = process_transcript(session_id, transcript_file, state, api_key, base_url, config=config)

            if turns > 0:
                break

            if attempt < max_attempts - 1:
                delay = 0.5 * (attempt + 1)
                debug(f"No turns processed (attempt {attempt + 1}/{max_attempts}), "
                      f"retrying in {delay}s...")
                time.sleep(delay)

        # Log execution time
        duration = (datetime.now() - script_start).total_seconds()
        log("INFO", f"Processed {turns} turns in {duration:.1f}s")

        if duration > 180:
            log("WARN", f"Hook took {duration:.1f}s (>3min), consider optimizing")

    except Exception as e:
        log("ERROR", f"Failed to process transcript: {e}")
        import traceback
        debug(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
