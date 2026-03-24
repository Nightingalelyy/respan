/**
 * Respan Hook for Gemini CLI
 *
 * Handles streaming: Gemini fires AfterModel per chunk. We accumulate text
 * and only send on the final chunk (text+STOP or empty after accumulated text).
 *
 * Handles tool calls: model calls a tool → turn ends with STOP → Gemini
 * executes tool → new model turn. Detected via message count changes.
 * BeforeTool/AfterTool hooks capture tool names, args, and output.
 *
 * Span tree per turn:
 *   Root: gemini-cli
 *     ├── gemini.chat    (generation — model, tokens, messages)
 *     ├── Reasoning       (if thinking tokens > 0)
 *     ├── Tool: Shell     (if run_shell_command)
 *     └── Tool: File Read (if read_file)
 */
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { execFile } from 'node:child_process';

import {
  initLogging,
  log,
  debug,
  resolveCredentials,
  loadRespanConfig,
  acquireLock,
  addDefaultsToAll,
  resolveSpanFields,
  buildMetadata,
  resolveTracingIngestEndpoint,
  toOtlpPayload,
  nowISO,
  latencySeconds,
  truncate,
  type SpanData,
  type RespanConfig,
} from './shared.js';

// ── Config ────────────────────────────────────────────────────────

const STATE_DIR = path.join(os.homedir(), '.gemini', 'state');
const LOG_FILE = path.join(STATE_DIR, 'respan_hook.log');
const LOCK_PATH = path.join(STATE_DIR, 'respan_hook.lock');
const DEBUG_MODE = (process.env.GEMINI_RESPAN_DEBUG ?? '').toLowerCase() === 'true';
const MAX_CHARS = parseInt(process.env.GEMINI_RESPAN_MAX_CHARS ?? '4000', 10) || 4000;
const SEND_DELAY = parseInt(process.env.GEMINI_RESPAN_SEND_DELAY ?? '10', 10) || 10;

initLogging(LOG_FILE, DEBUG_MODE);

// ── Types ─────────────────────────────────────────────────────────

type Msg = Record<string, unknown>;

interface StreamState {
  accumulated_text: string;
  last_tokens: number;
  first_chunk_time: string;
  msg_count?: number;
  tool_turns?: number;
  send_version?: number;
  tool_details?: ToolDetail[];
  pending_tools?: ToolDetail[];
  thoughts_tokens?: number;
  last_send_text_len?: number;
}

interface ToolDetail {
  name: string;
  input?: unknown;
  args?: unknown;
  output?: string;
  start_time?: string;
  end_time?: string;
  error?: string;
}

// ── Tool display names ────────────────────────────────────────────

const TOOL_DISPLAY_NAMES: Record<string, string> = {
  read_file: 'File Read',
  read_many_files: 'File Read',
  write_file: 'File Write',
  list_directory: 'Directory List',
  run_shell_command: 'Shell',
  google_web_search: 'Web Search',
  web_fetch: 'Web Fetch',
  glob: 'Find Files',
  grep_search: 'Search Text',
  search_file_content: 'Search Text',
  replace: 'File Edit',
  save_memory: 'Memory',
  write_todos: 'Todos',
  get_internal_docs: 'Docs',
};

function toolDisplayName(name: string): string {
  return TOOL_DISPLAY_NAMES[name] ?? (name || 'Unknown');
}

function formatToolInput(toolName: string, args: unknown): string {
  if (!args) return '';
  const a = args as Msg;
  if (toolName === 'run_shell_command' && typeof a === 'object') {
    const cmd = String(a.command ?? '');
    const dir = String(a.dir_path ?? '');
    return truncate(dir ? `[${dir}] Command: ${cmd}` : `Command: ${cmd}`, MAX_CHARS);
  }
  if (['read_file', 'write_file'].includes(toolName) && typeof a === 'object')
    return truncate(String(a.file_path ?? JSON.stringify(a)), MAX_CHARS);
  if (toolName === 'read_many_files' && typeof a === 'object')
    return truncate(String(a.include ?? JSON.stringify(a)), MAX_CHARS);
  if (toolName === 'google_web_search' && typeof a === 'object')
    return truncate(`Query: ${a.query ?? String(a)}`, MAX_CHARS);
  if (['glob', 'grep_search', 'search_file_content'].includes(toolName) && typeof a === 'object')
    return truncate(String(a.pattern ?? JSON.stringify(a)), MAX_CHARS);
  if (toolName === 'replace' && typeof a === 'object') {
    const fp = String(a.file_path ?? '');
    const old = String(a.old_string ?? '');
    if (fp && old) return truncate(`${fp}: ${JSON.stringify(old)} → ...`, MAX_CHARS);
  }
  try { return truncate(JSON.stringify(args, null, 2), MAX_CHARS); } catch {}
  return truncate(String(args), MAX_CHARS);
}

// ── Stream state management ───────────────────────────────────────

function statePath(sessionId: string): string {
  const safeId = sessionId.replace(/[/\\]/g, '_').slice(0, 64);
  return path.join(STATE_DIR, `respan_stream_${safeId}.json`);
}

function loadStreamState(sessionId: string): StreamState {
  const p = statePath(sessionId);
  if (fs.existsSync(p)) {
    try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch {}
  }
  return { accumulated_text: '', last_tokens: 0, first_chunk_time: '' };
}

function saveStreamState(sessionId: string, state: StreamState): void {
  const p = statePath(sessionId);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  const tmp = p + '.tmp.' + process.pid;
  try {
    fs.writeFileSync(tmp, JSON.stringify(state));
    fs.renameSync(tmp, p);
  } catch {
    try { fs.unlinkSync(tmp); } catch {}
    fs.writeFileSync(p, JSON.stringify(state));
  }
}

function clearStreamState(sessionId: string): void {
  try { fs.unlinkSync(statePath(sessionId)); } catch {}
}

// ── Message extraction ────────────────────────────────────────────

function extractMessages(hookData: Msg): Msg[] {
  const llmReq = (hookData.llm_request ?? {}) as Msg;
  const messages = (llmReq.messages ?? []) as Msg[];
  return messages.map((msg) => ({
    role: String(msg.role ?? 'user') === 'model' ? 'assistant' : String(msg.role ?? 'user'),
    content: truncate(String(msg.content ?? ''), MAX_CHARS),
  }));
}

function detectModel(hookData: Msg): string {
  const override = process.env.RESPAN_GEMINI_MODEL;
  if (override) return override;
  const llmReq = (hookData.llm_request ?? {}) as Msg;
  return String(llmReq.model ?? '') || 'gemini-cli';
}

// ── Span construction ─────────────────────────────────────────────

function buildSpans(
  hookData: Msg,
  outputText: string,
  tokens: { prompt_tokens: number; completion_tokens: number; total_tokens: number },
  config: RespanConfig | null,
  startTimeIso: string | undefined,
  toolTurns: number,
  toolDetails: ToolDetail[],
  thoughtsTokens: number,
): SpanData[] {
  const spans: SpanData[] = [];
  const sessionId = String(hookData.session_id ?? '');
  const model = detectModel(hookData);
  const now = nowISO();
  const endTime = String(hookData.timestamp ?? '') || now;
  const beginTime = startTimeIso || endTime;
  const lat = latencySeconds(beginTime, endTime);

  const promptMessages = extractMessages(hookData);
  const completionMessage = { role: 'assistant', content: truncate(outputText, MAX_CHARS) };

  const { workflowName, spanName, customerId } = resolveSpanFields(config, {
    workflowName: 'gemini-cli',
    spanName: 'gemini-cli',
  });

  const safeId = sessionId.replace(/[/\\]/g, '_').slice(0, 50);
  const traceUniqueId = `gcli_${safeId}`;
  const rootSpanId = `gcli_${safeId}_root`;
  const threadId = `gcli_${sessionId}`;

  // LLM config
  const llmReq = (hookData.llm_request ?? {}) as Msg;
  const reqConfig = (llmReq.config ?? {}) as Msg;

  // Metadata
  const baseMeta: Record<string, unknown> = { source: 'gemini-cli' };
  if (toolTurns > 0) baseMeta.tool_turns = toolTurns;
  if (thoughtsTokens > 0) baseMeta.reasoning_tokens = thoughtsTokens;
  const metadata = buildMetadata(config, baseMeta);

  // Root span
  spans.push({
    trace_unique_id: traceUniqueId,
    thread_identifier: threadId,
    customer_identifier: customerId,
    span_unique_id: rootSpanId,
    span_name: spanName,
    span_workflow_name: workflowName,
    model,
    provider_id: '',
    span_path: '',
    input: promptMessages.length ? JSON.stringify(promptMessages) : '',
    output: JSON.stringify(completionMessage),
    timestamp: endTime,
    start_time: beginTime,
    metadata,
    ...(lat !== undefined ? { latency: lat } : {}),
  });

  // Generation child span
  const genSpan: SpanData = {
    trace_unique_id: traceUniqueId,
    span_unique_id: `gcli_${safeId}_gen`,
    span_parent_id: rootSpanId,
    span_name: 'gemini.chat',
    span_workflow_name: workflowName,
    span_path: 'gemini_chat',
    model,
    provider_id: 'google',
    metadata: {},
    input: promptMessages.length ? JSON.stringify(promptMessages) : '',
    output: JSON.stringify(completionMessage),
    timestamp: endTime,
    start_time: beginTime,
    prompt_tokens: tokens.prompt_tokens,
    completion_tokens: tokens.completion_tokens,
    total_tokens: tokens.total_tokens,
    ...(lat !== undefined ? { latency: lat } : {}),
  };
  if (reqConfig.temperature != null) (genSpan as any).temperature = reqConfig.temperature;
  if (reqConfig.maxOutputTokens != null) (genSpan as any).max_tokens = reqConfig.maxOutputTokens;
  spans.push(genSpan);

  // Reasoning span
  if (thoughtsTokens > 0) {
    spans.push({
      trace_unique_id: traceUniqueId,
      span_unique_id: `gcli_${safeId}_reasoning`,
      span_parent_id: rootSpanId,
      span_name: 'Reasoning',
      span_workflow_name: workflowName,
      span_path: 'reasoning',
      provider_id: '',
      metadata: { reasoning_tokens: thoughtsTokens },
      input: '',
      output: `[Reasoning: ${thoughtsTokens} tokens]`,
      timestamp: endTime,
      start_time: beginTime,
    });
  }

  // Tool child spans
  for (let i = 0; i < toolTurns; i++) {
    const detail = toolDetails[i] ?? null;
    const toolName = detail?.name ?? '';
    const toolArgs = detail?.args ?? detail?.input ?? {};
    const toolOutput = detail?.output ?? '';
    const displayName = toolName ? toolDisplayName(toolName) : `Call ${i + 1}`;
    const toolInputStr = toolName ? formatToolInput(toolName, toolArgs) : '';
    const toolMeta: Record<string, unknown> = {};
    if (toolName) toolMeta.tool_name = toolName;
    if (detail?.error) toolMeta.error = detail.error;

    const toolStart = detail?.start_time ?? beginTime;
    const toolEnd = detail?.end_time ?? endTime;
    const toolLat = latencySeconds(toolStart, toolEnd);

    spans.push({
      trace_unique_id: traceUniqueId,
      span_unique_id: `gcli_${safeId}_tool_${i + 1}`,
      span_parent_id: rootSpanId,
      span_name: `Tool: ${displayName}`,
      span_workflow_name: workflowName,
      span_path: toolName ? `tool_${toolName}` : 'tool_call',
      provider_id: '',
      metadata: toolMeta,
      input: toolInputStr,
      output: truncate(toolOutput, MAX_CHARS),
      timestamp: toolEnd,
      start_time: toolStart,
      ...(toolLat !== undefined ? { latency: toolLat } : {}),
    });
  }

  return addDefaultsToAll(spans);
}

// ── Send spans (detached subprocess for Gemini CLI survival) ──────

function sendSpansDetached(spans: SpanData[], apiKey: string, baseUrl: string): void {
  const url = resolveTracingIngestEndpoint(baseUrl);
  debug(`Sending ${spans.length} span(s) to ${url}: ${spans.map(s => s.span_name).join(', ')}`);

  if (DEBUG_MODE) {
    const debugFile = path.join(STATE_DIR, 'respan_last_payload.json');
    fs.writeFileSync(debugFile, JSON.stringify(spans, null, 2));
  }

  // Convert to OTLP JSON and write to temp file for detached sender
  const payloadFile = path.join(STATE_DIR, `respan_send_${process.pid}.json`);
  fs.writeFileSync(payloadFile, JSON.stringify(toOtlpPayload(spans)));

  const senderScript = `
const fs = require('fs');
const pf = ${JSON.stringify(payloadFile)};
try {
  const data = fs.readFileSync(pf);
  (async () => {
    for (let i = 0; i < 3; i++) {
      try {
        const r = await fetch(${JSON.stringify(url)}, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Respan-Dogfood': '1',
            'Authorization': 'Bearer ' + process.env.RESPAN_API_KEY,
          },
          body: data,
          signal: AbortSignal.timeout(30000),
        });
        if (r.status < 500) break;
        if (i < 2) await new Promise(r => setTimeout(r, 1000));
      } catch(e) {
        if (i < 2) await new Promise(r => setTimeout(r, 1000));
      }
    }
  })().finally(() => { try { fs.unlinkSync(pf); } catch {} });
} catch(e) { try { fs.unlinkSync(pf); } catch {} }
`;

  const env = { ...process.env, RESPAN_API_KEY: apiKey };
  try {
    const child = execFile('node', ['-e', senderScript], {
      env,
      stdio: 'ignore' as any,
      detached: true,
    } as any);
    child.unref();
    debug('Launched sender subprocess');
  } catch (e) {
    log('ERROR', `Failed to launch sender: ${e}`);
    try { fs.unlinkSync(payloadFile); } catch {}
  }
}

function launchDelayedSend(
  sessionId: string,
  sendVersion: number,
  spans: SpanData[],
  apiKey: string,
  baseUrl: string,
): void {
  // Convert to OTLP JSON before writing — detached sender posts raw bytes
  const payloadFile = path.join(STATE_DIR, `respan_delayed_${process.pid}.json`);
  fs.writeFileSync(payloadFile, JSON.stringify(toOtlpPayload(spans)));

  const stateFile = statePath(sessionId);
  const url = resolveTracingIngestEndpoint(baseUrl);

  const script = `
const fs = require('fs');
setTimeout(async () => {
  const sf = ${JSON.stringify(stateFile)};
  const pf = ${JSON.stringify(payloadFile)};
  try {
    if (!fs.existsSync(sf)) { fs.unlinkSync(pf); process.exit(0); }
    const state = JSON.parse(fs.readFileSync(sf, 'utf-8'));
    if (state.send_version !== ${sendVersion}) { fs.unlinkSync(pf); process.exit(0); }
    const data = fs.readFileSync(pf);
    for (let i = 0; i < 3; i++) {
      try {
        const r = await fetch(${JSON.stringify(url)}, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Respan-Dogfood': '1',
            'Authorization': 'Bearer ' + process.env.RESPAN_API_KEY,
          },
          body: data,
          signal: AbortSignal.timeout(30000),
        });
        if (r.status < 500) break;
        if (i < 2) await new Promise(r => setTimeout(r, 1000));
      } catch(e) { if (i < 2) await new Promise(r => setTimeout(r, 1000)); }
    }
    try { fs.unlinkSync(sf); } catch {}
    try { fs.unlinkSync(pf); } catch {}
  } catch(e) { try { fs.unlinkSync(pf); } catch {} }
}, ${SEND_DELAY * 1000});
`;

  const env = { ...process.env, RESPAN_API_KEY: apiKey };
  try {
    const child = execFile('node', ['-e', script], {
      env,
      stdio: 'ignore' as any,
      detached: true,
    } as any);
    child.unref();
    debug(`Launched delayed sender (version=${sendVersion}, delay=${SEND_DELAY}s)`);
  } catch (e) {
    log('ERROR', `Failed to launch delayed sender: ${e}`);
    try { fs.unlinkSync(payloadFile); } catch {}
  }
}

// ── BeforeTool / AfterTool handlers ──────────────────────────────

function processBeforeTool(hookData: Msg): void {
  const sessionId = String(hookData.session_id ?? 'unknown');
  const toolName = String(hookData.tool_name ?? '');
  const toolInput = hookData.tool_input ?? {};
  debug(`BeforeTool: ${toolName}`);

  const state = loadStreamState(sessionId);
  const pending = state.pending_tools ?? [];
  pending.push({ name: toolName, input: toolInput, start_time: nowISO() });
  state.pending_tools = pending;
  saveStreamState(sessionId, state);

  process.stdout.write('{}\n');
}

function processAfterTool(hookData: Msg): void {
  const sessionId = String(hookData.session_id ?? 'unknown');
  const toolName = String(hookData.tool_name ?? '');
  const toolResponse = (hookData.tool_response ?? {}) as Msg;
  const output = String(toolResponse.llmContent ?? '');
  const error = toolResponse.error ? String(toolResponse.error) : undefined;
  debug(`AfterTool: ${toolName}, output_len=${output.length}, error=${error}`);

  const state = loadStreamState(sessionId);
  const pending = state.pending_tools ?? [];
  const completed = state.tool_details ?? [];

  // Match last pending tool with this name
  for (let i = pending.length - 1; i >= 0; i--) {
    if (pending[i].name === toolName) {
      const detail = pending.splice(i, 1)[0];
      detail.output = output;
      detail.end_time = nowISO();
      if (error) detail.error = error;
      completed.push(detail);
      break;
    }
  }

  state.pending_tools = pending;
  state.tool_details = completed;
  saveStreamState(sessionId, state);

  process.stdout.write('{}\n');
}

// ── AfterModel chunk processing ──────────────────────────────────

function processChunk(hookData: Msg): void {
  const sessionId = String(hookData.session_id ?? 'unknown');

  const llmResp = (hookData.llm_response ?? {}) as Msg;
  const chunkText = String(llmResp.text ?? '') || '';
  const usage = (llmResp.usageMetadata ?? {}) as Msg;
  const completionTokens = Number(usage.candidatesTokenCount ?? 0);
  const thoughtsTokens = Number(usage.thoughtsTokenCount ?? 0);

  // Check for finish signal and tool calls
  const candidates = (llmResp.candidates ?? []) as Msg[];
  let finishReason = '';
  let hasToolCall = false;
  const chunkToolDetails: ToolDetail[] = [];

  if (candidates.length > 0 && typeof candidates[0] === 'object') {
    finishReason = String(candidates[0].finishReason ?? '');
    const content = (candidates[0].content ?? {}) as Msg;
    if (typeof content === 'object') {
      for (const part of (content.parts ?? []) as Msg[]) {
        if (typeof part !== 'object') continue;
        const fc = (part.functionCall ?? part.toolCall) as Msg | undefined;
        if (fc) {
          hasToolCall = true;
          if (typeof fc === 'object') {
            chunkToolDetails.push({
              name: String(fc.name ?? ''),
              args: fc.args ?? {},
            });
          }
        }
      }
    }
  }

  const messages = ((hookData.llm_request as Msg)?.messages ?? []) as Msg[];
  const currentMsgCount = messages.length;

  let state = loadStreamState(sessionId);
  const isFinished = ['STOP', 'MAX_TOKENS', 'SAFETY'].includes(finishReason);

  // Detect tool-call resumption via message count
  const savedMsgCount = state.msg_count ?? 0;
  let toolCallDetected = false;

  if (savedMsgCount > 0 && currentMsgCount > savedMsgCount) {
    const newMsgs = messages.slice(savedMsgCount);
    const hasNewUserMsg = newMsgs.some((m) => m.role === 'user');
    if (hasNewUserMsg) {
      debug(`New user message detected (msgs ${savedMsgCount} → ${currentMsgCount}), starting fresh turn`);
      clearStreamState(sessionId);
      state = { accumulated_text: '', last_tokens: 0, first_chunk_time: '' };
    } else {
      state.tool_turns = (state.tool_turns ?? 0) + 1;
      state.send_version = (state.send_version ?? 0) + 1;
      toolCallDetected = true;
      debug(`Tool call detected via msg_count (${savedMsgCount} → ${currentMsgCount}), tool_turns=${state.tool_turns}`);
    }
  }
  state.msg_count = currentMsgCount;

  // Accumulate text
  if (chunkText) {
    if (!state.first_chunk_time) state.first_chunk_time = nowISO();
    state.accumulated_text += chunkText;
    state.last_tokens = completionTokens || state.last_tokens;
    if (thoughtsTokens > 0) state.thoughts_tokens = thoughtsTokens;
    saveStreamState(sessionId, state);
    debug(`Accumulated chunk: +${chunkText.length} chars, total=${state.accumulated_text.length}`);
  }

  // Tool call in response parts
  const isToolTurn = hasToolCall || ['TOOL_CALLS', 'FUNCTION_CALL', 'TOOL_USE'].includes(finishReason);
  if (isToolTurn) {
    state.tool_turns = (state.tool_turns ?? 0) + 1;
    state.send_version = (state.send_version ?? 0) + 1;
    if (chunkToolDetails.length) {
      state.tool_details = [...(state.tool_details ?? []), ...chunkToolDetails];
    }
    saveStreamState(sessionId, state);
    debug(`Tool call via response parts (finish=${finishReason}), tool_turns=${state.tool_turns}`);
    process.stdout.write('{}\n');
    return;
  }

  // Detect completion and send
  const hasNewText = state.accumulated_text.length > (state.last_send_text_len ?? 0);
  const shouldSend = (
    (!toolCallDetected || isFinished)
    && hasNewText
    && state.accumulated_text
    && (!chunkText || isFinished)
  );

  process.stdout.write('{}\n');

  if (!shouldSend) {
    if (toolCallDetected) saveStreamState(sessionId, state);
    return;
  }

  const creds = resolveCredentials();
  if (!creds) {
    log('ERROR', 'No API key found. Run: respan auth login');
    clearStreamState(sessionId);
    return;
  }

  const finalPrompt = Number(usage.promptTokenCount ?? 0);
  const finalCompletion = completionTokens || state.last_tokens;
  const finalTotal = Number(usage.totalTokenCount ?? 0) || (finalPrompt + finalCompletion);
  const tok = { prompt_tokens: finalPrompt, completion_tokens: finalCompletion, total_tokens: finalTotal };

  const config = loadRespanConfig(path.join(os.homedir(), '.gemini', 'respan.json'));
  const spans = buildSpans(
    hookData, state.accumulated_text, tok, config,
    state.first_chunk_time || undefined,
    state.tool_turns ?? 0,
    state.tool_details ?? [],
    state.thoughts_tokens ?? 0,
  );

  // Method b: text + STOP → send immediately
  if (isFinished && chunkText) {
    debug(`Immediate send (text+STOP, tool_turns=${state.tool_turns ?? 0}), ${state.accumulated_text.length} chars`);
    sendSpansDetached(spans, creds.apiKey, creds.baseUrl);
    clearStreamState(sessionId);
    return;
  }

  // Method a: delayed send
  state.send_version = (state.send_version ?? 0) + 1;
  state.last_send_text_len = state.accumulated_text.length;
  saveStreamState(sessionId, state);
  debug(`Delayed send (version=${state.send_version}, delay=${SEND_DELAY}s), ${state.accumulated_text.length} chars`);
  launchDelayedSend(sessionId, state.send_version!, spans, creds.apiKey, creds.baseUrl);
}

// ── Main ──────────────────────────────────────────────────────────

function main(): void {
  try {
    const raw = fs.readFileSync(0, 'utf-8');
    if (!raw.trim()) {
      process.stdout.write('{}\n');
      return;
    }

    const hookData = JSON.parse(raw) as Msg;
    const event = String(hookData.hook_event_name ?? '');

    const unlock = acquireLock(LOCK_PATH);
    try {
      if (event === 'BeforeTool') {
        processBeforeTool(hookData);
      } else if (event === 'AfterTool') {
        processAfterTool(hookData);
      } else {
        processChunk(hookData);
      }
    } finally {
      unlock?.();
    }
  } catch (e) {
    if (e instanceof SyntaxError) {
      log('ERROR', `Invalid JSON from stdin: ${e}`);
    } else {
      log('ERROR', `Hook error: ${e}`);
    }
    process.stdout.write('{}\n');
  }
}

main();
