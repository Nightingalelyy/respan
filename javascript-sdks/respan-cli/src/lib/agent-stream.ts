/**
 * Agent streaming display — parses Claude Code / Cursor JSONL output
 * and shows live progress with spinners for tool calls and dimmed text.
 */

import { ChildProcess } from 'node:child_process';
import * as readline from 'node:readline';

const DIM = '\x1b[2m';
const GREEN = '\x1b[32m';
const CYAN = '\x1b[36m';
const RESET = '\x1b[0m';

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPINNER_INTERVAL = 80;

// ── Stream line types (Claude Code / Cursor JSONL) ─────────────────────

interface StreamEventLine {
  type: 'stream_event';
  event: StreamEvent;
}

interface AssistantLine {
  type: 'assistant';
}

interface UserLine {
  type: 'user';
}

interface ResultLine {
  type: 'result';
  session_id?: string;
  cost?: number;
}

type StreamLine = StreamEventLine | AssistantLine | UserLine | ResultLine | { type: string };

interface ContentBlockStartEvent {
  type: 'content_block_start';
  index: number;
  content_block: ContentBlock;
}

interface ContentBlockDeltaEvent {
  type: 'content_block_delta';
  index: number;
  delta: Delta;
}

interface ContentBlockStopEvent {
  type: 'content_block_stop';
  index: number;
}

type StreamEvent =
  | ContentBlockStartEvent
  | ContentBlockDeltaEvent
  | ContentBlockStopEvent
  | { type: 'message_delta' }
  | { type: 'message_stop' }
  | { type: string };

interface TextContentBlock {
  type: 'text';
}

interface ToolUseContentBlock {
  type: 'tool_use';
  name: string;
}

type ContentBlock = TextContentBlock | ToolUseContentBlock | { type: string };

interface TextDelta {
  type: 'text_delta';
  text: string;
}

interface InputJsonDelta {
  type: 'input_json_delta';
  partial_json: string;
}

type Delta = TextDelta | InputJsonDelta | { type: string };

// ── Block state ────────────────────────────────────────────────────────

interface TextBlock {
  kind: 'text';
}

interface ToolUseBlock {
  kind: 'tool_use';
  name: string;
  partialInput: string;
}

type BlockState = TextBlock | ToolUseBlock;

// ── Display ────────────────────────────────────────────────────────────

class AgentStreamDisplay {
  private blocks = new Map<number, BlockState>();
  private spinnerTimer: ReturnType<typeof setInterval> | null = null;
  private spinnerFrame = 0;
  private spinnerMessage = '';
  private hasTextOutput = false;
  private isTTY: boolean;

  constructor() {
    this.isTTY = process.stderr.isTTY === true && !process.env.NO_COLOR;
  }

  handle(line: StreamLine): void {
    if (line.type === 'stream_event') {
      this.handleEvent((line as StreamEventLine).event);
    }
  }

  private handleEvent(event: StreamEvent): void {
    switch (event.type) {
      case 'content_block_start': {
        const e = event as ContentBlockStartEvent;
        if (e.content_block.type === 'text') {
          this.blocks.set(e.index, { kind: 'text' });
        } else if (e.content_block.type === 'tool_use') {
          this.clearSpinner();
          if (this.hasTextOutput) {
            process.stderr.write('\n');
            this.hasTextOutput = false;
          }
          const name = (e.content_block as ToolUseContentBlock).name;
          this.startSpinner(toolDisplay(name, ''));
          this.blocks.set(e.index, { kind: 'tool_use', name, partialInput: '' });
        }
        break;
      }

      case 'content_block_delta': {
        const e = event as ContentBlockDeltaEvent;
        if (e.delta.type === 'text_delta') {
          if (this.spinnerTimer) this.suspendSpinner();
          process.stderr.write(`${DIM}${(e.delta as TextDelta).text}${RESET}`);
          this.hasTextOutput = true;
        } else if (e.delta.type === 'input_json_delta') {
          const block = this.blocks.get(e.index);
          if (block?.kind === 'tool_use') {
            block.partialInput += (e.delta as InputJsonDelta).partial_json;
            this.updateSpinner(toolDisplay(block.name, block.partialInput));
          }
        }
        break;
      }

      case 'content_block_stop': {
        const e = event as ContentBlockStopEvent;
        const block = this.blocks.get(e.index);
        if (block) {
          this.blocks.delete(e.index);
          if (block.kind === 'text') {
            if (this.hasTextOutput) {
              process.stderr.write('\n');
              this.hasTextOutput = false;
            }
          } else if (block.kind === 'tool_use') {
            this.finishSpinner(toolDoneDisplay(block.name, block.partialInput));
          }
        }
        break;
      }
    }
  }

  private startSpinner(message: string): void {
    this.clearSpinner();
    this.spinnerMessage = message;
    if (!this.isTTY) {
      process.stderr.write(`  … ${DIM}${message}${RESET}\n`);
      return;
    }
    this.renderSpinner();
    this.spinnerTimer = setInterval(() => this.renderSpinner(), SPINNER_INTERVAL);
  }

  private renderSpinner(): void {
    process.stderr.write(`\r\x1b[K  ${CYAN}${SPINNER_FRAMES[this.spinnerFrame]}${RESET} ${DIM}${this.spinnerMessage}${RESET}`);
    this.spinnerFrame = (this.spinnerFrame + 1) % SPINNER_FRAMES.length;
  }

  private updateSpinner(message: string): void {
    this.spinnerMessage = message;
    if (this.isTTY && this.spinnerTimer) {
      process.stderr.write(`\r\x1b[K  ${CYAN}${SPINNER_FRAMES[this.spinnerFrame]}${RESET} ${DIM}${message}${RESET}`);
    }
  }

  private suspendSpinner(): void {
    if (this.spinnerTimer) {
      clearInterval(this.spinnerTimer);
      this.spinnerTimer = null;
      if (this.isTTY) process.stderr.write('\r\x1b[K');
    }
  }

  private clearSpinner(): void {
    if (this.spinnerTimer) {
      clearInterval(this.spinnerTimer);
      this.spinnerTimer = null;
    }
    if (this.isTTY) process.stderr.write('\r\x1b[K');
  }

  private finishSpinner(doneMessage: string): void {
    this.clearSpinner();
    process.stderr.write(`  ${GREEN}\u2713${RESET} ${DIM}${doneMessage}${RESET}\n`);
  }

  finish(): void {
    this.clearSpinner();
    if (this.hasTextOutput) {
      process.stderr.write('\n');
      this.hasTextOutput = false;
    }
  }
}

// ── Tool display helpers ───────────────────────────────────────────────

function toolDisplay(name: string, partialInput: string): string {
  const target = extractTarget(partialInput);
  const action: Record<string, string> = {
    Read: 'Reading',
    Write: 'Writing',
    Edit: 'Editing',
    MultiEdit: 'Editing',
    Grep: 'Searching',
    Glob: 'Finding files',
    WebFetch: 'Fetching',
    WebSearch: 'Searching web',
    NotebookEdit: 'Editing notebook',
  };

  if (name === 'Bash') {
    return target ? `Running: ${target}` : 'Running command';
  }

  const verb = action[name] || name;
  return target ? `${verb} ${target}` : verb;
}

function toolDoneDisplay(name: string, partialInput: string): string {
  const target = extractTarget(partialInput);
  const action: Record<string, string> = {
    Read: 'Read',
    Write: 'Wrote',
    Edit: 'Edited',
    MultiEdit: 'Edited',
    Grep: 'Searched',
    Glob: 'Found files',
    WebFetch: 'Fetched',
    WebSearch: 'Searched web',
    NotebookEdit: 'Edited notebook',
  };

  if (name === 'Bash') {
    return target ? `Ran: ${target}` : 'Ran command';
  }

  const verb = action[name] || name;
  return target ? `${verb} ${target}` : verb;
}

function extractTarget(partialJson: string): string | undefined {
  if (!partialJson) return undefined;

  try {
    const obj = JSON.parse(partialJson);
    if (obj.file_path) return shortPath(obj.file_path);
    if (obj.command) return truncate(obj.command, 50);
    if (obj.pattern) return `/${truncate(obj.pattern, 30)}/`;
  } catch {
    // Try regex on partial JSON
    const fpMatch = partialJson.match(/"file_path"\s*:\s*"([^"]+)"/);
    if (fpMatch) return shortPath(fpMatch[1]);

    const cmdMatch = partialJson.match(/"command"\s*:\s*"([^"]+)"/);
    if (cmdMatch) return truncate(cmdMatch[1], 50);
  }

  return undefined;
}

function shortPath(filepath: string): string {
  const parts = filepath.split('/').filter(Boolean);
  return parts.length <= 2 ? parts.join('/') : parts.slice(-2).join('/');
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max) + '…';
}

// ── Public entry point ─────────────────────────────────────────────────

/**
 * Stream agent output from a child process, showing live progress.
 * Returns the exit code.
 */
export async function streamAgentOutput(child: ChildProcess): Promise<number> {
  const display = new AgentStreamDisplay();

  return new Promise<number>((resolve) => {
    if (child.stdout) {
      const rl = readline.createInterface({ input: child.stdout });
      rl.on('line', (line) => {
        if (!line.trim()) return;
        try {
          const parsed = JSON.parse(line) as StreamLine;
          display.handle(parsed);
        } catch {
          // Not JSON — ignore
        }
      });
    }

    if (child.stderr) {
      const rl = readline.createInterface({ input: child.stderr });
      rl.on('line', (line) => {
        process.stderr.write(`${DIM}${line}${RESET}\n`);
      });
    }

    child.on('close', (code) => {
      display.finish();
      resolve(code ?? 0);
    });

    // Handle Ctrl+C
    process.on('SIGINT', () => {
      display.finish();
      process.stderr.write(`${DIM}Stopping agent…${RESET}\n`);
      child.kill('SIGTERM');
    });
  });
}
