/**
 * Simple TTY spinner for CLI feedback.
 * Falls back to plain text when not in a TTY or NO_COLOR is set.
 */

const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const INTERVAL = 80;

export interface Spinner {
  start(): void;
  succeed(text?: string): void;
  fail(text?: string): void;
  stop(): void;
}

export function createSpinner(label: string): Spinner {
  const isTTY = process.stderr.isTTY && !process.env.NO_COLOR;

  if (!isTTY) {
    return {
      start() {
        process.stderr.write(`${label}...\n`);
      },
      succeed(text?: string) {
        process.stderr.write(`${text || label} done.\n`);
      },
      fail(text?: string) {
        process.stderr.write(`${text || label} failed.\n`);
      },
      stop() {},
    };
  }

  let frameIndex = 0;
  let timer: ReturnType<typeof setInterval> | null = null;

  function clear(): void {
    process.stderr.write('\r\x1b[K');
  }

  function render(): void {
    clear();
    process.stderr.write(`${FRAMES[frameIndex]} ${label}`);
    frameIndex = (frameIndex + 1) % FRAMES.length;
  }

  return {
    start() {
      render();
      timer = setInterval(render, INTERVAL);
    },
    succeed(text?: string) {
      if (timer) clearInterval(timer);
      clear();
      process.stderr.write(`\x1b[32m✓\x1b[0m ${text || label}\n`);
    },
    fail(text?: string) {
      if (timer) clearInterval(timer);
      clear();
      process.stderr.write(`\x1b[31m✗\x1b[0m ${text || label}\n`);
    },
    stop() {
      if (timer) clearInterval(timer);
      clear();
    },
  };
}
