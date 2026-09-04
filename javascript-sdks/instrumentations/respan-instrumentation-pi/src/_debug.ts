/**
 * Debug output goes to stderr only, and only when `RESPAN_PI_DEBUG` is set.
 * The package never writes to disk and never prints to stdout (pi's TUI owns
 * stdout).
 */

const FALSY = new Set(["", "0", "false", "off", "no"]);

export function parseBooleanFlag(value: unknown): boolean | undefined {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value !== "string" && typeof value !== "number") {
    return undefined;
  }
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "1" || normalized === "true" || normalized === "on" || normalized === "yes") {
    return true;
  }
  if (FALSY.has(normalized)) {
    return false;
  }
  return undefined;
}

export function isDebugEnabled(
  env: Record<string, string | undefined> = process.env,
): boolean {
  const raw = env.RESPAN_PI_DEBUG;
  if (raw === undefined) {
    return false;
  }
  return !FALSY.has(raw.trim().toLowerCase());
}

export function debugLog(message: string, error?: unknown): void {
  if (!isDebugEnabled()) {
    return;
  }
  const detail =
    error === undefined
      ? ""
      : ` ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`;
  try {
    process.stderr.write(`[respan-pi] ${message}${detail}\n`);
  } catch {
    // Never let diagnostics interfere with pi.
  }
}
