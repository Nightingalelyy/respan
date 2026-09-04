/**
 * pi package entry (`"pi": { "extensions": ["./dist/extension.js"] }`).
 *
 * Loaded by the pi CLI after `pi install npm:@respan/instrumentation-pi`
 * (or `respan integrate pi`). It resolves configuration, creates a `Respan`
 * umbrella with an explicit `PiInstrumentor` (explicit instrumentations
 * disable Traceloop auto-instrumentation inside the pi process), initializes
 * tracing lazily on the first `session_start`, starts a flush after each
 * agent run without ever blocking pi on it, and shuts down on quit within a
 * bounded wait. Everything is fail-open: if tracing is unavailable pi keeps
 * running and the footer shows why.
 */

import { Respan, type RespanOptions } from "@respan/respan";

import { resolvePiRespanConfig, type PiRespanConfig } from "./_config.js";
import { PiInstrumentor } from "./index.js";
import type { PiExtensionAPI, PiExtensionContextLike } from "./_pi_types.js";

export type { PiRespanConfig, ResolvePiRespanConfigOptions } from "./_config.js";
export { resolvePiRespanConfig } from "./_config.js";

export interface RespanLike {
  initialize(): Promise<void>;
  flush(): Promise<void>;
  shutdown(): Promise<void>;
}

export type CreateRespanFn = (options: RespanOptions) => RespanLike;

export interface RespanPiExtensionOverrides {
  /** Use this configuration instead of resolving files/env. */
  config?: Partial<PiRespanConfig>;
  /** Factory for the Respan umbrella (tests inject a fake). */
  createRespan?: CreateRespanFn;
  cwd?: string;
  env?: Record<string, string | undefined>;
  homeDir?: string;
  /** Debug sink (default: stderr). */
  log?: (message: string) => void;
  /** Bound for the final flush / shutdown on `session_shutdown` (tests). Default `SHUTDOWN_FLUSH_TIMEOUT_MS`. */
  shutdownTimeoutMs?: number;
}

const STATUS_KEY = "respan";
const STATUS_TRACING = "Respan: tracing";
const STATUS_OFF = "Respan: tracing off (run `respan integrate pi`)";
const RUNTIME_REGISTRY_KEY = Symbol.for("respan.instrumentation-pi.runtimes");
const CONSOLE_FILTER_KEY = Symbol.for("respan.instrumentation-pi.console-filter");
/** Every stdout/stderr line `@respan/tracing` prints regardless of `logLevel` starts with this. */
const RESPAN_CONSOLE_PREFIX = "[Respan";
/**
 * pi awaits the `session_shutdown` handlers and, on quit, exits right after
 * them. The final export gets this long; then pi moves on (fail-open).
 */
export const SHUTDOWN_FLUSH_TIMEOUT_MS = 3000;

interface SharedRuntime {
  key?: string;
  respan: RespanLike;
  instrumentor: PiInstrumentor;
  initPromise?: Promise<boolean>;
  /** `beforeExit` flush hook installed (once per runtime). */
  exitHookInstalled?: boolean;
  initialized: boolean;
  failure?: string;
  shutdownPromise?: Promise<void>;
}

type RecordValue = Record<string, unknown>;

export function createRespanPiExtension(
  overrides: RespanPiExtensionOverrides = {},
): (pi: PiExtensionAPI) => void {
  return function respanPiExtension(pi: PiExtensionAPI): void {
    const log =
      overrides.log ??
      ((message: string): void => {
        try {
          process.stderr.write(`[respan-pi] ${message}\n`);
        } catch {
          // ignore
        }
      });

    let config: PiRespanConfig;
    try {
      config = overrides.config
        ? withDefaults(overrides.config)
        : resolvePiRespanConfig({
            cwd: overrides.cwd ?? process.cwd(),
            env: overrides.env,
            homeDir: overrides.homeDir,
          });
    } catch (error) {
      log(`failed to resolve configuration: ${messageOf(error)}`);
      config = { enabled: false, debug: false, sources: [] };
    }
    const debug = (message: string): void => {
      if (config.debug) {
        log(message);
      }
    };
    const shutdownTimeoutMs = overrides.shutdownTimeoutMs ?? SHUTDOWN_FLUSH_TIMEOUT_MS;

    if (!config.enabled || !config.apiKey) {
      debug(
        `tracing disabled (${config.apiKey ? "enabled=false" : "no API key"}; sources: ${
          config.sources.join(", ") || "defaults"
        })`,
      );
      pi.on("session_start", (_event: unknown, ctx: PiExtensionContextLike | undefined) => {
        setStatus(ctx, STATUS_OFF);
      });
      return;
    }

    let runtime: SharedRuntime;
    try {
      runtime = acquireRuntime(config, overrides.createRespan, debug);
    } catch (error) {
      log(`failed to create Respan runtime: ${messageOf(error)}`);
      pi.on("session_start", (_event: unknown, ctx: PiExtensionContextLike | undefined) => {
        setStatus(ctx, `Respan: tracing unavailable: ${messageOf(error)}`);
      });
      return;
    }
    debug(`tracing enabled (sources: ${config.sources.join(", ") || "defaults"})`);

    // Register the tracer's handlers first so they run before our flushes.
    runtime.instrumentor.extension(pi);

    const ensureInitialized = (): Promise<boolean> => {
      if (!runtime.initPromise) {
        runtime.initPromise = Promise.resolve()
          .then(() => runtime.respan.initialize())
          .then(
            () => {
              runtime.initialized = true;
              debug("tracing initialized");
              return true;
            },
            (error: unknown) => {
              runtime.failure = messageOf(error);
              debug(`initialize failed: ${runtime.failure}`);
              return false;
            },
          );
      }
      return runtime.initPromise;
    };

    const flush = async (): Promise<void> => {
      if (!runtime.initialized) {
        return;
      }
      try {
        await runtime.respan.flush();
      } catch (error) {
        debug(`flush failed: ${messageOf(error)}`);
      }
    };
    // Exports never sit on pi's event path: pi awaits extension handlers
    // before it notifies subscribers / resolves `session.prompt()`, and an
    // export can take the exporter's full timeout while Respan is unreachable.
    // The batch processor exports on its own schedule anyway — flushing here
    // only makes a finished run visible sooner, so it runs in the background.
    const flushInBackground = (): void => {
      void flush();
    };

    pi.on("session_start", async (_event: unknown, ctx: PiExtensionContextLike | undefined) => {
      try {
        const ok = await ensureInitialized();
        setStatus(
          ctx,
          ok ? STATUS_TRACING : `Respan: tracing unavailable: ${runtime.failure ?? "unknown error"}`,
        );
      } catch (error) {
        debug(`session_start handler failed: ${messageOf(error)}`);
      }
    });
    pi.on("agent_end", flushInBackground);
    pi.on("session_compact", flushInBackground);
    pi.on("session_tree", flushInBackground);
    pi.on("session_shutdown", async (event: unknown, ctx: PiExtensionContextLike | undefined) => {
      try {
        const reason = isRecord(event) ? event.reason : undefined;
        // The one place durability is worth a wait: pi exits right after this
        // handler on quit. Still bounded, so an outage cannot hang `/quit`.
        await withTimeout(
          flush().then(() => (reason === "quit" ? shutdownRuntime(runtime, debug) : undefined)),
          shutdownTimeoutMs,
        );
      } catch (error) {
        debug(`session_shutdown handler failed: ${messageOf(error)}`);
      } finally {
        setStatus(ctx, undefined);
      }
    });

    // Initialize eagerly instead of waiting for `session_start`: sessions
    // created through the SDK (`createAgentSession()` in a script) load
    // installed packages but do not necessarily emit `session_start`, and a
    // run that starts before initialization would otherwise emit nothing.
    void ensureInitialized();

    // SDK scripts also rarely emit `session_shutdown`; they just exit after
    // `session.dispose()`. Flush whatever is still batched when the event loop
    // drains, so short-lived scripts do not lose their last spans.
    if (!runtime.exitHookInstalled) {
      runtime.exitHookInstalled = true;
      process.once("beforeExit", () => {
        void flush();
      });
    }
  };
}

/** Default export: the pi extension factory. */
export default function respanPiExtension(pi: PiExtensionAPI): void {
  createRespanPiExtension()(pi);
}

// ── Runtime sharing ───────────────────────────────────────────────────────

/**
 * One `Respan` per process and configuration. pi re-runs extension factories
 * on reload / new / resume / fork; re-using the runtime keeps a single OTEL
 * pipeline (and avoids "already initialized" chatter) across those.
 */
function acquireRuntime(
  config: PiRespanConfig,
  createRespan: CreateRespanFn | undefined,
  debug: (message: string) => void,
): SharedRuntime {
  const registry = createRespan ? undefined : runtimeRegistry();
  const key = createRespan ? undefined : runtimeKey(config);
  if (registry && key) {
    const existing = registry.get(key);
    if (existing) {
      debug("re-using shared Respan runtime");
      return existing;
    }
  }

  if (!config.debug) {
    installConsoleFilter();
  }
  const instrumentor = new PiInstrumentor({
    workflowName: config.workflowName,
    agentName: config.agentName,
    traceScope: config.traceScope,
    customerIdentifier: config.customerIdentifier,
    metadata: config.metadata,
  });
  const options: RespanOptions = {
    apiKey: config.apiKey,
    baseURL: config.baseURL,
    appName: config.workflowName ?? "pi",
    instrumentations: [instrumentor],
    silenceInitializationMessage: true,
    logLevel: config.debug ? "debug" : "error",
  };
  const factory: CreateRespanFn = createRespan ?? ((opts) => new Respan(opts));
  const respan = factory(options);
  const runtime: SharedRuntime = { key, respan, instrumentor, initialized: false };
  if (registry && key) {
    registry.set(key, runtime);
  }
  return runtime;
}

async function shutdownRuntime(
  runtime: SharedRuntime,
  debug: (message: string) => void,
): Promise<void> {
  if (!runtime.shutdownPromise) {
    runtime.shutdownPromise = (async () => {
      if (runtime.key) {
        runtimeRegistry().delete(runtime.key);
      }
      if (!runtime.initialized) {
        return;
      }
      try {
        await runtime.respan.shutdown();
        debug("tracing shut down");
      } catch (error) {
        debug(`shutdown failed: ${messageOf(error)}`);
      }
    })();
  }
  return runtime.shutdownPromise;
}

function runtimeRegistry(): Map<string, SharedRuntime> {
  const holder = globalThis as unknown as Record<symbol, Map<string, SharedRuntime> | undefined>;
  let registry = holder[RUNTIME_REGISTRY_KEY];
  if (!registry) {
    registry = new Map();
    holder[RUNTIME_REGISTRY_KEY] = registry;
  }
  return registry;
}

function runtimeKey(config: PiRespanConfig): string {
  return JSON.stringify([
    config.apiKey,
    config.baseURL,
    config.workflowName,
    config.agentName,
    config.traceScope ?? "run",
    config.customerIdentifier,
    config.metadata ?? {},
    config.debug,
  ]);
}

// ── Helpers ───────────────────────────────────────────────────────────────

function withDefaults(partial: Partial<PiRespanConfig>): PiRespanConfig {
  return {
    ...partial,
    enabled: partial.enabled ?? Boolean(partial.apiKey),
    debug: partial.debug ?? false,
    sources: partial.sources ?? ["override"],
  };
}

function setStatus(ctx: PiExtensionContextLike | undefined, text: string | undefined): void {
  try {
    if (ctx?.hasUI) {
      ctx.ui?.setStatus?.(STATUS_KEY, text);
    }
  } catch {
    // The footer is best effort.
  }
}

/**
 * `@respan/tracing` writes `[Respan Debug] …` / `[Respan] …` lines to stdout
 * on initialize, flush, shutdown and on EVERY injected span, regardless of
 * `logLevel`. pi's TUI and `--mode json` own stdout, so — unless
 * `RESPAN_PI_DEBUG` is set — the console methods are wrapped once per process
 * with a pass-through filter that drops only those prefixed lines. Everything
 * else (pi's own output, other extensions, the host application) is forwarded
 * untouched, at any time, so nothing is lost while an export is in flight and
 * overlapping flushes cannot interfere. The wrapper stays installed on
 * purpose: it is transparent, and removing it at quit would let the tracing
 * library's `beforeExit` flush chatter through.
 */
function installConsoleFilter(): void {
  const holder = globalThis as unknown as Record<symbol, boolean | undefined>;
  if (holder[CONSOLE_FILTER_KEY]) {
    return;
  }
  holder[CONSOLE_FILTER_KEY] = true;
  for (const method of ["log", "info", "debug", "warn"] as const) {
    const original = console[method].bind(console);
    console[method] = (...args: unknown[]): void => {
      if (typeof args[0] === "string" && args[0].startsWith(RESPAN_CONSOLE_PREFIX)) {
        return;
      }
      original(...args);
    };
  }
}

/** Resolves with the promise's value, or `undefined` once `ms` have passed. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | undefined> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<undefined>((resolve) => {
    timer = setTimeout(() => resolve(undefined), ms);
    // Never keep the pi process alive just for this bound.
    (timer as { unref?: () => void }).unref?.();
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  });
}

function messageOf(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function isRecord(value: unknown): value is RecordValue {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
