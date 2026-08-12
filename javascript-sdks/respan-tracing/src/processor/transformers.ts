import { type Context, type Span } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

/**
 * A synchronous span transformation that runs immediately before Respan's
 * filtering and routing processor.
 *
 * `onStart` and `onEnd` may enrich the live span. `prepareForExport` may
 * return an export-only clone after every transformer's `onEnd` hook has run.
 */
export interface RespanSpanTransformer {
  onStart?(span: Span, parentContext: Context): void;
  onEnd?(span: ReadableSpan): void;
  prepareForExport?(span: ReadableSpan): ReadableSpan;
  /** Release transformer-owned correlation/cache state. Must be idempotent. */
  dispose?(): void;
}

/** Handle returned by {@link registerSpanTransformer}. */
export interface SpanTransformerRegistration {
  readonly key: string;
  unregister(): void;
}

interface TransformerEntry {
  readonly key: string;
  readonly transformer: RespanSpanTransformer;
  referenceCount: number;
  inFlightCount: number;
  disposalRequested: boolean;
  disposed: boolean;
}

/**
 * Per-span record of the entries whose `inFlightCount` a started span holds,
 * plus whether that hold has already been released (by `onEnd` or by GC).
 */
interface InFlightHold {
  readonly entries: readonly TransformerEntry[];
  settled: boolean;
}

interface SpanTransformerRegistryState {
  readonly entries: Map<string, TransformerEntry>;
  readonly retainedEntries: Set<TransformerEntry>;
  spanSnapshots: WeakMap<object, readonly TransformerEntry[]>;
  /** Live hold per started span, so `onEnd` can settle the GC watch. */
  inFlightHolds: WeakMap<object, InFlightHold>;
  /**
   * Releases the drain barrier for spans that start but are never ended
   * (abandoned/leaked). Without this, a single un-`end()`ed span would pin
   * `inFlightCount` forever and block disposal on `unregister()`.
   */
  abandonedSpanWatch: FinalizationRegistry<InFlightHold>;
  hostCount: number;
  readonly version: 1;
}

/** Release an in-flight hold exactly once, decrementing and draining entries. */
function settleInFlightHold(
  state: SpanTransformerRegistryState,
  hold: InFlightHold,
): void {
  if (hold.settled) {
    return;
  }
  hold.settled = true;
  for (const entry of hold.entries) {
    entry.inFlightCount = Math.max(0, entry.inFlightCount - 1);
    disposeEntryIfReady(state, entry);
  }
}

/**
 * Process-global registry key. `Symbol.for` keeps independently resolved
 * copies of `@respan/tracing` on the same transformer lifecycle.
 */
export const RESPAN_SPAN_TRANSFORMER_REGISTRY_SYMBOL = Symbol.for(
  "respan.tracing.spanTransformerRegistry.v1",
);

/**
 * Register a transformer under a stable package-owned key.
 *
 * Registrations with the same key are reference-counted. The first
 * transformer registered for a key remains authoritative until the last
 * owner unregisters, which makes repeated instrumentor activation
 * deterministic and prevents duplicate translation.
 */
export function registerSpanTransformer(
  key: string,
  transformer: RespanSpanTransformer,
): SpanTransformerRegistration {
  const normalizedKey = key.trim();
  if (!normalizedKey) {
    throw new Error("Span transformer key must be a non-empty string.");
  }
  if (!transformer || typeof transformer !== "object") {
    throw new Error(
      `Span transformer "${normalizedKey}" must be an object.`,
    );
  }

  const state = getRegistryState();
  if (state.hostCount === 0) {
    throw new Error(
      "No compatible Respan span-transformer host is active. Initialize a " +
        "@respan/tracing runtime with transformer registry v1 before " +
        `activating \"${normalizedKey}\".`,
    );
  }
  let entry = state.entries.get(normalizedKey);
  if (entry) {
    entry.referenceCount += 1;
  } else {
    entry = [...state.retainedEntries].find(
      (candidate) =>
        candidate.key === normalizedKey &&
        candidate.transformer === transformer &&
        !candidate.disposed,
    );
    if (entry) {
      // Reactivation while older spans drain must retain the exact transformer's
      // state. Otherwise the older entry could dispose shared correlation data
      // underneath the newly active registration.
      entry.referenceCount = 1;
      entry.disposalRequested = false;
    } else {
      entry = {
        key: normalizedKey,
        transformer,
        referenceCount: 1,
        inFlightCount: 0,
        disposalRequested: false,
        disposed: false,
      };
      state.retainedEntries.add(entry);
    }
    state.entries.set(normalizedKey, entry);
  }

  let registered = true;
  return {
    key: normalizedKey,
    unregister(): void {
      if (!registered) {
        return;
      }
      registered = false;

      entry.referenceCount = Math.max(0, entry.referenceCount - 1);
      if (
        entry.referenceCount === 0 &&
        state.entries.get(normalizedKey) === entry
      ) {
        state.entries.delete(normalizedKey);
        entry.disposalRequested = true;
        disposeEntryIfReady(state, entry);
      }
    },
  };
}

/** Return active transformer keys in their deterministic execution order. */
export function getRegisteredSpanTransformerKeys(): readonly string[] {
  return sortedActiveEntries(getRegistryState()).map((entry) => entry.key);
}

/** @internal Marks a compatible composite processor as active. */
export function acquireSpanTransformerHost(): void {
  const state = getRegistryState();
  state.hostCount += 1;
}

/**
 * @internal Releases a compatible composite processor. The final host owns
 * process-runtime cleanup so registrations and in-flight snapshots cannot
 * leak into a later tracing lifecycle.
 */
export function releaseSpanTransformerHost(): void {
  const state = getRegistryState();
  state.hostCount = Math.max(0, state.hostCount - 1);
  if (state.hostCount === 0) {
    state.entries.clear();
    for (const entry of state.retainedEntries) {
      disposeEntry(state, entry);
    }
    state.retainedEntries.clear();
    state.spanSnapshots = new WeakMap();
    state.inFlightHolds = new WeakMap();
  }
}

/** @internal Called by RespanCompositeProcessor before any other start logic. */
export function runSpanTransformersOnStart(
  span: Span,
  parentContext: Context,
): void {
  const state = getRegistryState();
  const entries = sortedActiveEntries(state);

  // Record even an empty snapshot. A transformer registered after this span
  // starts must not retroactively claim it at onEnd.
  state.spanSnapshots.set(span as object, entries);
  for (const entry of entries) {
    entry.inFlightCount += 1;
  }

  // Track the hold so an abandoned span (started but never ended) still
  // releases the drain barrier once it is garbage-collected. onEnd settles
  // this synchronously; the FinalizationRegistry is the fallback.
  if (entries.length > 0) {
    const spanObject = span as object;
    const hold: InFlightHold = { entries, settled: false };
    state.inFlightHolds.set(spanObject, hold);
    state.abandonedSpanWatch.register(spanObject, hold, spanObject);
  }

  for (const entry of entries) {
    if (!entry.transformer.onStart) {
      continue;
    }
    try {
      entry.transformer.onStart(span, parentContext);
    } catch (error) {
      reportTransformerError(entry.key, "onStart", error);
    }
  }
}

/**
 * @internal Called by RespanCompositeProcessor before any other end logic.
 * Returns the span (or export-only clone) that must continue through Respan.
 */
export function runSpanTransformersOnEnd(
  span: ReadableSpan,
): ReadableSpan {
  const state = getRegistryState();
  const spanObject = span as object;
  const hasStartSnapshot = state.spanSnapshots.has(spanObject);
  const entries = hasStartSnapshot
    ? state.spanSnapshots.get(spanObject) ?? []
    : sortedActiveEntries(state);

  if (hasStartSnapshot) {
    state.spanSnapshots.delete(spanObject);
  } else {
    // Synthetic spans injected directly into onEnd still participate in the
    // disposal barrier for the duration of their hooks.
    for (const entry of entries) {
      entry.inFlightCount += 1;
    }
  }

  try {
    for (const entry of entries) {
      if (!entry.transformer.onEnd) {
        continue;
      }
      try {
        entry.transformer.onEnd(span);
      } catch (error) {
        reportTransformerError(entry.key, "onEnd", error);
      }
    }

    let exportSpan = span;
    for (const entry of entries) {
      if (!entry.transformer.prepareForExport) {
        continue;
      }
      try {
        const prepared = entry.transformer.prepareForExport(exportSpan);
        if (prepared && typeof prepared === "object") {
          exportSpan = prepared;
        } else {
          reportTransformerError(
            entry.key,
            "prepareForExport",
            new Error("Transformer returned an invalid span."),
          );
        }
      } catch (error) {
        reportTransformerError(entry.key, "prepareForExport", error);
      }
    }

    return exportSpan;
  } finally {
    const hold = hasStartSnapshot
      ? state.inFlightHolds.get(spanObject)
      : undefined;
    if (hold) {
      // Started span: release the hold registered at onStart exactly once and
      // cancel its GC watch (settle decrements inFlightCount and drains).
      state.inFlightHolds.delete(spanObject);
      state.abandonedSpanWatch.unregister(spanObject);
      settleInFlightHold(state, hold);
    } else {
      // Synthetic span (no onStart hold): decrement the counts raised above.
      for (const entry of entries) {
        entry.inFlightCount = Math.max(0, entry.inFlightCount - 1);
        disposeEntryIfReady(state, entry);
      }
    }
  }
}

function getRegistryState(): SpanTransformerRegistryState {
  const registryGlobal = globalThis as Record<PropertyKey, unknown>;
  const existing = registryGlobal[
    RESPAN_SPAN_TRANSFORMER_REGISTRY_SYMBOL
  ] as SpanTransformerRegistryState | undefined;
  if (existing?.version === 1) {
    return existing;
  }

  const state: SpanTransformerRegistryState = {
    entries: new Map(),
    retainedEntries: new Set(),
    spanSnapshots: new WeakMap(),
    inFlightHolds: new WeakMap(),
    abandonedSpanWatch: undefined as unknown as FinalizationRegistry<InFlightHold>,
    hostCount: 0,
    version: 1,
  };
  // The watch callback fires when an abandoned span is collected; it releases
  // that span's drain hold. It is a no-op for spans already settled by onEnd.
  state.abandonedSpanWatch = new FinalizationRegistry<InFlightHold>((hold) => {
    settleInFlightHold(state, hold);
  });
  Object.defineProperty(
    registryGlobal,
    RESPAN_SPAN_TRANSFORMER_REGISTRY_SYMBOL,
    {
      configurable: false,
      enumerable: false,
      value: state,
      writable: false,
    },
  );
  return state;
}

function sortedActiveEntries(
  state: SpanTransformerRegistryState,
): readonly TransformerEntry[] {
  return [...state.entries.values()]
    .filter((entry) => entry.referenceCount > 0)
    // Code-point order, not locale-sensitive localeCompare, so dotted keys like
    // "respan.openai" order identically across ICU/locale environments.
    .sort((left, right) =>
      left.key < right.key ? -1 : left.key > right.key ? 1 : 0,
    );
}

function reportTransformerError(
  key: string,
  phase: "onStart" | "onEnd" | "prepareForExport" | "dispose",
  error: unknown,
): void {
  const message = error instanceof Error ? error.message : String(error);
  // Use console.error (matching RespanCompositeProcessor's postprocess-error
  // path) rather than diag.warn: startTracing installs a diag logger whose
  // warn() is a no-op, which would otherwise swallow every isolated hook
  // failure silently in production.
  console.error(
    `[Respan] Span transformer "${key}" failed during ${phase}; continuing export:`,
    error,
  );
}

function disposeEntryIfReady(
  state: SpanTransformerRegistryState,
  entry: TransformerEntry,
): void {
  if (entry.disposalRequested && entry.inFlightCount === 0) {
    disposeEntry(state, entry);
  }
}

function disposeEntry(
  state: SpanTransformerRegistryState,
  entry: TransformerEntry,
): void {
  if (entry.disposed) {
    return;
  }
  entry.disposed = true;
  state.retainedEntries.delete(entry);
  if (!entry.transformer.dispose) {
    return;
  }
  try {
    entry.transformer.dispose();
  } catch (error) {
    reportTransformerError(entry.key, "dispose", error);
  }
}
