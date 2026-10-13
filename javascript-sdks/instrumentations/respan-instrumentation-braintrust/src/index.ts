import {
  context,
  trace,
  SpanStatusCode,
  type Context,
  type Span,
  type SpanContext,
} from "@opentelemetry/api";
import { getPropagatedAttributes } from "@respan/tracing";
import {
  data,
  merge,
  primitive,
  snapshot,
  string,
  time,
  type Row,
} from "./_data.js";
import {
  attributes,
  logType,
  recordSnapshot,
  safeAttributes,
  usage,
} from "./_translator.js";
import { ContentGuard } from "./_privacy.js";
import { maskMergedRow } from "./_masking.js";

const INSTRUMENTATION_NAME = "@respan/instrumentation-braintrust";
const VERSION = "0.1.0";
const REGISTRY = Symbol.for("respan.braintrust.native-bridge.v1");

export interface BraintrustExportOptions {
  activeSpanContext?: SpanContext;
  propagatedAttributes?: Row;
}

export interface BraintrustInstrumentorOptions {
  includeBraintrustRawMetadata?: boolean;
  /** Capture payloads only when the real OTel sampler admits a sampled span. */
  traceContent?: boolean;
}

interface NativeState {
  bgLogger(): BackgroundLogger;
  setOverrideBgLogger(logger: BackgroundLogger | null): void;
}

interface BackgroundLogger {
  log(items: unknown[]): void;
  flush(): Promise<void>;
  pendingFlushBytes(): number;
  flushBackpressureBytes(): number;
  setMaskingFunction(fn: ((value: unknown) => unknown) | null): void;
}

interface Registry {
  entries: Map<NativeState, Bridge>;
  prototypes: Map<object, { original: Function; wrapper: Function }>;
  factories: WeakMap<
    BraintrustInstrumentor,
    {
      create: (
        row: Row,
        parent: Context,
        propagated?: Row,
        inherited?: ContentGuard,
        traceContent?: boolean,
      ) => Capture;
      traceContent: boolean;
    }
  >;
}

const globals = globalThis as typeof globalThis & { [REGISTRY]?: Registry };
const registry = (globals[REGISTRY] ??= {
  entries: new Map(),
  prototypes: new Map(),
  factories: new WeakMap(),
});

/** Converts actual Braintrust global-logger spans into the active OTel pipeline. */
export class BraintrustInstrumentor {
  readonly name = "braintrust";
  private readonly options: Required<BraintrustInstrumentorOptions>;
  private requested = false;
  private enabled = true;
  private activating?: Promise<void>;
  private bridge?: Bridge;
  private readonly contexts = new Map<string, SpanContext>();

  constructor(options: BraintrustInstrumentorOptions = {}) {
    this.options = {
      includeBraintrustRawMetadata:
        options.includeBraintrustRawMetadata ?? false,
      traceContent: options.traceContent ?? true,
    };
    registry.factories.set(this, {
      create: this.start.bind(this),
      traceContent: this.options.traceContent,
    });
  }

  isActive(): boolean {
    return this.requested && Boolean(this.bridge);
  }

  activate(): Promise<void> {
    this.requested = true;
    this.enabled = true;
    if (this.bridge) {
      this.bridge.owners.add(this);
      return Promise.resolve();
    }
    if (this.activating) return this.activating;
    const operation = this.install();
    this.activating = operation;
    void operation
      .finally(() => {
        if (this.activating === operation) this.activating = undefined;
      })
      .catch(() => {});
    return operation;
  }

  private async install(): Promise<void> {
    const native = await import("braintrust");
    if (!this.requested) return;
    const state = (native as any)._internalGetGlobalState?.() as
      NativeState | undefined;
    const prototype = (native as any).SpanImpl?.prototype;
    if (
      !state ||
      typeof state.setOverrideBgLogger !== "function" ||
      typeof prototype?.logInternal !== "function"
    )
      throw new Error(
        "Braintrust does not expose the supported native logger/span bridge",
      );
    let bridge = registry.entries.get(state);
    if (!bridge) {
      bridge = new Bridge(
        state,
        state.bgLogger(),
        data(state, "_overrideBgLogger") ?? null,
      );
      registry.entries.set(state, bridge);
      state.setOverrideBgLogger(bridge);
    }
    if (!registry.prototypes.has(prototype)) {
      const original = prototype.logInternal;
      const wrapper = function (this: Row, args: Row) {
        const current = registry.entries.get(data(this, "_state"));
        if (!current) return original.call(this, args);
        const capture = current.capture(this, args);
        const previous = current.current;
        current.current = capture;
        try {
          return original.call(this, args);
        } finally {
          current.current = previous;
        }
      };
      prototype.logInternal = wrapper;
      registry.prototypes.set(prototype, { original, wrapper });
    }
    this.bridge = bridge;
    bridge.owners.add(this);
  }

  async deactivate(): Promise<void> {
    this.requested = false;
    this.enabled = false;
    await this.activating;
    const bridge = this.bridge;
    this.bridge = undefined;
    if (!bridge) return;
    bridge.owners.delete(this);
    await bridge.flush();
    bridge.restoreIfIdle();
  }

  async flush(): Promise<void> {
    await this.bridge?.flush();
  }

  /** Explicit imported-row API retained for existing callers. Native examples use real SDK methods. */
  exportRecord(record: Row, options: BraintrustExportOptions = {}): boolean {
    if (!this.enabled || !primitive(data(record, "id"))) return false;
    let parent = context.active();
    if (options.activeSpanContext)
      parent = trace.setSpanContext(parent, options.activeSpanContext);
    const capture = this.start(record, parent, options.propagatedAttributes);
    capture.updateSafe(record);
    if (capture.guard.allowed())
      capture.row = merge(capture.row, recordSnapshot(record));
    capture.finish();
    return (
      capture.span.isRecording() ||
      Boolean(capture.span.spanContext().traceFlags & 1)
    );
  }

  exportRecords(records: Row[], options: BraintrustExportOptions = {}): number {
    const groups = new Map<string, Capture>();
    for (const record of records) {
      const id = data(record, "id");
      if (!primitive(id)) throw new Error("Braintrust row is missing an id");
      if (!this.enabled) continue;
      const key = String(id);
      let capture = groups.get(key);
      if (!capture) {
        let parent = context.active();
        if (options.activeSpanContext)
          parent = trace.setSpanContext(parent, options.activeSpanContext);
        capture = this.start(record, parent, options.propagatedAttributes);
        groups.set(key, capture);
      }
      capture.updateSafe(record);
      if (capture.guard.allowed())
        capture.row = merge(capture.row, recordSnapshot(record));
    }
    for (const capture of groups.values()) capture.finish();
    return [...groups.values()].filter((capture) =>
      Boolean(capture.span.spanContext().traceFlags & 1),
    ).length;
  }

  private start(
    row: Row,
    parent: Context,
    propagated?: Row,
    inherited?: ContentGuard,
    traceContent = true,
  ): Capture {
    const parents = data(row, "span_parents");
    const parentId = Array.isArray(parents)
      ? string(data(parents, String(parents.length - 1)))
      : string(data(row, "_parent_id"));
    const nativeParent = parentId ? this.contexts.get(parentId) : undefined;
    if (nativeParent) parent = trace.setSpanContext(parent, nativeParent);
    const safe = safeAttributes(row);
    const span = trace.getTracer(INSTRUMENTATION_NAME, VERSION).startSpan(
      string(data(data(row, "span_attributes"), "name")) ?? "braintrust.span",
      {
        attributes: safe,
        startTime:
          time(data(data(row, "metrics"), "start")) ??
          time(data(row, "created")),
      },
      parent,
    );
    const guard = new ContentGuard(
      span,
      this.options.traceContent && traceContent,
      parent,
      inherited,
    );
    const id = string(data(row, "span_id"));
    if (id) this.contexts.set(id, span.spanContext());
    const capture = new Capture(
      this.options.includeBraintrustRawMetadata,
      span,
      guard,
      propagated ?? (getPropagatedAttributes(parent) as Row | undefined),
    );
    capture.updateSafe(row);
    return capture;
  }
}

export { BraintrustInstrumentor as RespanBraintrustInstrumentor };

class Capture {
  row: Row = {};
  ended = false;
  private finished = false;
  private error = false;
  private endTime?: number;
  private readonly propagated: Row;

  constructor(
    private readonly includeRaw: boolean,
    readonly span: Span,
    readonly guard: ContentGuard,
    propagated?: Row,
  ) {
    this.propagated = guard.allowed() ? snapshot(propagated ?? {}) : {};
  }

  updateSafe(row: Row): void {
    const spanAttrs = data(row, "span_attributes");
    this.guard.observe(context.active(), spanAttrs);
    for (const key of ["id", "span_id", "root_span_id", "created"]) {
      const value = data(row, key);
      if (primitive(value) && this.row[key] === undefined)
        this.row[key] = value;
    }
    this.row.span_attributes ??= {};
    for (const key of ["name", "type", "model", "provider"]) {
      const value = data(spanAttrs, key);
      if (typeof value === "string") this.row.span_attributes[key] = value;
    }
    const metadata = data(row, "metadata");
    this.row.metadata ??= {};
    for (const key of ["model", "provider"]) {
      const value = data(metadata, key);
      if (typeof value === "string") this.row.metadata[key] = value;
    }
    const metrics = data(row, "metrics");
    this.row.metrics ??= {};
    for (const key of [
      "start",
      "end",
      "prompt_tokens",
      "input_tokens",
      "completion_tokens",
      "output_tokens",
      "tokens",
      "total_tokens",
      "cache_read_input_tokens",
    ]) {
      const value = data(metrics, key);
      if (typeof value === "number" && Number.isFinite(value))
        this.row.metrics[key] = value;
    }
    if (typeof data(metrics, "end") === "number") this.ended = true;
    if (typeof data(metrics, "end") === "number")
      this.endTime = time(data(metrics, "end"));
    if (data(row, "error") !== undefined && data(row, "error") !== null)
      this.error = true;
  }

  finish(): void {
    if (this.finished) return;
    this.finished = true;
    this.guard.internal(() => {
      this.span.setAttributes(safeAttributes(this.row));
      if (["chat", "embedding"].includes(logType(this.row)))
        this.span.setAttributes(usage(this.row));
      if (this.guard.allowed())
        this.span.setAttributes(
          attributes(this.row, this.propagated, this.includeRaw),
        );
      if (
        this.error ||
        (this.row.error !== undefined && this.row.error !== null)
      ) {
        const error = this.row.error;
        const message = this.guard.allowed()
          ? (string(error) ?? string(data(error, "message")))
          : undefined;
        this.span.setStatus({
          code: SpanStatusCode.ERROR,
          ...(message === undefined ? {} : { message }),
        });
      }
      if (this.guard.prepare()) this.span.end(this.endTime);
    });
  }
}

class Bridge implements BackgroundLogger {
  readonly owners = new Set<BraintrustInstrumentor>();
  readonly captures = new WeakMap<object, Capture>();
  readonly inflight = new Set<Capture>();
  readonly nativeCaptures = new Map<string, Capture>();
  current?: Capture;
  private queue: Array<{ item: any; capture: Capture }> = [];
  private flushing: Promise<void> = Promise.resolve();

  constructor(
    readonly state: NativeState,
    private readonly previous: BackgroundLogger,
    private readonly previousOverride: BackgroundLogger | null,
  ) {}

  capture(native: Row, args: Row): Capture | undefined {
    let capture = this.captures.get(native);
    const internal = data(args, "internalData");
    const event = data(args, "event");
    if (!capture) {
      const owner = this.owners.values().next().value;
      if (!owner) return undefined;
      const row: Row = {
        id: data(native, "_id"),
        span_id: data(native, "_spanId"),
        root_span_id: data(native, "_rootSpanId"),
        span_parents: data(native, "_spanParents"),
        created: data(internal, "created"),
        span_attributes: data(internal, "span_attributes"),
        metadata: data(event, "metadata"),
        metrics: data(internal, "metrics"),
      };
      const parents = row.span_parents;
      const parentId = Array.isArray(parents)
        ? string(data(parents, String(parents.length - 1)))
        : undefined;
      const parent = parentId ? this.nativeCaptures.get(parentId) : undefined;
      capture = registry.factories.get(owner)!.create(
        row,
        parent
          ? trace.setSpan(context.active(), parent.span)
          : context.active(),
        undefined,
        parent?.guard,
        [...this.owners].every(
          (active) => registry.factories.get(active)?.traceContent !== false,
        ),
      );
      this.captures.set(native, capture);
      this.inflight.add(capture);
      if (typeof row.span_id === "string")
        this.nativeCaptures.set(row.span_id, capture);
    }
    capture.updateSafe(internal ?? {});
    capture.updateSafe(event ?? {});
    return capture;
  }

  log(items: unknown[]): void {
    const capture = this.current;
    if (!capture) {
      if (this.owners.size === 0) this.previous.log(items);
      // Feedback and remote update rows are not a new native span lifecycle.
      return;
    }
    if (capture.guard.allowed())
      for (const item of items) this.queue.push({ item, capture });
  }

  flush(): Promise<void> {
    const operation = this.flushing
      .catch(() => {})
      .then(async () => {
        while (this.queue.length) {
          const batch = this.queue;
          this.queue = [];
          for (const { item, capture } of batch) {
            if (!capture.guard.allowed()) continue;
            const resolved =
              typeof item === "function"
                ? await item()
                : typeof item?.get === "function"
                  ? await item.get()
                  : item;
            if (
              !capture.guard.allowed() ||
              !resolved ||
              typeof resolved !== "object"
            )
              continue;
            capture.row = merge(capture.row, recordSnapshot(resolved));
          }
        }
        for (const capture of this.inflight) {
          if (!capture.ended) continue;
          if (capture.guard.allowed())
            capture.row = recordSnapshot(
              maskMergedRow(this.previous, capture.row),
            );
          capture.finish();
          this.inflight.delete(capture);
        }
        this.restoreIfIdle();
      });
    this.flushing = operation;
    return operation;
  }

  pendingFlushBytes(): number {
    return this.queue.length;
  }
  flushBackpressureBytes(): number {
    return Infinity;
  }
  setMaskingFunction(fn: ((value: unknown) => unknown) | null): void {
    this.previous.setMaskingFunction(fn);
  }

  restoreIfIdle(): void {
    if (this.owners.size || this.inflight.size || this.queue.length) return;
    if (this.state.bgLogger() === this)
      this.state.setOverrideBgLogger(this.previousOverride);
    registry.entries.delete(this.state);
    if (registry.entries.size) return;
    for (const [prototype, patch] of registry.prototypes) {
      if ((prototype as Row).logInternal === patch.wrapper)
        (prototype as Row).logInternal = patch.original;
    }
    registry.prototypes.clear();
  }
}
