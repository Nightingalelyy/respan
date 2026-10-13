import {
  context,
  trace,
  TraceFlags,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { data, primitive, type Row } from "./_data.js";
import { types } from "node:util";

const GUARDS = Symbol.for("respan.codex-sdk.content-guards.v1");
const globalGuards = globalThis as typeof globalThis & {
  [GUARDS]?: WeakMap<object, ContentGuard>;
};
const guards = (globalGuards[GUARDS] ??= new WeakMap<object, ContentGuard>());
const SAFE = new Set([
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  RespanSpanAttributes.RESPAN_THREADS_ID,
  RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL,
  "allow_trace_content",
]);

export class ContentGuard {
  private ceiling: boolean;
  private readonly sampled: boolean;
  private readonly parent?: Span;
  private readonly capturedContext: Context;
  private readonly finalizers: Array<() => void> = [];
  private exportable = true;
  private internalWrites = 0;
  private rawAttributes?: unknown;
  private safeMetadata?: string;

  constructor(
    private readonly span: Span,
    traceContent: boolean,
    parent: Context,
    inherited?: ContentGuard,
  ) {
    this.capturedContext = parent;
    this.parent = trace.getSpan(parent);
    this.ceiling =
      traceContent &&
      process.env.RESPAN_TRACE_CONTENT !== "false" &&
      process.env.TRACELOOP_TRACE_CONTENT !== "false" &&
      parent.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) !== false &&
      !isTracingSuppressed(parent) &&
      (inherited?.captureAllowed() ?? true);
    this.sampled =
      span.isRecording() &&
      Boolean(span.spanContext().traceFlags & TraceFlags.SAMPLED);
    guards.set(span, this);
    this.captureAllowed();
    this.install();
  }

  captureAllowed(): boolean {
    if (
      data(
        data(this.span, "attributes") ?? this.rawAttributes,
        "allow_trace_content",
      ) === false
    )
      this.ceiling = false;
    const parentGuard = this.parent ? guards.get(this.parent) : undefined;
    if (
      this.capturedContext.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) ===
        false ||
      (parentGuard
        ? !parentGuard.captureAllowed()
        : data(data(this.parent, "attributes"), "allow_trace_content") ===
          false)
    )
      this.ceiling = false;
    return this.ceiling;
  }

  allowed(): boolean {
    return this.sampled && this.captureAllowed();
  }

  prepare(): boolean {
    for (const finalize of this.finalizers) finalize();
    return this.exportable;
  }

  internal<T>(fn: () => T): T {
    this.internalWrites++;
    try {
      return fn();
    } finally {
      this.internalWrites--;
    }
  }

  metadata(value: Row): void {
    this.safeMetadata = JSON.stringify(value);
    let output = this.safeMetadata;
    const previous = data(
      this.rawAttributes,
      RespanSpanAttributes.RESPAN_METADATA,
    );
    if (this.allowed() && typeof previous === "string") {
      try {
        const parsed = JSON.parse(previous);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
          output = JSON.stringify({ ...parsed, ...value });
      } catch {
        /* Preserve only valid canonical JSON metadata. */
      }
    }
    this.internal(() =>
      this.span.setAttribute(RespanSpanAttributes.RESPAN_METADATA, output!),
    );
  }

  private observeWrite(): void {
    if (this.internalWrites === 0) this.observe();
  }

  observe(ctx: Context = context.active(), attrs?: unknown): void {
    this.captureAllowed();
    if (
      process.env.RESPAN_TRACE_CONTENT === "false" ||
      process.env.TRACELOOP_TRACE_CONTENT === "false" ||
      ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) === false ||
      isTracingSuppressed(ctx) ||
      data(attrs, "allow_trace_content") === false
    )
      this.ceiling = false;
  }

  private filter(attrs: unknown): Row {
    const result: Row = {};
    if (!attrs || typeof attrs !== "object" || types.isProxy(attrs))
      return result;
    const allowed = this.allowed();
    for (const key of Object.keys(attrs)) {
      if (
        !allowed &&
        !SAFE.has(key) &&
        key !== RespanSpanAttributes.RESPAN_METADATA
      )
        continue;
      const value = data(attrs, key);
      if (
        !allowed &&
        key === RespanSpanAttributes.RESPAN_METADATA &&
        value !== this.safeMetadata
      )
        continue;
      if (
        primitive(value) ||
        (allowed && Array.isArray(value) && value.every(primitive))
      )
        result[key] = value;
    }
    if (!this.captureAllowed()) result.allow_trace_content = false;
    return result;
  }

  private install(): void {
    const target = this.span as Span & Row;
    const setAttribute = target.setAttribute.bind(target);
    const setAttributes = target.setAttributes.bind(target);
    const addEvent = target.addEvent.bind(target);
    const recordException = target.recordException.bind(target);
    const setStatus = target.setStatus.bind(target);
    target.setAttribute = (key, value) => {
      this.observeWrite();
      if (key === "allow_trace_content" && value === false)
        this.ceiling = false;
      if (
        (this.allowed() ||
          SAFE.has(key) ||
          (key === RespanSpanAttributes.RESPAN_METADATA &&
            value === this.safeMetadata)) &&
        (primitive(value) ||
          (this.allowed() && Array.isArray(value) && value.every(primitive)))
      )
        setAttribute(key, value);
      return target;
    };
    target.setAttributes = (attrs) => {
      this.observeWrite();
      if (data(attrs, "allow_trace_content") === false) this.ceiling = false;
      setAttributes(this.filter(attrs) as Parameters<Span["setAttributes"]>[0]);
      return target;
    };
    target.addEvent = (...args: any[]) => {
      this.observeWrite();
      if (this.allowed()) (addEvent as any)(...args);
      return target;
    };
    target.recordException = (...args: any[]) => {
      this.observeWrite();
      if (this.allowed()) (recordException as any)(...args);
    };
    target.setStatus = (status) => {
      this.observeWrite();
      const code = data(status, "code");
      const clean: any = { code: code === 1 || code === 2 ? code : 0 };
      const message = this.allowed() ? data(status, "message") : undefined;
      if (typeof message === "string") clean.message = message;
      setStatus(clean);
      return target;
    };
    // Guard the actual RecordingSpan/ReadableSpan fields as well as API methods.
    for (const key of ["attributes", "events", "status"]) {
      const descriptor = Object.getOwnPropertyDescriptor(target, key);
      if (!descriptor || !("value" in descriptor)) continue;
      let value = descriptor.value;
      if (key === "attributes") this.rawAttributes = value;
      const sanitize = () => {
        if (!descriptor.configurable)
          value = Object.getOwnPropertyDescriptor(target, key)?.value;
        if (key === "attributes") {
          if (data(value, "allow_trace_content") === false)
            this.ceiling = false;
          if (!this.allowed()) {
            const clean = this.filter(value);
            value = clean;
            this.rawAttributes = value;
          }
        } else if (key === "events") {
          if (!this.allowed()) value = [];
        } else {
          const code = data(value, "code");
          const message = this.allowed() ? data(value, "message") : undefined;
          value = {
            code: code === 1 || code === 2 ? code : 0,
            ...(typeof message === "string" ? { message } : {}),
          };
        }
        if (!descriptor.configurable) {
          if (descriptor.writable) target[key] = value;
          else if (!this.allowed()) {
            const locked = descriptor.value;
            for (const field of Object.keys(locked ?? {})) {
              const clean = data(value, field);
              if (clean === data(locked, field) && primitive(clean)) continue;
              if (Object.getOwnPropertyDescriptor(locked, field)?.configurable)
                delete locked[field];
              else this.exportable = false;
            }
          }
        }
        return value;
      };
      this.finalizers.push(sanitize);
      if (descriptor.configurable)
        Object.defineProperty(target, key, {
          configurable: false,
          enumerable: descriptor.enumerable,
          get: sanitize,
          set: (next) => {
            this.observeWrite();
            value = key === "attributes" ? this.filter(next) : next;
            if (key === "attributes") this.rawAttributes = value;
            sanitize();
          },
        });
      else sanitize();
    }
  }
}
