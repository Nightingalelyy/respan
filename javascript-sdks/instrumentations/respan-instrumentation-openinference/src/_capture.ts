import { types } from "node:util";
import {
  context,
  trace,
  TraceFlags,
  type Context,
  type Span,
} from "@opentelemetry/api";
import {
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { isTracingSuppressed } from "@opentelemetry/core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import { RespanSpanAttributes } from "@respan/respan-sdk";

export interface CaptureOptions {
  traceContent?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
}
interface CaptureState {
  owned: boolean;
  blocked?: boolean;
  inputs: boolean;
  outputs: boolean;
  parent?: Span;
  parentContext: Context;
  raw: Record<string, unknown>;
  attributes: Record<string, unknown>;
}
const states = new WeakMap<object, CaptureState>();
const ownProxies = new WeakSet<object>();
function foreignProxy(value: object): boolean {
  return types.isProxy(value) && !ownProxies.has(value);
}
function ownedProxy<T extends object>(target: T, handler: ProxyHandler<T>): T {
  const proxy = new Proxy(target, handler);
  ownProxies.add(proxy);
  return proxy;
}

const guardedGetters = new WeakSet<Function>();
const SAFE = new Set<string>([
  SemanticConventions.OPENINFERENCE_SPAN_KIND,
  SemanticConventions.LLM_MODEL_NAME,
  SemanticConventions.LLM_PROVIDER,
  SemanticConventions.LLM_SYSTEM,
  SemanticConventions.EMBEDDING_MODEL_NAME,
  SemanticConventions.LLM_REQUEST_MODEL_NAME,
  SemanticConventions.LLM_RESPONSE_MODEL_NAME,
  SemanticConventions.AGENT_NAME,
  SemanticConventions.TOOL_NAME,
  SemanticConventions.TOOL_CALL_ID,
  SemanticConventions.TOOL_ID,
  SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
  SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
  SemanticConventions.LLM_TOKEN_COUNT_TOTAL,
  SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
  SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  SpanAttributes.LLM_REQUEST_TYPE,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  SpanAttributes.TRACELOOP_WORKFLOW_NAME,
  RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN,
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID,
  RespanSpanAttributes.RESPAN_SESSION_ID,
  RespanSpanAttributes.RESPAN_THREADS_ID,
  RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
  RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  "allow_trace_content",
]);
function valueOf(object: unknown, key: string): unknown {
  if (!object || typeof object !== "object" || foreignProxy(object))
    return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(object, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}
function primitive(value: unknown): boolean {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}
function nativeValue(value: unknown): unknown {
  if (primitive(value)) return value;
  if (!Array.isArray(value) || foreignProxy(value)) return undefined;
  const result: unknown[] = [];
  for (let i = 0; i < value.length; i++) {
    const item = valueOf(value, String(i));
    if (!primitive(item)) return undefined;
    result.push(item);
  }
  return result;
}
function falseFlag(span: unknown): boolean {
  if (!span || typeof span !== "object") return false;
  const descriptor = Object.getOwnPropertyDescriptor(span, "attributes");
  let attrs: unknown;
  if (descriptor && "value" in descriptor) attrs = descriptor.value;
  else if (descriptor?.get && guardedGetters.has(descriptor.get))
    attrs = descriptor.get.call(span);
  return valueOf(attrs, "allow_trace_content") === false;
}
function contextDenied(ctx: Context): boolean {
  return ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT as never) === false;
}
function observe(state: CaptureState, ambient = true): void {
  if (
    state.attributes.allow_trace_content === false ||
    falseFlag(state.parent) ||
    contextDenied(state.parentContext) ||
    (ambient && contextDenied(context.active())) ||
    process.env.RESPAN_TRACE_CONTENT === "false" ||
    process.env.TRACELOOP_TRACE_CONTENT === "false"
  ) {
    state.inputs = false;
    state.outputs = false;
  }
  if (state.owned && (!state.inputs || !state.outputs)) {
    for (const key of Object.keys(state.raw))
      if (
        !allowed(state, key) ||
        (SAFE.has(key) && !primitive(valueOf(state.raw, key)))
      )
        delete state.raw[key];
    for (const key of Object.keys(state.attributes))
      if (
        !allowed(state, key) ||
        (SAFE.has(key) && !primitive(valueOf(state.attributes, key)))
      )
        delete state.attributes[key];
  }
}
function allowed(state: CaptureState, key: string): boolean {
  if (SAFE.has(key)) return true;
  const input =
    key === SemanticConventions.INPUT_VALUE ||
    key === SemanticConventions.INPUT_MIME_TYPE ||
    key.startsWith(`${SemanticConventions.LLM_INPUT_MESSAGES}.`) ||
    key.startsWith(`${SemanticConventions.LLM_TOOLS}.`) ||
    key === SemanticConventions.LLM_TOOLS ||
    key === SemanticConventions.LLM_INVOCATION_PARAMETERS ||
    key === SemanticConventions.TOOL_PARAMETERS ||
    (key.startsWith(`${SemanticConventions.EMBEDDING_EMBEDDINGS}.`) &&
      key.endsWith(`.${SemanticConventions.EMBEDDING_TEXT}`)) ||
    key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
    key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
    key.startsWith(`${ATTR_GEN_AI_PROMPT}.`);
  const output =
    key === SemanticConventions.OUTPUT_VALUE ||
    key === SemanticConventions.OUTPUT_MIME_TYPE ||
    key.startsWith(`${SemanticConventions.LLM_OUTPUT_MESSAGES}.`) ||
    (key.startsWith(`${SemanticConventions.EMBEDDING_EMBEDDINGS}.`) &&
      key.endsWith(`.${SemanticConventions.EMBEDDING_VECTOR}`)) ||
    key.startsWith(`${SemanticConventions.RETRIEVAL_DOCUMENTS}.`) ||
    key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
    key.startsWith(`${ATTR_GEN_AI_COMPLETION}.`);
  if (input) return state.inputs;
  if (output) return state.outputs;
  return state.inputs && state.outputs;
}
function openInferenceKey(key: string): boolean {
  return (
    key === SemanticConventions.OPENINFERENCE_SPAN_KIND ||
    key === SemanticConventions.LLM_MODEL_NAME ||
    key === SemanticConventions.LLM_PROVIDER ||
    key === SemanticConventions.LLM_INVOCATION_PARAMETERS ||
    key === SemanticConventions.EMBEDDING_MODEL_NAME ||
    key.startsWith(`${SemanticConventions.LLM_INPUT_MESSAGES}.`) ||
    key.startsWith(`${SemanticConventions.LLM_OUTPUT_MESSAGES}.`) ||
    key.startsWith(`${SemanticConventions.EMBEDDING_EMBEDDINGS}.`) ||
    key.startsWith(`${SemanticConventions.RETRIEVAL_DOCUMENTS}.`) ||
    key.startsWith(`${SemanticConventions.LLM_TOOLS}.`)
  );
}
function recognize(state: CaptureState, attrs: unknown): void {
  if (
    attrs &&
    typeof attrs === "object" &&
    !foreignProxy(attrs) &&
    Object.keys(attrs).some(openInferenceKey)
  )
    state.owned = true;
}
function assign(
  state: CaptureState,
  key: string,
  value: unknown,
  ambient = true,
): boolean {
  if (
    key === SemanticConventions.OPENINFERENCE_SPAN_KIND &&
    typeof value === "string"
  )
    state.owned = true;
  if (key === "allow_trace_content" && value === false) {
    state.inputs = false;
    state.outputs = false;
  }
  observe(state, ambient);
  if (state.owned && !allowed(state, key)) return false;
  if (
    state.owned &&
    (!state.inputs || !state.outputs) &&
    SAFE.has(key) &&
    !primitive(value)
  )
    return false;
  const accepted = nativeValue(value);
  if (accepted === undefined) return false;
  state.raw[key] = accepted;
  return true;
}
function sanitize(
  state: CaptureState,
  attrs: unknown,
  ambient = true,
): Record<string, unknown> {
  recognize(state, attrs);
  if (valueOf(attrs, "allow_trace_content") === false) {
    state.inputs = false;
    state.outputs = false;
  }
  observe(state, ambient);
  const result: Record<string, unknown> = {};
  if (!attrs || typeof attrs !== "object" || foreignProxy(attrs)) return result;
  for (const key of Object.keys(attrs)) {
    if (state.owned && !allowed(state, key)) continue;
    const descriptor = Object.getOwnPropertyDescriptor(attrs, key);
    if (!descriptor || !("value" in descriptor)) continue;
    if (assign(state, key, descriptor.value, ambient))
      result[key] = state.raw[key];
  }
  return result;
}
function installProperty(
  span: any,
  key: string,
  get: () => unknown,
  set: (value: unknown) => void,
  deny: () => void,
): void {
  const descriptor = Object.getOwnPropertyDescriptor(span, key);
  if (descriptor?.configurable === false) {
    if ("value" in descriptor && descriptor.writable) {
      // A locked but writable native data property can hold the guarded proxy.
      Object.defineProperty(span, key, { value: get(), writable: false });
    } else {
      // Immutable outer properties cannot host a guard. Scrub their native data
      // in place, seal future writes, and fail closed at export.
      deny();
      const actual = "value" in descriptor ? descriptor.value : undefined;
      if (actual && typeof actual === "object" && !foreignProxy(actual)) {
        const clean = get();
        for (const field of Object.keys(actual))
          Reflect.deleteProperty(actual, field);
        if (Array.isArray(actual)) actual.length = 0;
        if (clean && typeof clean === "object")
          for (const field of Object.keys(clean)) {
            const value = valueOf(clean, field);
            if (primitive(value))
              try {
                Object.defineProperty(actual, field, {
                  value,
                  enumerable: true,
                  configurable: true,
                  writable: true,
                });
              } catch {}
          }
        Object.freeze(actual);
      }
    }
    return;
  }
  guardedGetters.add(get);
  Object.defineProperty(span, key, {
    enumerable: true,
    configurable: false,
    get,
    set,
  });
}
/** Runs after the native sampler, before OpenInference writes its attributes. */
export function guardSpan(
  span: Span,
  parentContext: Context,
  options: CaptureOptions,
  admission?: CaptureState,
): void {
  if (states.has(span as object)) return;
  const native = span as any;
  const attrs = valueOf(native, "attributes") as
    Record<string, unknown> | undefined;
  if (!attrs) return;
  const admitted =
    span.isRecording() &&
    (span.spanContext().traceFlags & TraceFlags.SAMPLED) !== 0;
  const state: CaptureState = {
    owned:
      admission?.owned ??
      typeof valueOf(attrs, SemanticConventions.OPENINFERENCE_SPAN_KIND) ===
        "string",
    inputs:
      admitted &&
      options.traceContent !== false &&
      options.recordInputs !== false &&
      !isTracingSuppressed(parentContext),
    outputs:
      admitted &&
      options.traceContent !== false &&
      options.recordOutputs !== false &&
      !isTracingSuppressed(parentContext),
    parent: trace.getSpan(parentContext),
    parentContext,
    raw: {},
    attributes: attrs,
  };
  if (admission) {
    state.owned = admission.owned;
    state.inputs = admission.inputs;
    state.outputs = admission.outputs;
    state.parent = admission.parent;
    state.parentContext = admission.parentContext;
    state.blocked = admission.blocked;
  }
  recognize(state, attrs);
  states.set(span as object, state);
  const setAttribute = span.setAttribute;
  const setAttributes = span.setAttributes;
  const addEvent = span.addEvent;
  const recordException = span.recordException;
  const setStatus = span.setStatus;
  let protectedSpan = false;
  function protect(): void {
    if (protectedSpan || !state.owned) return;
    protectedSpan = true;
    const deny = () => {
      state.blocked = true;
      state.inputs = state.outputs = false;
    };
    state.attributes = sanitize(state, state.attributes);
    const makeProxy = () =>
      ownedProxy(state.attributes, {
        get(target, key, receiver) {
          observe(state);
          return Reflect.get(target, key, receiver);
        },
        set(target, key, value) {
          if (typeof key === "string" && assign(state, key, value))
            target[key] = state.raw[key];
          return true;
        },
        defineProperty(target, key, descriptor) {
          if (
            typeof key === "string" &&
            "value" in descriptor &&
            assign(state, key, descriptor.value)
          ) {
            Object.defineProperty(target, key, {
              ...descriptor,
              value: state.raw[key],
              configurable: true,
            });
          }
          return true;
        },
      });
    let proxy = makeProxy();
    installProperty(
      native,
      "attributes",
      () => {
        observe(state);
        return proxy;
      },
      (value) => {
        state.attributes = sanitize(state, value);
        proxy = makeProxy();
      },
      deny,
    );
    let events: any[] = Array.isArray(valueOf(native, "events"))
      ? (valueOf(native, "events") as any[])
      : [];
    installProperty(
      native,
      "events",
      () => {
        observe(state);
        return !state.inputs || !state.outputs ? [] : events;
      },
      (value) => {
        observe(state);
        if (state.inputs && state.outputs)
          events = Array.isArray(value) ? value : [];
      },
      deny,
    );
    let status: any = valueOf(native, "status") ?? { code: 0 };
    const safeStatus = (value: unknown) => {
      const code = valueOf(value, "code");
      return { code: code === 1 || code === 2 ? code : 0 };
    };
    installProperty(
      native,
      "status",
      () => {
        observe(state);
        if (!state.inputs || !state.outputs) status = safeStatus(status);
        return ownedProxy(status, {
          set(target, key, value) {
            observe(state);
            if ((!state.inputs || !state.outputs) && key !== "code")
              return true;
            if (key === "code" && value !== 0 && value !== 1 && value !== 2)
              return true;
            target[key] = value;
            return true;
          },
        });
      },
      (value) => {
        observe(state);
        status = !state.inputs || !state.outputs ? safeStatus(value) : value;
      },
      deny,
    );
    span.addEvent = function (this: Span, ...args: any[]) {
      observe(state);
      if (state.inputs && state.outputs) (addEvent as any).apply(this, args);
      return this;
    } as typeof span.addEvent;
    span.recordException = function (...args: any[]) {
      observe(state);
      if (state.inputs && state.outputs)
        (recordException as any).apply(this, args);
    };
    span.setStatus = function (value) {
      observe(state);
      if (!state.blocked)
        setStatus.call(
          this,
          !state.inputs || !state.outputs ? safeStatus(value) : value,
        );
      return this;
    };
  }
  span.setAttribute = function (key, value) {
    if (openInferenceKey(key)) state.owned = true;
    if (key === "allow_trace_content" && value === false) {
      state.inputs = false;
      state.outputs = false;
    }
    if (!state.owned) return setAttribute.call(this, key, value);
    protect();
    if (assign(state, key, value) && !state.blocked)
      setAttribute.call(this, key, state.raw[key] as never);
    return this;
  };
  span.setAttributes = function (values) {
    recognize(state, values);
    if (!state.owned) return setAttributes.call(this, values);
    protect();
    const filtered = sanitize(state, values);
    if (!state.blocked) setAttributes.call(this, filtered as never);
    return this;
  };
  protect();
}
export function capturedAttributes(
  span: object,
): Record<string, unknown> | undefined {
  const state = states.get(span);
  if (!state?.owned) return undefined;
  // Exporter ambient suppression is deliberately irrelevant to prior admission.
  observe(state, false);
  return { ...state.raw, ...sanitize(state, state.attributes, false) };
}
export function captureLanes(span: object): {
  inputs: boolean;
  outputs: boolean;
} {
  const state = states.get(span);
  if (!state) return { inputs: true, outputs: true };
  observe(state, false);
  return { inputs: state.inputs, outputs: state.outputs };
}
export function safeDescriptorValue(object: unknown, key: string): unknown {
  return valueOf(object, key);
}

export function guardExportClone(clone: Span, source: object): void {
  const admission = states.get(source);
  if (admission) guardSpan(clone, admission.parentContext, {}, admission);
}

export function captureBlocked(span: object): boolean {
  return states.get(span)?.blocked === true;
}
