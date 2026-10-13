import { types } from "node:util";
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
import {
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_REQUEST_TOP_K,
} from "@opentelemetry/semantic-conventions/incubating";

export interface CaptureOptions {
  traceContent?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
}
export interface Capture {
  inputs: boolean;
  outputs: boolean;
  observe(ambient?: boolean): void;
}
const captures = new WeakMap<object, Capture>();
const getters = new WeakSet<Function>();
const safeKeys = new Set<string>([
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  SpanAttributes.TRACELOOP_WORKFLOW_NAME,
  SpanAttributes.LLM_REQUEST_TYPE,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  SpanAttributes.LLM_FREQUENCY_PENALTY,
  SpanAttributes.LLM_PRESENCE_PENALTY,
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID,
  RespanSpanAttributes.RESPAN_SESSION_ID,
  RespanSpanAttributes.RESPAN_THREADS_ID,
  RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
  RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_REQUEST_TOP_K,
  "allow_trace_content",
  "status_code",
]);
export function dataValue(object: unknown, key: string): any {
  if (!object || typeof object !== "object" || types.isProxy(object))
    return undefined;
  const d = Object.getOwnPropertyDescriptor(object, key);
  return d && "value" in d ? d.value : undefined;
}
function scalar(value: unknown): boolean {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}
function nativeValue(value: unknown): boolean {
  return scalar(value) || (Array.isArray(value) && value.every(scalar));
}
function attributesOf(span: any): any {
  if (!span || typeof span !== "object") return undefined;
  const d = Object.getOwnPropertyDescriptor(span, "attributes");
  return d && "value" in d
    ? d.value
    : d?.get && getters.has(d.get)
      ? d.get.call(span)
      : undefined;
}
/** Actual span guard, installed only after its native sampler has decided. */
export function protect(
  span: Span,
  parentContext: Context,
  options: CaptureOptions,
): Capture {
  const prior = captures.get(span as object);
  if (prior) return prior;
  const admitted =
    span.isRecording() &&
    (span.spanContext().traceFlags & TraceFlags.SAMPLED) !== 0 &&
    !isTracingSuppressed(parentContext);
  const state: Capture = {
    inputs:
      admitted &&
      options.traceContent !== false &&
      options.recordInputs !== false,
    outputs:
      admitted &&
      options.traceContent !== false &&
      options.recordOutputs !== false,
    observe,
  };
  captures.set(span as object, state);
  const parent = trace.getSpan(parentContext);
  let attrs: Record<string, any> = attributesOf(span) ?? {};
  let events: any[] = dataValue(span, "events") ?? [];
  let status: any = dataValue(span, "status") ?? { code: 0 };
  function allowed(key: string): boolean {
    if (safeKeys.has(key)) return true;
    if (
      key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
      key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
      key.startsWith(`${ATTR_GEN_AI_PROMPT}.`)
    )
      return state.inputs;
    if (
      key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
      key.startsWith(`${ATTR_GEN_AI_COMPLETION}.`)
    )
      return state.outputs;
    return state.inputs && state.outputs;
  }
  function deniedValue(key: string, value: unknown): boolean {
    return (
      !allowed(key) ||
      ((!state.inputs || !state.outputs) && safeKeys.has(key) && !scalar(value))
    );
  }
  function observe(ambient = true): void {
    const descriptor = Object.getOwnPropertyDescriptor(span, "attributes");
    if (
      descriptor &&
      "value" in descriptor &&
      descriptor.value &&
      typeof descriptor.value === "object" &&
      !types.isProxy(descriptor.value)
    )
      attrs = descriptor.value;

    if (
      dataValue(attrs, "allow_trace_content") === false ||
      dataValue(attributesOf(parent), "allow_trace_content") === false ||
      parentContext.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) === false ||
      (ambient &&
        (context.active().getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) === false ||
          isTracingSuppressed(context.active()))) ||
      process.env.RESPAN_TRACE_CONTENT === "false" ||
      process.env.TRACELOOP_TRACE_CONTENT === "false"
    )
      state.inputs = state.outputs = false;
    if (!state.inputs || !state.outputs) {
      for (const key of Object.keys(attrs))
        if (deniedValue(key, dataValue(attrs, key))) delete attrs[key];
      events.length = 0;
      status = {
        code: [0, 1, 2].includes(dataValue(status, "code"))
          ? dataValue(status, "code")
          : 0,
      };
    }
  }
  function clean(value: unknown): Record<string, any> {
    const result: Record<string, any> = {};
    if (!value || typeof value !== "object" || types.isProxy(value))
      return result;
    if (dataValue(value, "allow_trace_content") === false)
      state.inputs = state.outputs = false;
    observe();
    for (const key of Object.keys(value)) {
      if (!allowed(key)) continue;
      const d = Object.getOwnPropertyDescriptor(value, key);
      if (
        !d ||
        !("value" in d) ||
        deniedValue(key, d.value) ||
        !nativeValue(d.value)
      )
        continue;
      result[key] = d.value;
    }
    return result;
  }
  function install(
    key: string,
    get: () => any,
    set: (value: any) => void,
  ): void {
    const d = Object.getOwnPropertyDescriptor(span, key);
    if (d?.configurable === false) {
      if (key === "attributes") {
        const cleanAttrs = clean(attrs);
        for (const key of Object.keys(attrs))
          if (!(key in cleanAttrs)) delete attrs[key];
      }
      return;
    }
    getters.add(get);
    Object.defineProperty(span, key, {
      enumerable: true,
      configurable: false,
      get,
      set,
    });
  }
  install(
    "attributes",
    () => {
      observe(false);
      return attrs;
    },
    (value) => {
      attrs = clean(value);
    },
  );
  install(
    "events",
    () => {
      observe(false);
      return state.inputs && state.outputs ? events : [];
    },
    (value) => {
      observe();
      events =
        state.inputs && state.outputs && Array.isArray(value) ? value : [];
    },
  );
  install(
    "status",
    () => {
      observe(false);
      return status;
    },
    (value) => {
      observe();
      status =
        state.inputs && state.outputs
          ? value
          : {
              code: [0, 1, 2].includes(dataValue(value, "code"))
                ? dataValue(value, "code")
                : 0,
            };
    },
  );
  const setAttribute = span.setAttribute.bind(span),
    setAttributes = span.setAttributes.bind(span),
    setStatus = span.setStatus.bind(span),
    addEvent = span.addEvent.bind(span),
    recordException = span.recordException.bind(span),
    end = span.end.bind(span);
  span.setAttribute = function (key, value) {
    observe();
    if (!deniedValue(key, value)) setAttribute(key, value);
    return this;
  };
  span.setAttributes = function (value) {
    observe();
    setAttributes(clean(value));
    return this;
  };
  span.setStatus = function (value) {
    observe();
    setStatus(
      state.inputs && state.outputs
        ? value
        : {
            code: [0, 1, 2].includes(dataValue(value, "code"))
              ? dataValue(value, "code")
              : 0,
          },
    );
    return this;
  };
  span.addEvent = function (...args: any[]) {
    observe();
    if (state.inputs && state.outputs) (addEvent as any)(...args);
    return this;
  };
  span.recordException = function (...args: any[]) {
    observe();
    if (state.inputs && state.outputs) (recordException as any)(...args);
  };
  span.end = function (...args: any[]) {
    observe();
    (end as any)(...args);
  };
  observe();
  return state;
}
