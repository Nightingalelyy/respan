import {
  context,
  trace,
  TraceFlags,
  createContextKey,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import {
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { types } from "node:util";
import {
  LK_CHAT_CTX,
  LK_FUNCTION_TOOLS,
  LK_PROVIDER_TOOLS,
  LK_TOOL_SETS,
  LK_USER_INPUT,
  LK_INSTRUCTIONS,
  LK_FUNCTION_TOOL_ARGS,
  LK_TTS_INPUT_TEXT,
  LK_RESPONSE_TEXT,
  LK_RESPONSE_FUNCTION_CALLS,
  LK_FUNCTION_TOOL_OUTPUT,
  LK_FUNCTION_TOOL_IS_ERROR,
  LK_USER_TRANSCRIPT,
  LIVEKIT_SPAN_NAME_ATTRIBUTE,
  LK_FUNCTION_TOOL_ID,
  LK_FUNCTION_TOOL_NAME,
  LK_AGENT_LABEL,
  LK_AGENT_NAME,
  LK_RETRY_COUNT,
} from "./_constants.js";

export interface CaptureOptions {
  traceContent?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
}

export interface CapturePolicy {
  inputs: boolean;
  outputs: boolean;
  sampled: boolean;
  parent?: Span;
}

const policies = new WeakMap<object, CapturePolicy>();
const guarded = new WeakSet<object>();
const suppressLanguageModel = createContextKey(
  "suppress_language_model_instrumentation",
);

export function data(value: unknown, key: string): any {
  if (
    !value ||
    (typeof value !== "object" && typeof value !== "function") ||
    types.isProxy(value)
  )
    return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}

/** Copy admitted data properties without executing customer getters or toJSON. */
export function snapshot(value: unknown, seen = new WeakSet<object>()): any {
  if (value === null || ["string", "number", "boolean"].includes(typeof value))
    return value;
  if (
    !value ||
    typeof value !== "object" ||
    types.isProxy(value) ||
    seen.has(value)
  )
    return undefined;
  seen.add(value);
  const out: any = Array.isArray(value) ? [] : {};
  for (const key of Object.keys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !("value" in descriptor)) continue;
    const copied = snapshot(descriptor.value, seen);
    if (copied !== undefined)
      Object.defineProperty(out, key, {
        value: copied,
        enumerable: true,
        configurable: true,
        writable: true,
      });
  }
  seen.delete(value);
  return out;
}

function denied(value: unknown): boolean {
  return (
    value === false ||
    (typeof value === "string" &&
      ["false", "0", "no", "off"].includes(value.trim().toLowerCase()))
  );
}

function inspectParent(
  policy: CapturePolicy,
  parent: Span | undefined,
  visited = new Set<object>(),
): void {
  if (!parent || visited.has(parent)) return;
  visited.add(parent);
  if (types.isProxy(parent)) {
    policy.inputs = policy.outputs = false;
    return;
  }
  const inherited = policies.get(parent);
  const attrs = guarded.has(parent)
    ? (parent as any).attributes
    : data(parent, "attributes");
  if (denied(data(attrs, "allow_trace_content"))) {
    policy.inputs = policy.outputs = false;
    if (inherited) inherited.inputs = inherited.outputs = false;
    else policies.set(parent, { inputs: false, outputs: false, sampled: true });
  }
  if (inherited) {
    policy.inputs &&= inherited.inputs;
    policy.outputs &&= inherited.outputs;
    inspectParent(policy, inherited.parent, visited);
  }
}

export function refresh(
  policy: CapturePolicy,
  observeContext = false,
): CapturePolicy {
  if (
    denied(process.env.RESPAN_TRACE_CONTENT) ||
    denied(process.env.TRACELOOP_TRACE_CONTENT) ||
    (observeContext &&
      (denied(context.active().getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)) ||
        isTracingSuppressed(context.active()) ||
        context.active().getValue(suppressLanguageModel) === true))
  ) {
    policy.inputs = policy.outputs = false;
  }
  inspectParent(policy, policy.parent);
  if (!policy.sampled) policy.inputs = policy.outputs = false;
  return policy;
}

export function capture(
  span: Span,
  ctx: Context,
  owners: Iterable<CaptureOptions>,
): CapturePolicy {
  const policy: CapturePolicy = {
    inputs: true,
    outputs: true,
    sampled: Boolean(span.spanContext().traceFlags & TraceFlags.SAMPLED),
    parent: trace.getSpan(ctx),
  };
  for (const options of owners) {
    policy.inputs &&=
      options.traceContent !== false && options.recordInputs !== false;
    policy.outputs &&=
      options.traceContent !== false && options.recordOutputs !== false;
  }
  if (
    denied(ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)) ||
    isTracingSuppressed(ctx) ||
    ctx.getValue(suppressLanguageModel) === true
  )
    policy.inputs = policy.outputs = false;
  return refresh(policy);
}

const safeKeys = new Set<string>([
  "allow_trace_content",
  LIVEKIT_SPAN_NAME_ATTRIBUTE,
  LK_FUNCTION_TOOL_ID,
  LK_FUNCTION_TOOL_NAME,
  LK_AGENT_LABEL,
  LK_AGENT_NAME,
  LK_RETRY_COUNT,
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN,
  RespanSpanAttributes.RESPAN_INTERNAL_EXPORT_PARENT,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL,
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  SpanAttributes.TRACELOOP_WORKFLOW_NAME,
  SpanAttributes.LLM_REQUEST_TYPE,
  SpanAttributes.LLM_SYSTEM,
  SpanAttributes.LLM_REQUEST_MODEL,
  SpanAttributes.LLM_RESPONSE_MODEL,
  SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
  SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
]);

const inputKeys = new Set<string>([
  ...[
    LK_CHAT_CTX,
    LK_FUNCTION_TOOLS,
    LK_PROVIDER_TOOLS,
    LK_TOOL_SETS,
    LK_USER_INPUT,
    LK_INSTRUCTIONS,
    LK_FUNCTION_TOOL_ARGS,
    LK_TTS_INPUT_TEXT,
  ].flatMap((key) => [key, `lk.pii.${key.slice(3)}`]),
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
]);
const outputKeys = new Set<string>([
  ...[
    LK_RESPONSE_TEXT,
    LK_RESPONSE_FUNCTION_CALLS,
    LK_FUNCTION_TOOL_OUTPUT,
    LK_FUNCTION_TOOL_IS_ERROR,
    LK_USER_TRANSCRIPT,
  ].flatMap((key) => [key, `lk.pii.${key.slice(3)}`]),
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
]);
function allowedKey(key: string, policy: CapturePolicy): boolean {
  if (safeKeys.has(key)) return true;
  if (!policy.sampled) return false;
  if (
    policy.inputs &&
    (key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
      key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
      key.startsWith(`${SpanAttributes.LLM_PROMPTS}.`) ||
      inputKeys.has(key))
  )
    return true;
  if (
    policy.outputs &&
    (key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
      key.startsWith(`${SpanAttributes.LLM_COMPLETIONS}.`) ||
      outputKeys.has(key))
  )
    return true;
  return policy.inputs && policy.outputs;
}

function primitive(value: unknown): boolean {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}

export function filterAttributes(
  value: unknown,
  policy: CapturePolicy,
): Record<string, any> {
  refresh(policy);
  if (denied(data(value, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  const out: Record<string, any> = {};
  if (!value || typeof value !== "object" || types.isProxy(value)) return out;
  const privateSpan = !policy.sampled || !policy.inputs || !policy.outputs;
  for (const key of Object.keys(value)) {
    if (!allowedKey(key, policy)) continue;
    const item = data(value, key);
    if (privateSpan && !primitive(item)) continue;
    const copied = snapshot(item);
    if (copied !== undefined)
      Object.defineProperty(out, key, {
        value: copied,
        enumerable: true,
        configurable: true,
        writable: true,
      });
  }
  return out;
}

function strip(attrs: Record<string, unknown>, policy: CapturePolicy): void {
  refresh(policy);
  if (denied(data(attrs, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  const privateSpan = !policy.sampled || !policy.inputs || !policy.outputs;
  for (const key of Object.keys(attrs)) {
    if (!allowedKey(key, policy)) {
      delete attrs[key];
      continue;
    }
    const descriptor = Object.getOwnPropertyDescriptor(attrs, key);
    if (
      !descriptor ||
      !("value" in descriptor) ||
      (privateSpan && !primitive(descriptor.value))
    )
      delete attrs[key];
  }
}

function privateStatus(value: unknown): { code: number } {
  const code = data(value, "code");
  return {
    code: typeof code === "number" && [0, 1, 2].includes(code) ? code : 0,
  };
}

/** Guard the actual native span; exporter clones alone do not prevent late leaks. */
export function guard(span: Span, policy: CapturePolicy): void {
  policies.set(span, policy);
  guarded.add(span);
  const target = span as any;
  const originalAttribute = span.setAttribute,
    originalAttributes = span.setAttributes;
  const originalEvent = span.addEvent,
    originalException = span.recordException,
    originalStatus = span.setStatus;
  const admitted = () => {
    refresh(policy, true);
    return policy.sampled && policy.inputs && policy.outputs;
  };
  span.setAttribute = function (key, value) {
    refresh(policy, true);
    if (key === "allow_trace_content" && denied(value))
      policy.inputs = policy.outputs = false;
    if (
      !allowedKey(key, policy) ||
      ((!policy.sampled || !policy.inputs || !policy.outputs) &&
        !primitive(value))
    )
      return this;
    return originalAttribute.call(this, key, value);
  };
  span.setAttributes = function (value) {
    refresh(policy, true);
    return originalAttributes.call(this, filterAttributes(value, policy));
  };
  span.addEvent = function (...args: any[]) {
    if (!admitted()) return this;
    return originalEvent.apply(this, args as any);
  };
  span.recordException = function (...args: any[]) {
    if (admitted()) originalException.apply(this, args as any);
  };
  span.setStatus = function (value) {
    return originalStatus.call(this, admitted() ? value : privateStatus(value));
  };
  let attrs = data(target, "attributes") ?? {};
  let events = data(target, "events") ?? [];
  let status = data(target, "status") ?? { code: 0 };
  Object.defineProperties(target, {
    attributes: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        return attrs;
      },
      set(value) {
        attrs = filterAttributes(value, policy);
      },
    },
    events: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        if (!policy.sampled || !policy.inputs || !policy.outputs) events = [];
        return events;
      },
      set(value) {
        refresh(policy);
        events =
          policy.sampled && policy.inputs && policy.outputs
            ? (snapshot(value) ?? [])
            : [];
      },
    },
    status: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        if (!policy.sampled || !policy.inputs || !policy.outputs)
          status = privateStatus(status);
        return status;
      },
      set(value) {
        refresh(policy);
        status =
          policy.sampled && policy.inputs && policy.outputs
            ? (snapshot(value) ?? { code: 0 })
            : privateStatus(value);
      },
    },
  });
  strip(attrs, policy);
}

export function policyFor(span: object): CapturePolicy | undefined {
  return policies.get(span);
}
