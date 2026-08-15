import { trace } from "@opentelemetry/api";
import type { Context } from "@opentelemetry/api";
import type { ReadableSpan, Span, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_AGENT_DESCRIPTION,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_CONVERSATION_ID,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_TOOL_TYPE,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const GOOGLE_ADK_INSTRUMENTATION_NAME = "google-adk";
const GOOGLE_ADK_SCOPE_NAME = "gcp.vertex.agent";
const GOOGLE_ADK_LOG_METHOD_TS_TRACING = "ts_tracing";

const ADK_PREFIX = "gcp.vertex.agent.";
const ADK_LLM_REQUEST = "gcp.vertex.agent.llm_request";
const ADK_LLM_RESPONSE = "gcp.vertex.agent.llm_response";
const ADK_TOOL_CALL_ARGS = "gcp.vertex.agent.tool_call_args";
const ADK_TOOL_RESPONSE = "gcp.vertex.agent.tool_response";

const OFF_CONTRACT_ALIASES = new Set([
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "tools",
  "tool_calls",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  RespanSpanAttributes.RESPAN_SPAN_TOOLS,
  RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
]);

type Attributes = Record<string, any>;
type SpanSetAttribute = (key: string, value: unknown) => Span;
type ProcessorOnStart = (span: Span, parentContext: Context) => void;
type ProcessorOnEnd = (span: ReadableSpan) => void;

interface GoogleADKResponseCapture {
  readonly accumulator: GoogleADKResponseAccumulator;
  readonly originalSetAttribute: SpanSetAttribute;
  readonly wrappedSetAttribute: SpanSetAttribute;
}

interface GoogleADKResponseAccumulator {
  // Keep only assembled state: ADK may write a growing cumulative response on
  // every chunk, so retaining raw frames would grow quadratically.
  response?: Attributes;
  content?: GoogleADKContentAccumulator;
}

interface GoogleADKContentAccumulator {
  readonly attributes: Attributes;
  readonly parts: Attributes[];
  readonly partIndexes: Map<string, number>;
}

const GOOGLE_ADK_RESPONSE_CAPTURES = new WeakMap<
  object,
  GoogleADKResponseCapture
>();

export class GoogleADKTranslator implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    const writableSpan = span as any;
    const spanName = String(writableSpan.name ?? "");
    const logType = resolveLogTypeFromName(spanName);
    if (logType === undefined) {
      return;
    }

    if (logType === RespanLogType.CHAT) {
      installResponseCapture(writableSpan);
    }

    writableSpan.setAttribute(
      RespanSpanAttributes.RESPAN_LOG_METHOD,
      GOOGLE_ADK_LOG_METHOD_TS_TRACING,
    );
    writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, logType);

    const entityName = resolveEntityNameFromName(spanName, logType);
    writableSpan.setAttribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
    writableSpan.setAttribute(
      SpanAttributes.TRACELOOP_ENTITY_PATH,
      logType === RespanLogType.WORKFLOW ? "" : entityName,
    );
  }

  onEnd(span: ReadableSpan): void {
    try {
      translateGoogleADKSpan(span);
    } finally {
      restoreResponseCapture(span);
    }
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }
}

export function isGoogleADKSpan(span: ReadableSpan): boolean {
  const attrs = getAttributes(span);
  if (!attrs) {
    return false;
  }

  if (getInstrumentationScopeName(span) === GOOGLE_ADK_SCOPE_NAME) {
    return true;
  }

  if (attrs[ATTR_GEN_AI_SYSTEM] === GOOGLE_ADK_SCOPE_NAME) {
    return true;
  }

  if (typeof attrs[ATTR_GEN_AI_OPERATION_NAME] === "string") {
    return true;
  }

  return Object.keys(attrs).some((key) => key.startsWith(ADK_PREFIX));
}

export function translateGoogleADKSpan(span: ReadableSpan): void {
  const attrs = getAttributes(span);
  if (!attrs || !isGoogleADKSpan(span)) {
    return;
  }

  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = GOOGLE_ADK_LOG_METHOD_TS_TRACING;

  const logType = resolveLogType(span, attrs);
  attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = logType;

  const entityName = resolveEntityName(span, attrs, logType);
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
  setDefault(
    attrs,
    SpanAttributes.TRACELOOP_ENTITY_PATH,
    logType === RespanLogType.WORKFLOW ? "" : entityName,
  );

  if (logType === RespanLogType.CHAT) {
    normalizeChatSpan(attrs, getCapturedResponse(span));
  } else if (logType === RespanLogType.TOOL) {
    normalizeToolSpan(attrs, entityName);
  } else if (logType === RespanLogType.AGENT) {
    normalizeAgentSpan(attrs, entityName);
  } else if (logType === RespanLogType.WORKFLOW) {
    normalizeWorkflowSpan(attrs, entityName);
  }

  cleanupAttrs(attrs);
  restoreResponseCapture(span);
}

export class GoogleADKInstrumentor {
  public readonly name = GOOGLE_ADK_INSTRUMENTATION_NAME;

  private static _translatorRegistered = false;
  private static _translatorHookRefCount = 0;
  private static _patchedProcessor: any = null;
  private static _originalProcessorOnStart: ProcessorOnStart | null = null;
  private static _wrappedProcessorOnStart: ProcessorOnStart | null = null;
  private static _originalProcessorOnEnd: ProcessorOnEnd | null = null;
  private static _wrappedProcessorOnEnd: ProcessorOnEnd | null = null;

  private _ownsTranslatorHook = false;

  activate(): void {
    if (this._ownsTranslatorHook) {
      return;
    }

    GoogleADKInstrumentor._registerTranslatorProcessor();
    GoogleADKInstrumentor._installTranslatorHook();
    GoogleADKInstrumentor._translatorHookRefCount += 1;
    this._ownsTranslatorHook = true;
  }

  deactivate(): void {
    if (!this._ownsTranslatorHook) {
      return;
    }

    GoogleADKInstrumentor._translatorHookRefCount = Math.max(
      0,
      GoogleADKInstrumentor._translatorHookRefCount - 1,
    );
    this._ownsTranslatorHook = false;

    if (GoogleADKInstrumentor._translatorHookRefCount === 0) {
      GoogleADKInstrumentor._restoreTranslatorHook();
    }
  }

  isActive(): boolean {
    return this._ownsTranslatorHook;
  }

  private static _getActiveSpanProcessor(): any {
    const tracerProvider = trace.getTracerProvider() as any;
    return (
      tracerProvider?.activeSpanProcessor ??
      tracerProvider?._activeSpanProcessor ??
      tracerProvider?._delegate?.activeSpanProcessor ??
      tracerProvider?._delegate?._activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?.activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?._activeSpanProcessor
    );
  }

  private static _getTracerProviderWithSpanProcessor(): any {
    const tracerProvider = trace.getTracerProvider() as any;
    return (
      (typeof tracerProvider?.addSpanProcessor === "function" && tracerProvider) ||
      (typeof tracerProvider?._delegate?.addSpanProcessor === "function" && tracerProvider._delegate) ||
      (typeof tracerProvider?._delegate?._tracerProvider?.addSpanProcessor === "function" &&
        tracerProvider._delegate._tracerProvider) ||
      null
    );
  }

  private static _registerTranslatorProcessor(): void {
    if (GoogleADKInstrumentor._translatorRegistered) {
      return;
    }

    const provider = GoogleADKInstrumentor._getTracerProviderWithSpanProcessor();
    if (!provider) {
      return;
    }

    provider.addSpanProcessor(new GoogleADKTranslator());
    GoogleADKInstrumentor._translatorRegistered = true;
  }

  private static _installTranslatorHook(): boolean {
    const processor = GoogleADKInstrumentor._getActiveSpanProcessor();
    if (!processor || typeof processor.onEnd !== "function") {
      return false;
    }

    if (GoogleADKInstrumentor._patchedProcessor === processor) {
      return true;
    }

    GoogleADKInstrumentor._restoreTranslatorHook();

    const translator = new GoogleADKTranslator();
    const originalProcessorOnStart =
      typeof processor.onStart === "function"
        ? (processor.onStart as ProcessorOnStart)
        : null;
    const wrappedProcessorOnStart: ProcessorOnStart | null =
      originalProcessorOnStart
        ? (span: Span, parentContext: Context) => {
            try {
              translator.onStart(span, parentContext);
            } catch {
              // Translation must never block span creation.
            }
            return originalProcessorOnStart.call(
              processor,
              span,
              parentContext,
            );
          }
        : null;
    const originalProcessorOnEnd = processor.onEnd as ProcessorOnEnd;
    const wrappedProcessorOnEnd = (span: ReadableSpan) => {
      try {
        translator.onEnd(span);
      } catch {
        // Translation must never block span export.
      }
      return originalProcessorOnEnd.call(processor, span);
    };

    if (wrappedProcessorOnStart) {
      processor.onStart = wrappedProcessorOnStart;
    }
    processor.onEnd = wrappedProcessorOnEnd;
    GoogleADKInstrumentor._patchedProcessor = processor;
    GoogleADKInstrumentor._originalProcessorOnStart = originalProcessorOnStart;
    GoogleADKInstrumentor._wrappedProcessorOnStart = wrappedProcessorOnStart;
    GoogleADKInstrumentor._originalProcessorOnEnd = originalProcessorOnEnd;
    GoogleADKInstrumentor._wrappedProcessorOnEnd = wrappedProcessorOnEnd;
    return true;
  }

  private static _restoreTranslatorHook(): void {
    const processor = GoogleADKInstrumentor._patchedProcessor;
    const originalOnStart = GoogleADKInstrumentor._originalProcessorOnStart;
    const wrappedOnStart = GoogleADKInstrumentor._wrappedProcessorOnStart;
    const originalOnEnd = GoogleADKInstrumentor._originalProcessorOnEnd;
    const wrappedOnEnd = GoogleADKInstrumentor._wrappedProcessorOnEnd;

    if (processor && originalOnStart) {
      if (!wrappedOnStart || processor.onStart === wrappedOnStart) {
        processor.onStart = originalOnStart;
      } else {
        console.warn(
          "[respan] GoogleADKInstrumentor: active span processor onStart was modified externally; original handler could not be restored.",
        );
      }
    }

    if (processor && originalOnEnd) {
      if (!wrappedOnEnd || processor.onEnd === wrappedOnEnd) {
        processor.onEnd = originalOnEnd;
      } else {
        console.warn(
          "[respan] GoogleADKInstrumentor: active span processor onEnd was modified externally; original handler could not be restored.",
        );
      }
    }

    GoogleADKInstrumentor._patchedProcessor = null;
    GoogleADKInstrumentor._originalProcessorOnStart = null;
    GoogleADKInstrumentor._wrappedProcessorOnStart = null;
    GoogleADKInstrumentor._originalProcessorOnEnd = null;
    GoogleADKInstrumentor._wrappedProcessorOnEnd = null;
  }
}

function resolveLogTypeFromName(spanName: string): RespanLogType | undefined {
  const normalizedName = spanName.toLowerCase();
  if (normalizedName === "call_llm") {
    return RespanLogType.CHAT;
  }
  if (normalizedName.startsWith("execute_tool")) {
    return RespanLogType.TOOL;
  }
  if (normalizedName.startsWith("invoke_agent")) {
    return RespanLogType.AGENT;
  }
  if (normalizedName === "invocation") {
    return RespanLogType.WORKFLOW;
  }
  return undefined;
}

function resolveEntityNameFromName(spanName: string, logType: RespanLogType): string {
  if (logType === RespanLogType.CHAT) {
    return "google_adk.call_llm";
  }
  if (logType === RespanLogType.WORKFLOW) {
    return "google_adk.invocation";
  }

  const [, ...rest] = spanName.split(/\s+/);
  return rest.join(" ") || spanName || "google_adk.task";
}

function getAttributes(span: ReadableSpan): Attributes | undefined {
  return (span as any).attributes as Attributes | undefined;
}

function getInstrumentationScopeName(span: ReadableSpan): string {
  return (
    ((span as any).instrumentationScope?.name as string | undefined) ??
    ((span as any).instrumentationScope?.name as string | undefined) ??
    ""
  );
}

function resolveLogType(span: ReadableSpan, attrs: Attributes): RespanLogType {
  const operation = String(attrs[ATTR_GEN_AI_OPERATION_NAME] ?? "").toLowerCase();
  const spanName = span.name.toLowerCase();

  if (attrs[ATTR_GEN_AI_SYSTEM] === GOOGLE_ADK_SCOPE_NAME || spanName === "call_llm") {
    return RespanLogType.CHAT;
  }
  if (operation === "execute_tool" || spanName.startsWith("execute_tool")) {
    return RespanLogType.TOOL;
  }
  if (operation === "invoke_agent" || spanName.startsWith("invoke_agent")) {
    return RespanLogType.AGENT;
  }
  if (spanName === "invocation") {
    return RespanLogType.WORKFLOW;
  }
  return RespanLogType.TASK;
}

function resolveEntityName(
  span: ReadableSpan,
  attrs: Attributes,
  logType: RespanLogType,
): string {
  if (logType === RespanLogType.AGENT && attrs[ATTR_GEN_AI_AGENT_NAME]) {
    return String(attrs[ATTR_GEN_AI_AGENT_NAME]);
  }
  if (logType === RespanLogType.TOOL && attrs[ATTR_GEN_AI_TOOL_NAME]) {
    return String(attrs[ATTR_GEN_AI_TOOL_NAME]);
  }
  if (logType === RespanLogType.CHAT) {
    return "google_adk.call_llm";
  }
  if (logType === RespanLogType.WORKFLOW) {
    return "google_adk.invocation";
  }
  return span.name || "google_adk.task";
}

function installResponseCapture(span: Span): void {
  const writableSpan = span as any;
  if (
    GOOGLE_ADK_RESPONSE_CAPTURES.has(writableSpan) ||
    typeof writableSpan.setAttribute !== "function"
  ) {
    return;
  }

  const accumulator: GoogleADKResponseAccumulator = {};
  const originalSetAttribute = writableSpan.setAttribute as SpanSetAttribute;
  const wrappedSetAttribute: SpanSetAttribute = function (
    this: Span,
    key: string,
    value: unknown,
  ): Span {
    if (key === ADK_LLM_RESPONSE) {
      accumulateResponse(accumulator, value);
    }
    return originalSetAttribute.call(this, key, value);
  };

  try {
    writableSpan.setAttribute = wrappedSetAttribute;
    GOOGLE_ADK_RESPONSE_CAPTURES.set(writableSpan, {
      accumulator,
      originalSetAttribute,
      wrappedSetAttribute,
    });
  } catch {
    // Some third-party Span implementations may not allow method wrapping.
    // Their final scalar response still follows the existing unary path.
  }
}

function getCapturedResponse(span: ReadableSpan): Attributes | undefined {
  const accumulator = GOOGLE_ADK_RESPONSE_CAPTURES.get(
    span as object,
  )?.accumulator;
  return accumulator ? materializeResponse(accumulator) : undefined;
}

function restoreResponseCapture(span: ReadableSpan): void {
  const writableSpan = span as any;
  const capture = GOOGLE_ADK_RESPONSE_CAPTURES.get(writableSpan);
  if (!capture) {
    return;
  }

  try {
    if (writableSpan.setAttribute === capture.wrappedSetAttribute) {
      writableSpan.setAttribute = capture.originalSetAttribute;
    }
  } catch {
    // The response data has already been normalized; an ended span with a
    // non-writable method does not need further mutation.
  } finally {
    GOOGLE_ADK_RESPONSE_CAPTURES.delete(writableSpan);
  }
}

function normalizeChatSpan(
  attrs: Attributes,
  capturedResponse?: Attributes,
): void {
  attrs[ATTR_GEN_AI_SYSTEM] = "google";
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;

  const request = parseJson(attrs[ADK_LLM_REQUEST]);
  if (isRecord(request)) {
    if (request.model !== undefined) {
      setDefault(attrs, ATTR_GEN_AI_REQUEST_MODEL, request.model);
    }

    const config = isRecord(request.config) ? request.config : undefined;
    const systemInstruction = firstDefined(
      config?.systemInstruction,
      config?.system_instruction,
    );
    if (typeof systemInstruction === "string" && systemInstruction) {
      setMessage(attrs, ATTR_GEN_AI_PROMPT, 0, {
        role: "system",
        content: systemInstruction,
      });
    }

    const tools = extractTools(config);
    if (tools.length > 0) {
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
    }

    const contents = Array.isArray(request.contents) ? request.contents : [];
    const startIndex = attrs[`${ATTR_GEN_AI_PROMPT}.0.content`] ? 1 : 0;
    for (const [offset, content] of contents.entries()) {
      addContentMessage(attrs, ATTR_GEN_AI_PROMPT, startIndex + offset, content);
    }
  }

  const response = capturedResponse ?? parseJson(attrs[ADK_LLM_RESPONSE]);
  if (isRecord(response)) {
    addContentMessage(attrs, ATTR_GEN_AI_COMPLETION, 0, response.content);

    const usage = isRecord(response.usageMetadata)
      ? response.usageMetadata
      : isRecord(response.usage_metadata)
        ? response.usage_metadata
        : undefined;
    const inputTokens = numberValue(firstDefined(
      usage?.promptTokenCount,
      usage?.prompt_token_count,
    ));
    const outputTokens = numberValue(firstDefined(
      usage?.candidatesTokenCount,
      usage?.candidates_token_count,
    ));
    const thoughtsTokens = numberValue(firstDefined(
      usage?.thoughtsTokenCount,
      usage?.thoughts_token_count,
    ));
    const totalTokens = numberValue(firstDefined(
      usage?.totalTokenCount,
      usage?.total_token_count,
    ));
    const normalizedOutputTokens =
      outputTokens === undefined
        ? undefined
        : outputTokens + (thoughtsTokens ?? 0);

    if (inputTokens !== undefined) {
      attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
      attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = inputTokens;
    }
    if (normalizedOutputTokens !== undefined) {
      attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = normalizedOutputTokens;
      attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = normalizedOutputTokens;
    }
    if (totalTokens !== undefined) {
      attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
    } else if (inputTokens !== undefined && normalizedOutputTokens !== undefined) {
      attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] =
        inputTokens + normalizedOutputTokens;
    }

    const finishReason = firstDefined(response.finishReason, response.finish_reason);
    if (finishReason !== undefined) {
      attrs[`${ATTR_GEN_AI_COMPLETION}.0.finish_reason`] =
        String(finishReason).toLowerCase();
    }
  }
}

function accumulateResponse(
  accumulator: GoogleADKResponseAccumulator,
  value: unknown,
): void {
  const response = parseJson(value);
  if (!isRecord(response)) {
    return;
  }

  accumulator.response ??= {};
  for (const [key, responseValue] of Object.entries(response)) {
    if (key !== "content") {
      accumulator.response[key] = responseValue;
    }
  }

  if (isRecord(response.content)) {
    accumulator.content ??= {
      attributes: {},
      parts: [],
      partIndexes: new Map<string, number>(),
    };
    accumulateResponseContent(
      accumulator.content,
      response.content,
      response.partial === true,
    );
  }
}

function materializeResponse(
  accumulator: GoogleADKResponseAccumulator,
): Attributes | undefined {
  if (!accumulator.response && !accumulator.content) {
    return undefined;
  }

  const response = { ...(accumulator.response ?? {}) };
  if (accumulator.content) {
    response.content = {
      ...accumulator.content.attributes,
      ...(accumulator.content.parts.length > 0
        ? { parts: accumulator.content.parts }
        : {}),
    };
  }
  return response;
}

function accumulateResponseContent(
  accumulator: GoogleADKContentAccumulator,
  content: Attributes,
  isPartialResponse: boolean,
): void {
  for (const [key, contentValue] of Object.entries(content)) {
    if (key !== "parts") {
      accumulator.attributes[key] = contentValue;
    }
  }
  if (!Array.isArray(content.parts)) {
    return;
  }

  const functionCallSequences = new Map<string, number>();
  for (const part of content.parts) {
    if (!isRecord(part)) {
      continue;
    }

    if (isRecord(part.functionCall)) {
      const name = String(part.functionCall.name ?? "");
      const sequence = functionCallSequences.get(name) ?? 0;
      functionCallSequences.set(name, sequence + 1);
      accumulateFunctionCallPart(accumulator, part, name, sequence);
      continue;
    }

    if (typeof part.text === "string") {
      accumulateTextPart(accumulator, part, isPartialResponse);
      continue;
    }

    const identity = `part:${safeJson(part)}`;
    if (!accumulator.partIndexes.has(identity)) {
      accumulator.partIndexes.set(identity, accumulator.parts.length);
      accumulator.parts.push({ ...part });
    }
  }
}

function accumulateTextPart(
  accumulator: GoogleADKContentAccumulator,
  part: Attributes,
  isPartialResponse: boolean,
): void {
  const textShape = { ...part };
  delete textShape.text;
  const identity = `text:${safeJson(textShape)}`;
  const existingIndex = accumulator.partIndexes.get(identity);
  if (existingIndex === undefined) {
    accumulator.partIndexes.set(identity, accumulator.parts.length);
    accumulator.parts.push({ ...part });
    return;
  }

  const existingPart = accumulator.parts[existingIndex];
  existingPart.text = mergeStreamText(
    String(existingPart.text ?? ""),
    part.text,
    !isPartialResponse,
  );
}

function accumulateFunctionCallPart(
  accumulator: GoogleADKContentAccumulator,
  part: Attributes,
  name: string,
  sequence: number,
): void {
  const functionCall = part.functionCall as Attributes;
  const sequenceIdentity = `functionCall:name:${name}:sequence:${sequence}`;
  const idIdentity = functionCall.id === undefined
    ? undefined
    : `functionCall:id:${String(functionCall.id)}`;
  const existingIndex = (
    idIdentity === undefined
      ? undefined
      : accumulator.partIndexes.get(idIdentity)
  ) ?? accumulator.partIndexes.get(sequenceIdentity);

  if (existingIndex === undefined) {
    const nextIndex = accumulator.parts.length;
    accumulator.parts.push(mergeFunctionCallPart({}, part));
    accumulator.partIndexes.set(sequenceIdentity, nextIndex);
    if (idIdentity !== undefined) {
      accumulator.partIndexes.set(idIdentity, nextIndex);
    }
    return;
  }

  accumulator.parts[existingIndex] = mergeFunctionCallPart(
    accumulator.parts[existingIndex],
    part,
  );
  accumulator.partIndexes.set(sequenceIdentity, existingIndex);
  if (idIdentity !== undefined) {
    accumulator.partIndexes.set(idIdentity, existingIndex);
  }
}

function mergeStreamText(
  current: string,
  next: string,
  deduplicateEqualValue = false,
): string {
  if (!current) {
    return next;
  }
  if (!next) {
    return current;
  }

  // Some ADK providers emit deltas while others emit a cumulative terminal
  // response. Replacing a prefix-complete value keeps each chunk exactly once.
  if (
    next.startsWith(current) &&
    (next.length > current.length || deduplicateEqualValue)
  ) {
    return next;
  }
  return current + next;
}

function mergeFunctionCallPart(
  current: Attributes,
  next: Attributes,
): Attributes {
  const currentCall = isRecord(current.functionCall)
    ? current.functionCall
    : {};
  const nextCall = isRecord(next.functionCall) ? next.functionCall : {};
  const hasCurrentArgs = isRecord(currentCall.args);
  const args = hasCurrentArgs ? currentCall.args : {};

  if (Array.isArray(nextCall.partialArgs)) {
    for (const partialArg of nextCall.partialArgs) {
      applyPartialArg(args, partialArg);
    }
  }
  if (isRecord(nextCall.args)) {
    mergeRecordInPlace(args, nextCall.args);
  }

  const functionCall = { ...currentCall, ...nextCall };
  delete functionCall.partialArgs;
  delete functionCall.willContinue;
  if (
    hasCurrentArgs ||
    Array.isArray(nextCall.partialArgs) ||
    isRecord(nextCall.args)
  ) {
    functionCall.args = args;
  }

  return {
    ...current,
    ...next,
    functionCall,
  };
}

function applyPartialArg(args: Attributes, value: unknown): void {
  if (!isRecord(value) || typeof value.jsonPath !== "string") {
    return;
  }

  const path = parseJsonPath(value.jsonPath);
  const partialValue = partialArgValue(value);
  if (!path || path.length === 0 || !partialValue.present) {
    return;
  }

  let target: any = args;
  for (let index = 0; index < path.length - 1; index += 1) {
    const key = path[index];
    const nextKey = path[index + 1];
    const expectedContainer = typeof nextKey === "number" ? [] : {};
    const existing = target[key];
    if (
      (Array.isArray(expectedContainer) && !Array.isArray(existing)) ||
      (!Array.isArray(expectedContainer) && !isRecord(existing))
    ) {
      target[key] = expectedContainer;
    }
    target = target[key];
  }

  const key = path[path.length - 1];
  const existing = target[key];
  target[key] =
    typeof existing === "string" && typeof partialValue.value === "string"
      ? existing + partialValue.value
      : partialValue.value;
}

function partialArgValue(value: Attributes): {
  present: boolean;
  value?: unknown;
} {
  for (const key of [
    "stringValue",
    "numberValue",
    "boolValue",
    "nullValue",
  ]) {
    if (Object.prototype.hasOwnProperty.call(value, key)) {
      return {
        present: true,
        value: key === "nullValue" ? null : value[key],
      };
    }
  }
  return { present: false };
}

function parseJsonPath(path: string): Array<string | number> | undefined {
  if (!path.startsWith("$")) {
    return undefined;
  }

  const segments: Array<string | number> = [];
  let index = 1;
  while (index < path.length) {
    if (path[index] === ".") {
      const start = index + 1;
      index = start;
      while (index < path.length && path[index] !== "." && path[index] !== "[") {
        index += 1;
      }
      if (index === start) {
        return undefined;
      }
      segments.push(path.slice(start, index));
      continue;
    }

    if (path[index] === "[") {
      const end = path.indexOf("]", index + 1);
      if (end === -1) {
        return undefined;
      }
      const selector = path.slice(index + 1, end).trim();
      if (/^\d+$/.test(selector)) {
        segments.push(Number(selector));
      } else if (
        (selector.startsWith('"') && selector.endsWith('"')) ||
        (selector.startsWith("'") && selector.endsWith("'"))
      ) {
        const quote = selector[0];
        const property = selector
          .slice(1, -1)
          .replace(new RegExp(`\\\\${quote}`, "g"), quote)
          .replace(/\\\\\\\\/g, "\\");
        segments.push(property);
      } else {
        return undefined;
      }
      index = end + 1;
      continue;
    }

    return undefined;
  }
  return segments;
}

function mergeRecordInPlace(target: Attributes, source: Attributes): void {
  for (const [key, value] of Object.entries(source)) {
    if (isRecord(value) && isRecord(target[key])) {
      mergeRecordInPlace(target[key], value);
    } else if (isRecord(value)) {
      target[key] = { ...value };
    } else if (Array.isArray(value)) {
      target[key] = [...value];
    } else {
      target[key] = value;
    }
  }
}

function normalizeToolSpan(attrs: Attributes, entityName: string): void {
  const rawArgs = parseJson(attrs[ADK_TOOL_CALL_ARGS]);
  const args = rawArgs === "N/A" ? undefined : rawArgs;
  const input = {
    name: entityName,
    arguments: args ?? {},
  };
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(input);

  const rawResponse = parseJson(attrs[ADK_TOOL_RESPONSE]);
  if (rawResponse !== undefined && rawResponse !== "<not specified>") {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(rawResponse);
  }

  if (attrs[ATTR_GEN_AI_TOOL_CALL_ID]) {
    attrs[metadataKey("google_adk_tool_call_id")] =
      String(attrs[ATTR_GEN_AI_TOOL_CALL_ID]);
  }
  if (attrs[ATTR_GEN_AI_TOOL_TYPE]) {
    attrs[metadataKey("google_adk_tool_type")] = String(attrs[ATTR_GEN_AI_TOOL_TYPE]);
  }
  if (attrs[ATTR_GEN_AI_TOOL_DESCRIPTION]) {
    attrs[metadataKey("google_adk_tool_description")] =
      String(attrs[ATTR_GEN_AI_TOOL_DESCRIPTION]);
  }
}

function normalizeAgentSpan(attrs: Attributes, entityName: string): void {
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = entityName;

  if (attrs[ATTR_GEN_AI_AGENT_DESCRIPTION]) {
    attrs[metadataKey("google_adk_agent_description")] =
      String(attrs[ATTR_GEN_AI_AGENT_DESCRIPTION]);
  }
  if (attrs[ATTR_GEN_AI_CONVERSATION_ID]) {
    attrs[metadataKey("google_adk_conversation_id")] =
      String(attrs[ATTR_GEN_AI_CONVERSATION_ID]);
  }
}

function normalizeWorkflowSpan(attrs: Attributes, entityName: string): void {
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, "");
}

function addContentMessage(
  attrs: Attributes,
  prefix: string,
  index: number,
  value: unknown,
): void {
  if (!isRecord(value)) {
    return;
  }

  const role = normalizeRole(value.role);
  const { text, toolCalls, toolResponse } = extractContentParts(value.parts);
  setMessage(attrs, prefix, index, {
    role: toolResponse !== undefined ? "tool" : role,
    content: toolResponse !== undefined ? safeJson(toolResponse) : text,
    toolCalls,
  });
}

function setMessage(
  attrs: Attributes,
  prefix: string,
  index: number,
  message: {
    role?: string;
    content?: string;
    toolCalls?: Array<Record<string, unknown>>;
  },
): void {
  const target = `${prefix}.${index}`;
  if (message.role) {
    attrs[`${target}.role`] = message.role;
  }
  if (message.content !== undefined) {
    attrs[`${target}.content`] = message.content;
  }
  if (message.toolCalls && message.toolCalls.length > 0) {
    attrs[`${target}.tool_calls`] = safeJson(message.toolCalls);
  }
}

function extractContentParts(parts: unknown): {
  text?: string;
  toolCalls?: Array<Record<string, unknown>>;
  toolResponse?: unknown;
} {
  if (!Array.isArray(parts)) {
    return {};
  }

  const textParts: string[] = [];
  const toolCalls: Array<Record<string, unknown>> = [];
  let toolResponse: unknown;

  for (const part of parts) {
    if (!isRecord(part)) {
      continue;
    }

    if (typeof part.text === "string") {
      textParts.push(part.text);
    }

    if (isRecord(part.functionCall)) {
      const functionCall = part.functionCall;
      const functionPayload: Record<string, unknown> = {};
      if (functionCall.name !== undefined) {
        functionPayload.name = functionCall.name;
      }
      if (functionCall.args !== undefined) {
        functionPayload.arguments = safeJson(functionCall.args);
      }

      const toolCall: Record<string, unknown> = {
        type: "function",
        function: functionPayload,
      };
      if (functionCall.id !== undefined) {
        toolCall.id = functionCall.id;
      }
      toolCalls.push(toolCall);
    }

    if (isRecord(part.functionResponse)) {
      toolResponse = part.functionResponse.response ?? part.functionResponse;
    }
  }

  return {
    text: textParts.length > 0 ? textParts.join("\n") : undefined,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    toolResponse,
  };
}

function extractTools(config: Attributes | undefined): Array<Record<string, unknown>> {
  if (!config || !Array.isArray(config.tools)) {
    return [];
  }

  const tools: Array<Record<string, unknown>> = [];
  for (const tool of config.tools) {
    if (!isRecord(tool)) {
      continue;
    }
    const declarations = firstDefined(
      tool.functionDeclarations,
      tool.function_declarations,
    );
    if (!Array.isArray(declarations)) {
      continue;
    }
    for (const declaration of declarations) {
      if (isRecord(declaration)) {
        tools.push(declaration);
      }
    }
  }
  return tools;
}

function cleanupAttrs(attrs: Attributes): void {
  for (const key of Object.keys(attrs)) {
    if (
      key.startsWith(ADK_PREFIX) ||
      key === ATTR_GEN_AI_OPERATION_NAME ||
      key === ATTR_GEN_AI_AGENT_DESCRIPTION ||
      key === ATTR_GEN_AI_AGENT_NAME ||
      key === ATTR_GEN_AI_CONVERSATION_ID ||
      key === ATTR_GEN_AI_TOOL_CALL_ID ||
      key === ATTR_GEN_AI_TOOL_DESCRIPTION ||
      key === ATTR_GEN_AI_TOOL_NAME ||
      key === ATTR_GEN_AI_TOOL_TYPE ||
      OFF_CONTRACT_ALIASES.has(key)
    ) {
      delete attrs[key];
    }
  }
}

function normalizeRole(role: unknown): string {
  if (role === "model") {
    return "assistant";
  }
  if (typeof role === "string" && role) {
    return role;
  }
  return "user";
}

function parseJson(value: unknown): unknown {
  if (typeof value !== "string") {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function safeJson(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function isRecord(value: unknown): value is Attributes {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function setDefault(attrs: Attributes, key: string, value: unknown): void {
  if (attrs[key] === undefined && value !== undefined) {
    attrs[key] = value;
  }
}

function firstDefined<T>(...values: Array<T | undefined>): T | undefined {
  for (const value of values) {
    if (value !== undefined) {
      return value;
    }
  }
  return undefined;
}

function metadataKey(name: string): string {
  return `respan.metadata.${name}`;
}
