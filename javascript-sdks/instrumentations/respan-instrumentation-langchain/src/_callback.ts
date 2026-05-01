import { hrTime } from "@opentelemetry/core";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import {
  DIRECT_COMPLETION_TOKENS,
  DIRECT_INPUT,
  DIRECT_MODEL,
  DIRECT_OUTPUT,
  DIRECT_PROMPT_TOKENS,
  DIRECT_TOTAL_REQUEST_TOKENS,
  ERROR_MESSAGE_ATTR,
  GEN_AI_COMPLETION_PREFIX,
  GEN_AI_PROMPT_PREFIX,
  GEN_AI_TOOL_CALL_ARGUMENTS,
  GEN_AI_TOOL_CALL_RESULT,
  GEN_AI_TOOL_NAME,
  GEN_AI_USAGE_INPUT_TOKENS,
  GEN_AI_USAGE_OUTPUT_TOKENS,
  GEN_AI_USAGE_TOTAL_TOKENS,
  LANGCHAIN_FRAMEWORK_ATTR,
  LANGCHAIN_METADATA_ATTR,
  LANGCHAIN_PARENT_RUN_ID_ATTR,
  LANGCHAIN_RUN_ID_ATTR,
  LANGCHAIN_SERIALIZED_ATTR,
  LANGCHAIN_TAGS_ATTR,
  LLM_USAGE_TOTAL_TOKENS,
  RESPAN_LOG_METHOD_TS_TRACING,
  STATUS_CODE_ATTR,
  TL_ENTITY_INPUT,
  TL_ENTITY_NAME,
  TL_ENTITY_OUTPUT,
  TL_ENTITY_PATH,
  TL_SPAN_KIND,
  deriveSpanId,
  detectFramework,
  extractLlmOutput,
  extractModel,
  extractName,
  extractToolCallsFromMessages,
  extractToolNamesFromSerialized,
  extractUsage,
  generateTraceId,
  getActiveOtelParent,
  getErrorMessage,
  isPlainRecord,
  normalizeChatMessages,
  normalizeMetadata,
  normalizeOutputForLogging,
  normalizeTags,
  runIdToHex,
  safeJsonString,
  setIfPresent,
  toSerializableValue,
  trimMap,
  type RespanCallbackHandlerOptions,
  type RunRecord,
  type SpanAttributesRecord,
} from "./_callback_helpers.js";

export type { RespanCallbackHandlerOptions } from "./_callback_helpers.js";

export function getCallbackHandler(
  options: RespanCallbackHandlerOptions = {},
): RespanCallbackHandler {
  return new RespanCallbackHandler({
    groupLangflowRootRuns: true,
    ...options,
  });
}

export function addRespanCallback(
  config: Record<string, any> = {},
  handler: RespanCallbackHandler = getCallbackHandler(),
): Record<string, any> {
  const nextConfig = { ...config };
  nextConfig.callbacks = withRespanCallback(nextConfig.callbacks, handler);
  return nextConfig;
}

function isRespanCallbackHandler(value: unknown): boolean {
  return value instanceof RespanCallbackHandler ||
    Boolean(isPlainRecord(value) && value._respanCallbackHandler === true);
}

function callbackListContains(callbacks: unknown[], handler: RespanCallbackHandler): boolean {
  return callbacks.some((callback) => callback === handler || isRespanCallbackHandler(callback));
}

function withRespanCallback(callbacks: unknown, handler: RespanCallbackHandler): unknown {
  if (callbacks === undefined || callbacks === null) {
    return [handler];
  }

  if (Array.isArray(callbacks)) {
    return callbackListContains(callbacks, handler) ? callbacks : [...callbacks, handler];
  }

  if (isPlainRecord(callbacks) && Array.isArray(callbacks.handlers)) {
    if (!callbackListContains(callbacks.handlers, handler)) {
      if (typeof callbacks.addHandler === "function") {
        callbacks.addHandler(handler, true);
      } else {
        callbacks.handlers.push(handler);
      }
    }
    return callbacks;
  }

  if (isRespanCallbackHandler(callbacks)) {
    return [callbacks];
  }

  return [callbacks, handler];
}

export class RespanCallbackHandler {
  public readonly name = "RespanCallbackHandler";
  public readonly _respanCallbackHandler = true;
  public readonly ignoreLLM = false;
  public readonly ignoreChain = false;
  public readonly ignoreAgent = false;
  public readonly ignoreRetriever = false;
  public readonly ignoreCustomEvent = false;
  public readonly raiseError = false;
  public readonly awaitHandlers = false;

  public includeContent: boolean;
  public includeMetadata: boolean;
  public groupLangflowRootRuns: boolean;
  public maxCachedRuns: number;

  private readonly _runs = new Map<string, RunRecord>();
  private readonly _runTraceIds = new Map<string, string>();
  private readonly _runPaths = new Map<string, string>();
  private readonly _langflowTraceId = generateTraceId();

  constructor(options: RespanCallbackHandlerOptions = {}) {
    this.includeContent = options.includeContent ?? true;
    this.includeMetadata = options.includeMetadata ?? true;
    this.groupLangflowRootRuns = options.groupLangflowRootRuns ?? false;
    this.maxCachedRuns = options.maxCachedRuns ?? 4096;
  }

  private _rememberRun(record: RunRecord): void {
    this._runTraceIds.set(record.runId, record.traceId);
    this._runPaths.set(record.runId, record.entityPath);
    trimMap(this._runTraceIds, this.maxCachedRuns);
    trimMap(this._runPaths, this.maxCachedRuns);
  }

  private _startRun({
    runId,
    parentRunId,
    name,
    logType,
    spanKind,
    inputValue,
    serialized,
    tags,
    metadata,
    extraAttributes = {},
  }: {
    runId: unknown;
    parentRunId?: unknown;
    name: string;
    logType: string;
    spanKind: string;
    inputValue?: unknown;
    serialized?: unknown;
    tags?: string[];
    metadata?: Record<string, unknown>;
    extraAttributes?: SpanAttributesRecord;
  }): void {
    const runHex = runIdToHex(runId);
    const parentHex = parentRunId !== undefined && parentRunId !== null
      ? runIdToHex(parentRunId)
      : undefined;
    const framework = detectFramework({ serialized, tags, metadata, name });
    const activeParent = parentHex ? undefined : getActiveOtelParent();
    const fallbackTraceId =
      framework === "langflow" &&
      this.groupLangflowRootRuns &&
      !activeParent &&
      !parentHex
        ? this._langflowTraceId
        : parentHex ?? runHex;

    const traceId =
      (parentHex && this._runs.get(parentHex)?.traceId) ??
      this._runTraceIds.get(parentHex ?? "") ??
      activeParent?.traceId ??
      fallbackTraceId;

    const parentSpanId =
      parentHex !== undefined
        ? deriveSpanId(parentHex)
        : activeParent?.spanId;

    const parentPath =
      (parentHex && this._runs.get(parentHex)?.entityPath) ??
      this._runPaths.get(parentHex ?? "");
    const entityPath = parentPath ? `${parentPath}.${name}` : name;

    this._runs.set(runHex, {
      runId: runHex,
      traceId,
      spanId: deriveSpanId(runHex),
      parentRunId: parentHex,
      parentSpanId,
      name,
      entityPath,
      logType,
      spanKind,
      startTime: hrTime(),
      inputValue: toSerializableValue(inputValue),
      serialized: toSerializableValue(serialized),
      tags,
      metadata,
      framework,
      extraAttributes,
      streamedTokens: [],
    });
  }

  private _buildAttributes(record: RunRecord, outputValue?: unknown): SpanAttributesRecord {
    const attrs: SpanAttributesRecord = {
      [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: record.logType,
      [TL_SPAN_KIND]: record.spanKind,
      [TL_ENTITY_NAME]: record.name,
      [TL_ENTITY_PATH]: record.entityPath,
      [LANGCHAIN_RUN_ID_ATTR]: record.runId,
      [LANGCHAIN_FRAMEWORK_ATTR]: record.framework,
    };
    setIfPresent(attrs, LANGCHAIN_PARENT_RUN_ID_ATTR, record.parentRunId);

    if (this.includeMetadata) {
      setIfPresent(attrs, LANGCHAIN_TAGS_ATTR, safeJsonString(record.tags));
      setIfPresent(attrs, LANGCHAIN_METADATA_ATTR, safeJsonString(record.metadata));
      setIfPresent(attrs, LANGCHAIN_SERIALIZED_ATTR, safeJsonString(record.serialized));
    }

    if (this.includeContent) {
      const inputString = safeJsonString(record.inputValue);
      const outputString = safeJsonString(normalizeOutputForLogging(outputValue));
      setIfPresent(attrs, TL_ENTITY_INPUT, inputString);
      setIfPresent(attrs, DIRECT_INPUT, inputString);
      setIfPresent(attrs, TL_ENTITY_OUTPUT, outputString);
      setIfPresent(attrs, DIRECT_OUTPUT, outputString);
    }

    Object.assign(attrs, record.extraAttributes);
    return attrs;
  }

  private _endRun({
    runId,
    outputValue,
    error,
    extraAttributes,
  }: {
    runId: unknown;
    outputValue?: unknown;
    error?: unknown;
    extraAttributes?: SpanAttributesRecord;
  }): boolean {
    try {
      const runHex = runIdToHex(runId);
      const record = this._runs.get(runHex);
      if (!record) return false;
      this._runs.delete(runHex);

      let resolvedOutput = outputValue;
      if (
        record.streamedTokens.length > 0 &&
        (resolvedOutput === undefined || resolvedOutput === null || resolvedOutput === "")
      ) {
        resolvedOutput = record.streamedTokens.join("");
      }
      if (extraAttributes) {
        Object.assign(record.extraAttributes, extraAttributes);
      }

      const attrs = this._buildAttributes(record, resolvedOutput);
      const errorMessage = error !== undefined && error !== null ? getErrorMessage(error) : undefined;
      if (errorMessage) {
        attrs[ERROR_MESSAGE_ATTR] = errorMessage;
        attrs[STATUS_CODE_ATTR] = 500;
      }

      const span = buildReadableSpan({
        name: record.name,
        traceId: record.traceId,
        spanId: record.spanId,
        parentId: record.parentSpanId,
        startTimeHr: record.startTime,
        endTimeHr: hrTime(),
        attributes: attrs,
        statusCode: errorMessage ? 500 : 200,
        errorMessage,
      });
      this._rememberRun(record);
      return injectSpan(span);
    } catch {
      return false;
    }
  }

  private _emitEventSpan({
    parentRunId,
    name,
    logType = RespanLogType.TASK,
    spanKind = RespanLogType.TASK,
    inputValue,
    outputValue,
    tags,
    metadata,
    error,
    extraAttributes = {},
  }: {
    parentRunId?: unknown;
    name: string;
    logType?: string;
    spanKind?: string;
    inputValue?: unknown;
    outputValue?: unknown;
    tags?: string[];
    metadata?: Record<string, unknown>;
    error?: unknown;
    extraAttributes?: SpanAttributesRecord;
  }): boolean {
    try {
      const parentHex = parentRunId !== undefined && parentRunId !== null
        ? runIdToHex(parentRunId)
        : undefined;
      const activeParent = parentHex ? undefined : getActiveOtelParent();
      const traceId =
        (parentHex && this._runs.get(parentHex)?.traceId) ??
        this._runTraceIds.get(parentHex ?? "") ??
        activeParent?.traceId ??
        parentHex ??
        generateTraceId();
      const parentSpanId =
        parentHex !== undefined
          ? deriveSpanId(parentHex)
          : activeParent?.spanId;
      const parentPath =
        (parentHex && this._runs.get(parentHex)?.entityPath) ??
        this._runPaths.get(parentHex ?? "");
      const entityPath = parentPath ? `${parentPath}.${name}` : name;
      const record: RunRecord = {
        runId: runIdToHex(`${traceId}:${parentHex ?? ""}:${name}:${Date.now()}:${Math.random()}`),
        traceId,
        spanId: deriveSpanId(traceId, parentHex, name, Date.now(), Math.random()),
        parentRunId: parentHex,
        parentSpanId,
        name,
        entityPath,
        logType,
        spanKind,
        startTime: hrTime(),
        inputValue: toSerializableValue(inputValue),
        tags,
        metadata,
        framework: detectFramework({ tags, metadata, name }),
        extraAttributes,
        streamedTokens: [],
      };
      const attrs = this._buildAttributes(record, outputValue);
      const errorMessage = error !== undefined && error !== null ? getErrorMessage(error) : undefined;
      if (errorMessage) {
        attrs[ERROR_MESSAGE_ATTR] = errorMessage;
        attrs[STATUS_CODE_ATTR] = 500;
      }
      return injectSpan(buildReadableSpan({
        name,
        traceId,
        spanId: record.spanId,
        parentId: parentSpanId,
        startTimeHr: record.startTime,
        endTimeHr: hrTime(),
        attributes: attrs,
        statusCode: errorMessage ? 500 : 200,
        errorMessage,
      }));
    } catch {
      return false;
    }
  }

  handleChainStart(
    serialized: unknown,
    inputs: unknown,
    runId: unknown,
    parentRunId?: unknown,
    tags?: unknown,
    metadata?: unknown,
    _runType?: unknown,
    runName?: unknown,
  ): void {
    const isRoot = parentRunId === undefined || parentRunId === null;
    this._startRun({
      runId,
      parentRunId,
      name: extractName(serialized, "chain", runName),
      logType: isRoot ? RespanLogType.WORKFLOW : RespanLogType.TASK,
      spanKind: isRoot ? RespanLogType.WORKFLOW : RespanLogType.TASK,
      inputValue: inputs,
      serialized,
      tags: normalizeTags(tags),
      metadata: normalizeMetadata(metadata),
    });
  }

  handleChainEnd(outputs: unknown, runId: unknown): void {
    this._endRun({ runId, outputValue: toSerializableValue(outputs) });
  }

  handleChainError(error: unknown, runId: unknown): void {
    this._endRun({ runId, error });
  }

  handleChatModelStart(
    serialized: unknown,
    messages: unknown,
    runId: unknown,
    parentRunId?: unknown,
    _extraParams?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
  ): void {
    const normalizedMessages = normalizeChatMessages(messages);
    const firstConversation = normalizedMessages[0] ?? [];
    const normalizedMetadata = normalizeMetadata(metadata);
    const extraAttributes: SpanAttributesRecord = {
      [RespanSpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
    };
    const model = extractModel(serialized, undefined, normalizedMetadata);
    setIfPresent(extraAttributes, RespanSpanAttributes.GEN_AI_REQUEST_MODEL, model);
    setIfPresent(extraAttributes, DIRECT_MODEL, model);
    for (const [index, message] of firstConversation.entries()) {
      for (const [key, value] of Object.entries(message)) {
        setIfPresent(
          extraAttributes,
          `${GEN_AI_PROMPT_PREFIX}.${index}.${key}`,
          isPlainRecord(value) || Array.isArray(value) ? safeJsonString(value) : value,
        );
      }
    }
    setIfPresent(
      extraAttributes,
      RespanSpanAttributes.RESPAN_SPAN_TOOLS,
      safeJsonString(extractToolNamesFromSerialized(serialized)),
    );

    this._startRun({
      runId,
      parentRunId,
      name: extractName(serialized, "chat_model", runName),
      logType: RespanLogType.CHAT,
      spanKind: RespanLogType.CHAT,
      inputValue: normalizedMessages,
      serialized,
      tags: normalizeTags(tags),
      metadata: normalizedMetadata,
      extraAttributes,
    });
  }

  handleLLMStart(
    serialized: unknown,
    prompts: unknown,
    runId: unknown,
    parentRunId?: unknown,
    _extraParams?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
  ): void {
    const normalizedPrompts = Array.isArray(prompts) ? prompts : [prompts];
    const normalizedMetadata = normalizeMetadata(metadata);
    const extraAttributes: SpanAttributesRecord = {
      [RespanSpanAttributes.LLM_REQUEST_TYPE]: "completion",
    };
    const model = extractModel(serialized, undefined, normalizedMetadata);
    setIfPresent(extraAttributes, RespanSpanAttributes.GEN_AI_REQUEST_MODEL, model);
    setIfPresent(extraAttributes, DIRECT_MODEL, model);
    for (const [index, prompt] of normalizedPrompts.entries()) {
      extraAttributes[`${GEN_AI_PROMPT_PREFIX}.${index}.role`] = "user";
      extraAttributes[`${GEN_AI_PROMPT_PREFIX}.${index}.content`] = String(prompt ?? "");
    }

    this._startRun({
      runId,
      parentRunId,
      name: extractName(serialized, "llm", runName),
      logType: RespanLogType.TEXT,
      spanKind: "completion",
      inputValue: normalizedPrompts,
      serialized,
      tags: normalizeTags(tags),
      metadata: normalizedMetadata,
      extraAttributes,
    });
  }

  handleLLMNewToken(token: string, _idx?: unknown, runId?: unknown): void {
    const resolvedRunId = runId ?? (typeof _idx === "number" ? undefined : _idx);
    if (resolvedRunId === undefined || token === undefined || token === null) return;
    const record = this._runs.get(runIdToHex(resolvedRunId));
    if (record) {
      record.streamedTokens.push(String(token));
    }
  }

  handleLLMEnd(output: unknown, runId: unknown): void {
    const runHex = runIdToHex(runId);
    const record = this._runs.get(runHex);
    const extracted = extractLlmOutput(output);
    const outputPayload = normalizeOutputForLogging(extracted.outputPayload);
    const completionMessages = normalizeOutputForLogging(extracted.completionMessages) as Record<string, unknown>[];
    const extraAttributes: SpanAttributesRecord = {};
    if (record) {
      const model = extractModel(record.serialized, output, record.metadata);
      setIfPresent(extraAttributes, RespanSpanAttributes.GEN_AI_REQUEST_MODEL, model);
      setIfPresent(extraAttributes, DIRECT_MODEL, model);
    }

    for (const [index, message] of completionMessages.entries()) {
      for (const [key, value] of Object.entries(message)) {
        setIfPresent(
          extraAttributes,
          `${GEN_AI_COMPLETION_PREFIX}.${index}.${key}`,
          isPlainRecord(value) || Array.isArray(value) ? safeJsonString(value) : value,
        );
      }
    }

    const toolCalls = extractToolCallsFromMessages(completionMessages);
    setIfPresent(extraAttributes, RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS, toolCalls);

    const usage = extractUsage(output);
    setIfPresent(extraAttributes, RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, usage.promptTokens);
    setIfPresent(extraAttributes, GEN_AI_USAGE_INPUT_TOKENS, usage.promptTokens);
    setIfPresent(extraAttributes, DIRECT_PROMPT_TOKENS, usage.promptTokens);
    setIfPresent(extraAttributes, RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS, usage.completionTokens);
    setIfPresent(extraAttributes, GEN_AI_USAGE_OUTPUT_TOKENS, usage.completionTokens);
    setIfPresent(extraAttributes, DIRECT_COMPLETION_TOKENS, usage.completionTokens);
    setIfPresent(extraAttributes, GEN_AI_USAGE_TOTAL_TOKENS, usage.totalTokens);
    setIfPresent(extraAttributes, LLM_USAGE_TOTAL_TOKENS, usage.totalTokens);
    setIfPresent(extraAttributes, DIRECT_TOTAL_REQUEST_TOKENS, usage.totalTokens);

    this._endRun({ runId, outputValue: outputPayload, extraAttributes });
  }

  handleLLMError(error: unknown, runId: unknown): void {
    this._endRun({ runId, error });
  }

  handleToolStart(
    serialized: unknown,
    input: unknown,
    runId: unknown,
    parentRunId?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
  ): void {
    const name = extractName(serialized, "tool", runName);
    const inputValue = toSerializableValue(input);
    this._startRun({
      runId,
      parentRunId,
      name,
      logType: RespanLogType.TOOL,
      spanKind: RespanLogType.TOOL,
      inputValue,
      serialized,
      tags: normalizeTags(tags),
      metadata: normalizeMetadata(metadata),
      extraAttributes: {
        [GEN_AI_TOOL_NAME]: name,
        [GEN_AI_TOOL_CALL_ARGUMENTS]: safeJsonString(inputValue),
      },
    });
  }

  handleToolEnd(output: unknown, runId: unknown): void {
    const outputValue = toSerializableValue(output);
    this._endRun({
      runId,
      outputValue,
      extraAttributes: {
        [GEN_AI_TOOL_CALL_RESULT]: safeJsonString(outputValue),
      },
    });
  }

  handleToolError(error: unknown, runId: unknown): void {
    this._endRun({ runId, error });
  }

  handleRetrieverStart(
    serialized: unknown,
    query: unknown,
    runId: unknown,
    parentRunId?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
  ): void {
    this._startRun({
      runId,
      parentRunId,
      name: extractName(serialized, "retriever", runName),
      logType: RespanLogType.TASK,
      spanKind: RespanLogType.TASK,
      inputValue: query,
      serialized,
      tags: normalizeTags(tags),
      metadata: normalizeMetadata(metadata),
    });
  }

  handleRetrieverEnd(documents: unknown, runId: unknown): void {
    this._endRun({ runId, outputValue: toSerializableValue(documents) });
  }

  handleRetrieverError(error: unknown, runId: unknown): void {
    this._endRun({ runId, error });
  }

  handleText(text: string, runId?: unknown): void {
    if (runId === undefined || text === undefined || text === null) return;
    const record = this._runs.get(runIdToHex(runId));
    if (record) {
      record.streamedTokens.push(String(text));
    }
  }

  handleAgentAction(action: unknown, runId?: unknown): void {
    const actionRecord = isPlainRecord(action) ? action : {};
    const toolName = String(actionRecord.tool ?? actionRecord.name ?? "agent_action");
    const toolInput = actionRecord.toolInput ?? actionRecord.tool_input ?? actionRecord.input;
    this._emitEventSpan({
      parentRunId: runId,
      name: toolName,
      logType: RespanLogType.TOOL,
      spanKind: RespanLogType.TOOL,
      inputValue: toolInput,
      outputValue: actionRecord.log,
      extraAttributes: {
        [GEN_AI_TOOL_NAME]: toolName,
        [GEN_AI_TOOL_CALL_ARGUMENTS]: safeJsonString(toSerializableValue(toolInput)),
      },
    });
  }

  handleAgentEnd(action: unknown, runId?: unknown): void {
    const actionRecord = isPlainRecord(action) ? action : {};
    this._emitEventSpan({
      parentRunId: runId,
      name: "agent_finish",
      logType: RespanLogType.AGENT,
      spanKind: RespanLogType.AGENT,
      outputValue: actionRecord.returnValues ?? actionRecord.return_values ?? action,
    });
  }

  handleCustomEvent(
    eventName: string,
    data: unknown,
    runId?: unknown,
    tags?: unknown,
    metadata?: unknown,
  ): void {
    this._emitEventSpan({
      parentRunId: runId,
      name: eventName,
      logType: RespanLogType.TASK,
      spanKind: RespanLogType.TASK,
      inputValue: data,
      outputValue: data,
      tags: normalizeTags(tags),
      metadata: normalizeMetadata(metadata),
    });
  }
}
