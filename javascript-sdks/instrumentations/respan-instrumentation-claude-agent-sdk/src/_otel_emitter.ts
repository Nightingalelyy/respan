import { context, trace } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  RespanLogType,
  RespanSpanAttributes,
  ToolCallSchema,
} from "@respan/respan-sdk";
import {
  buildReadableSpan,
  ensureSpanId,
  ensureTraceId,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "1.0.0";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const CLAUDE_AGENT_INSTRUMENTATION_NAME = "@respan/instrumentation-claude-agent-sdk";
const LLM_REQUEST_FUNCTIONS = "llm.request.functions";
const GEN_AI_COMPLETION_PREFIX = "gen_ai.completion.0";
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;

export interface PendingToolState {
  spanId: string;
  startTime: [number, number];
  toolName: string;
  toolInput: unknown;
}

export interface QueryState {
  agentName: string;
  agentSpanId: string;
  completionTokens?: number;
  errorMessage?: string;
  finalOutput?: unknown;
  inputMessages: Record<string, unknown>[];
  model?: string;
  outputMessages: Record<string, unknown>[];
  parentSpanId?: string;
  pendingTools: Map<string, PendingToolState>;
  prompt: unknown;
  promptCacheCreationTokens?: number;
  promptCacheHitTokens?: number;
  promptTokens?: number;
  sessionId?: string;
  startTime: [number, number];
  statusCode: number;
  toolCalls: Record<string, unknown>[];
  toolDefinitions?: Record<string, unknown>[];
  totalCostUsd?: number;
  totalRequestTokens?: number;
  traceId: string;
}

export function createQueryState({
  prompt,
  options,
  agentName,
}: {
  prompt: unknown;
  options?: Record<string, unknown>;
  agentName?: string;
}): QueryState {
  const activeSpan = trace.getSpan(context.active());
  const activeSpanContext = activeSpan?.spanContext();

  return {
    agentName: agentName?.trim() || inferAgentName(options) || "claude-agent-sdk",
    agentSpanId: ensureSpanId(),
    inputMessages: normalizeInputMessages(prompt, options),
    outputMessages: [],
    parentSpanId: activeSpanContext?.spanId,
    pendingTools: new Map(),
    prompt,
    startTime: hrTime(),
    statusCode: 200,
    toolCalls: [],
    toolDefinitions: normalizeConfiguredToolDefinitions(options),
    traceId: ensureTraceId(activeSpanContext?.traceId),
  };
}

function inferAgentName(options?: Record<string, unknown>): string | undefined {
  const candidate =
    options?.agentName ??
    options?.name ??
    options?.agent_name;
  return typeof candidate === "string" && candidate.trim() ? candidate.trim() : undefined;
}

function normalizeInputMessages(
  prompt: unknown,
  options?: Record<string, unknown>,
): Record<string, unknown>[] {
  const messages: Record<string, unknown>[] = [];
  const systemInstructions =
    options?.system ??
    options?.systemPrompt ??
    options?.instructions;

  if (typeof systemInstructions === "string" && systemInstructions.trim()) {
    messages.push({
      role: "system",
      content: systemInstructions,
    });
  }

  if (typeof prompt === "string") {
    messages.push({
      role: "user",
      content: prompt,
    });
    return messages;
  }

  const serializedPrompt = toSerializableValue(prompt);
  if (Array.isArray(serializedPrompt)) {
    messages.push({
      role: "user",
      content: serializedPrompt,
    });
    return messages;
  }

  if (serializedPrompt !== undefined) {
    messages.push({
      role: "user",
      content: serializedPrompt,
    });
  }

  return messages;
}

function normalizeConfiguredToolDefinitions(
  options?: Record<string, unknown>,
): Record<string, unknown>[] | undefined {
  const normalizedTools = [
    ...normalizeToolDefinitions(options?.tools),
    ...normalizeMcpServerToolDefinitions(options?.mcpServers),
  ];

  if (normalizedTools.length === 0) {
    return undefined;
  }

  return dedupeToolDefinitions(normalizedTools);
}

function normalizeToolDefinitions(
  tools: unknown,
): Record<string, unknown>[] {
  if (!Array.isArray(tools)) {
    return [];
  }

  return tools
    .map((tool) => normalizeToolDefinition(tool))
    .filter((tool): tool is Record<string, unknown> => tool !== null);
}

function normalizeMcpServerToolDefinitions(
  mcpServers: unknown,
): Record<string, unknown>[] {
  if (!mcpServers || typeof mcpServers !== "object" || Array.isArray(mcpServers)) {
    return [];
  }

  const normalizedTools: Record<string, unknown>[] = [];
  for (const [serverAlias, serverConfig] of Object.entries(
    mcpServers as Record<string, unknown>,
  )) {
    if (
      !serverConfig ||
      typeof serverConfig !== "object" ||
      Array.isArray(serverConfig)
    ) {
      continue;
    }

    const serverRecord = serverConfig as Record<string, unknown>;
    const registeredTools =
      serverRecord.instance &&
      typeof serverRecord.instance === "object" &&
      !Array.isArray(serverRecord.instance)
        ? (serverRecord.instance as Record<string, unknown>)._registeredTools
        : undefined;

    if (
      !registeredTools ||
      typeof registeredTools !== "object" ||
      Array.isArray(registeredTools)
    ) {
      continue;
    }

    for (const [toolName, toolConfig] of Object.entries(
      registeredTools as Record<string, unknown>,
    )) {
      const normalizedTool = normalizeToolDefinition({
        ...(toolConfig && typeof toolConfig === "object" && !Array.isArray(toolConfig)
          ? (toolConfig as Record<string, unknown>)
          : {}),
        name: `mcp__${serverAlias}__${toolName}`,
      });

      if (normalizedTool) {
        normalizedTools.push(normalizedTool);
      }
    }
  }

  return normalizedTools;
}

function normalizeToolDefinition(
  tool: unknown,
): Record<string, unknown> | null {
  if (typeof tool === "string" && tool) {
    return {
      type: "function",
      function: { name: tool },
    };
  }

  if (!tool || typeof tool !== "object" || Array.isArray(tool)) {
    return null;
  }

  const record = tool as Record<string, unknown>;
  const functionPayload =
    record.function && typeof record.function === "object" && !Array.isArray(record.function)
      ? (record.function as Record<string, unknown>)
      : null;

  if (functionPayload) {
    const functionName = functionPayload.name;
    if (typeof functionName !== "string" || !functionName) {
      return null;
    }

    const normalizedFunction: Record<string, unknown> = { name: functionName };
    for (const key of ["description", "parameters", "strict"]) {
      if (functionPayload[key] !== undefined) {
        normalizedFunction[key] =
          key === "parameters"
            ? normalizeToolParameters(functionPayload[key])
            : toSerializableValue(functionPayload[key]);
      }
    }

    return {
      type: record.type ?? "function",
      function: normalizedFunction,
    };
  }

  const toolName = record.name;
  if (typeof toolName !== "string" || !toolName) {
    return null;
  }

  const normalizedFunction: Record<string, unknown> = { name: toolName };
  if (record.description !== undefined) {
    normalizedFunction.description = toSerializableValue(record.description);
  }
  const parameters = record.input_schema ?? record.inputSchema ?? record.parameters;
  if (parameters !== undefined) {
    normalizedFunction.parameters = normalizeToolParameters(parameters);
  }

  return {
    type: record.type ?? "function",
    function: normalizedFunction,
  };
}

function normalizeToolParameters(parameters: unknown): unknown {
  const jsonSchema =
    extractJsonSchema(parameters) ?? extractZodLikeJsonSchema(parameters);
  return toSerializableValue(jsonSchema ?? parameters);
}

function extractJsonSchema(parameters: unknown): unknown | undefined {
  if (!parameters || typeof parameters !== "object" || Array.isArray(parameters)) {
    return undefined;
  }

  const record = parameters as Record<string, unknown> & {
    toJSONSchema?: (...args: unknown[]) => unknown;
  };

  if (typeof record.toJSONSchema === "function") {
    try {
      const jsonSchema = record.toJSONSchema();
      if (jsonSchema !== undefined) {
        return jsonSchema;
      }
    } catch {
      // Fall back to serializing a stripped-down version of the original object.
    }
  }

  return undefined;
}

function extractZodLikeJsonSchema(parameters: unknown): unknown | undefined {
  if (!isZodLikeSchema(parameters)) {
    return undefined;
  }

  const schemaRecord = parameters as Record<string, unknown>;
  const schemaDef =
    schemaRecord.def && typeof schemaRecord.def === "object" && !Array.isArray(schemaRecord.def)
      ? (schemaRecord.def as Record<string, unknown>)
      : {};
  const schemaType =
    typeof schemaRecord.type === "string"
      ? schemaRecord.type
      : typeof schemaDef.type === "string"
        ? schemaDef.type
        : undefined;

  switch (schemaType) {
    case "object": {
      const shape =
        schemaDef.shape &&
        typeof schemaDef.shape === "object" &&
        !Array.isArray(schemaDef.shape)
          ? (schemaDef.shape as Record<string, unknown>)
          : {};
      const properties: Record<string, unknown> = {};
      const required: string[] = [];

      for (const [key, fieldSchema] of Object.entries(shape)) {
        const normalizedField =
          extractZodLikeJsonSchema(fieldSchema) ?? toSerializableValue(fieldSchema);
        if (normalizedField !== undefined) {
          properties[key] = normalizedField;
        }

        if (!isOptionalZodLikeSchema(fieldSchema)) {
          required.push(key);
        }
      }

      return {
        type: "object",
        properties,
        ...(required.length > 0 ? { required } : {}),
      };
    }
    case "optional":
      return extractZodLikeJsonSchema(schemaDef.innerType);
    case "string":
    case "number":
    case "boolean":
      return { type: schemaType };
    case "array": {
      const items =
        extractZodLikeJsonSchema(schemaDef.element) ??
        extractZodLikeJsonSchema(schemaDef.innerType);
      return {
        type: "array",
        ...(items !== undefined ? { items } : {}),
      };
    }
    default:
      return undefined;
  }
}

function isOptionalZodLikeSchema(schema: unknown): boolean {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    return false;
  }

  const record = schema as Record<string, unknown>;
  if (record.type === "optional") {
    return true;
  }

  return Boolean(
    record.def &&
    typeof record.def === "object" &&
    !Array.isArray(record.def) &&
    (record.def as Record<string, unknown>).type === "optional",
  );
}

function isZodLikeSchema(schema: unknown): boolean {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    return false;
  }

  const record = schema as Record<string, unknown>;
  const standard =
    record["~standard"] &&
    typeof record["~standard"] === "object" &&
    !Array.isArray(record["~standard"])
      ? (record["~standard"] as Record<string, unknown>)
      : undefined;
  if (standard?.vendor === "zod") {
    return true;
  }

  const def =
    record.def && typeof record.def === "object" && !Array.isArray(record.def)
      ? (record.def as Record<string, unknown>)
      : undefined;
  if (!def || typeof def.type !== "string") {
    return false;
  }

  return true;
}

export function trackClaudeMessage(state: QueryState, message: unknown): void {
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return;
  }

  const record = message as Record<string, unknown>;
  updateSessionId(state, record.session_id ?? record.sessionId);

  switch (record.type) {
    case "system":
      handleSystemMessage(state, record);
      break;
    case "assistant":
      handleAssistantMessage(state, record);
      break;
    case "result":
      handleResultMessage(state, record);
      break;
    case "stream_event":
      updateSessionId(state, record.session_id ?? record.sessionId);
      break;
    default:
      break;
  }
}

function handleSystemMessage(state: QueryState, message: Record<string, unknown>): void {
  const data =
    message.data && typeof message.data === "object" && !Array.isArray(message.data)
      ? (message.data as Record<string, unknown>)
      : undefined;
  updateSessionId(state, data?.session_id ?? data?.sessionId ?? data?.id);
}

function handleAssistantMessage(state: QueryState, message: Record<string, unknown>): void {
  const payload = resolveAssistantPayload(message);
  const model = payload.model ?? message.model;
  if (typeof model === "string" && model) {
    state.model = model;
  }

  updateUsageFromMessage(
    state,
    payload.usage && typeof payload.usage === "object" && !Array.isArray(payload.usage)
      ? (payload.usage as Record<string, unknown>)
      : message.usage && typeof message.usage === "object" && !Array.isArray(message.usage)
        ? (message.usage as Record<string, unknown>)
      : undefined,
  );

  const rawContent = payload.content ?? message.content;
  const content = Array.isArray(rawContent)
    ? toSerializableValue(rawContent)
    : toSerializableValue(rawContent);
  if (hasMeaningfulContent(content)) {
    state.outputMessages.push({
      role: "assistant",
      content,
    });
  }

  if (Array.isArray(rawContent)) {
    for (const block of rawContent) {
      const toolCall = normalizeToolCall(block);
      if (toolCall) {
        addToolCall(state, toolCall);
      }
    }
  }
}

function handleResultMessage(state: QueryState, message: Record<string, unknown>): void {
  updateUsageFromMessage(
    state,
    message.usage && typeof message.usage === "object" && !Array.isArray(message.usage)
      ? (message.usage as Record<string, unknown>)
      : undefined,
  );

  if (typeof message.total_cost_usd === "number") {
    state.totalCostUsd = message.total_cost_usd;
  }

  if (message.is_error) {
    state.statusCode = 500;
    state.errorMessage = `agent_result_error:${String(message.subtype ?? "error")}`;
  }

  const outputValue =
    message.result ??
    message.structured_output;
  if (
    outputValue !== undefined &&
    !hasRenderableAssistantOutput(state.outputMessages)
  ) {
    state.finalOutput = toSerializableValue(outputValue);
    state.outputMessages.push({
      role: "assistant",
      content: toSerializableValue(outputValue),
    });
  } else if (outputValue !== undefined) {
    state.finalOutput = toSerializableValue(outputValue);
  }
}

function resolveAssistantPayload(
  message: Record<string, unknown>,
): Record<string, unknown> {
  const nestedMessage = message.message;
  if (
    nestedMessage &&
    typeof nestedMessage === "object" &&
    !Array.isArray(nestedMessage)
  ) {
    return nestedMessage as Record<string, unknown>;
  }
  return message;
}

function hasMeaningfulContent(value: unknown): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

function hasRenderableAssistantOutput(
  messages: Record<string, unknown>[],
): boolean {
  return messages.some((message) => hasRenderableContent(message.content));
}

function hasRenderableContent(value: unknown): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.some((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return hasRenderableContent(item);
      }

      const record = item as Record<string, unknown>;
      if (record.type === "thinking" || record.type === "tool_use") {
        return false;
      }
      if (record.type === "text") {
        return typeof record.text === "string" && record.text.trim().length > 0;
      }
      if ("content" in record) {
        return hasRenderableContent(record.content);
      }
      return Object.keys(record).length > 0;
    });
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

function updateUsageFromMessage(
  state: QueryState,
  usage?: Record<string, unknown>,
): void {
  if (!usage) {
    return;
  }

  const rawPromptTokens = coerceInteger(usage.input_tokens ?? usage.prompt_tokens);
  const completionTokens = coerceInteger(
    usage.output_tokens ?? usage.completion_tokens,
  );
  const cacheHitTokens = coerceInteger(usage.cache_read_input_tokens);
  const cacheCreationTokens = coerceInteger(usage.cache_creation_input_tokens);
  const totalTokens = coerceInteger(
    usage.total_tokens ??
      usage.totalTokens,
  );
  let promptTokens = rawPromptTokens;

  if (
    promptTokens !== null &&
    (cacheHitTokens !== null || cacheCreationTokens !== null)
  ) {
    const uncachedPromptTokens =
      promptTokens -
      ((cacheHitTokens ?? 0) + (cacheCreationTokens ?? 0));
    if (uncachedPromptTokens >= 0) {
      promptTokens = uncachedPromptTokens;
    }
  }

  if (promptTokens !== null) {
    state.promptTokens = promptTokens;
  }
  if (completionTokens !== null) {
    state.completionTokens = completionTokens;
  }
  if (cacheHitTokens !== null) {
    state.promptCacheHitTokens = cacheHitTokens;
  }
  if (cacheCreationTokens !== null) {
    state.promptCacheCreationTokens = cacheCreationTokens;
  }

  if (totalTokens !== null) {
    state.totalRequestTokens = totalTokens;
  }
}

function resolveToolUseId(
  input: Record<string, unknown>,
  toolUseId?: string,
  pendingTools?: Map<string, PendingToolState>,
): string {
  const directToolUseId = input.tool_use_id ?? toolUseId;
  if (directToolUseId !== undefined && directToolUseId !== null) {
    const normalizedToolUseId = String(directToolUseId);
    if (normalizedToolUseId) {
      return normalizedToolUseId;
    }
  }

  if (pendingTools && pendingTools.size === 1) {
    const pendingTool = pendingTools.keys().next().value;
    if (pendingTool) {
      return pendingTool;
    }
  }

  return ensureSpanId();
}

export function registerPromptSubmit(
  state: QueryState,
  input: Record<string, unknown>,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
}

export function registerPendingTool(
  state: QueryState,
  input: Record<string, unknown>,
  toolUseId?: string,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
  const resolvedToolUseId = resolveToolUseId(input, toolUseId);
  const toolName = String(input.tool_name ?? "tool");
  const toolInput = input.tool_input;

  state.pendingTools.set(resolvedToolUseId, {
    spanId: ensureSpanId(),
    startTime: hrTime(),
    toolInput,
    toolName,
  });

  addToolCall(
    state,
    createToolCall({
      id: resolvedToolUseId,
      toolName,
      args: toolInput ?? {},
    }),
  );
}

export function emitCompletedTool(
  state: QueryState,
  input: Record<string, unknown>,
  toolUseId?: string,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
  const resolvedToolUseId = resolveToolUseId(
    input,
    toolUseId,
    state.pendingTools,
  );
  const pendingTool =
    state.pendingTools.get(resolvedToolUseId) ?? {
      spanId: ensureSpanId(),
      startTime: hrTime(),
      toolInput: input.tool_input,
      toolName: String(input.tool_name ?? "tool"),
    };
  state.pendingTools.delete(resolvedToolUseId);

  const toolName = String(input.tool_name ?? pendingTool.toolName ?? "tool");
  const toolError =
    typeof input.error === "string" && input.error.trim().length > 0
      ? input.error
      : undefined;
  const toolOutput =
    toolError ??
    input.tool_response ??
    input.tool_result ??
    input.output ??
    "";
  const attrs = baseAttrs(toolName, toolName, RespanLogType.TOOL);
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
    pendingTool.toolInput ?? input.tool_input ?? {},
  );
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(toolOutput);

  if (state.model) {
    attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = state.model;
    attrs.model = state.model;
  }
  if (state.sessionId) {
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.sessionId;
  }

  injectSpan(
    buildClaudeReadableSpan({
      name: `${toolName}.tool`,
      traceId: state.traceId,
      spanId: pendingTool.spanId,
      parentId: state.agentSpanId,
      startTimeHr: pendingTool.startTime,
      endTimeHr: hrTime(),
      attributes: attrs,
      statusCode: toolError ? 500 : undefined,
      errorMessage: toolError,
    }),
  );
}

export function emitAgentSpan(state: QueryState): void {
  const attrs = baseAttrs(state.agentName, state.agentName, RespanLogType.AGENT);
  const dedupedToolCalls = dedupeToolCalls(state.toolCalls);
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[RespanSpanAttributes.GEN_AI_SYSTEM] = "anthropic";
  attrs[RespanSpanAttributes.LLM_SYSTEM] = "anthropic";
  attrs[RespanSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = state.agentName;
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = state.agentName;

  if (state.inputMessages.length > 0) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  }
  const formattedOutput = formatAgentOutput(state);
  if (formattedOutput) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = formattedOutput;
  }
  if (state.model) {
    attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = state.model;
    attrs.model = state.model;
  }
  if (state.promptTokens !== undefined) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = state.promptTokens;
    attrs.prompt_tokens = state.promptTokens;
  }
  if (state.completionTokens !== undefined) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = state.completionTokens;
    attrs.completion_tokens = state.completionTokens;
  }
  if (state.totalRequestTokens !== undefined) {
    attrs.total_request_tokens = state.totalRequestTokens;
  }
  if (state.promptCacheHitTokens !== undefined) {
    attrs.prompt_cache_hit_tokens = state.promptCacheHitTokens;
  }
  if (state.promptCacheCreationTokens !== undefined) {
    attrs.prompt_cache_creation_tokens = state.promptCacheCreationTokens;
  }
  if (state.totalCostUsd !== undefined) {
    attrs.cost = state.totalCostUsd;
  }
  if (state.sessionId) {
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.sessionId;
  }
  if (state.toolDefinitions && state.toolDefinitions.length > 0) {
    attrs.tools = state.toolDefinitions;
    attrs[LLM_REQUEST_FUNCTIONS] = safeJson(state.toolDefinitions);
  }
  if (dedupedToolCalls.length > 0) {
    attrs.tool_calls = dedupedToolCalls;
    attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
    attrs[GEN_AI_COMPLETION_CONTENT] = formattedOutput;
    attrs[GEN_AI_COMPLETION_TOOL_CALLS] = dedupedToolCalls;
    attrs.has_tool_calls = true;
    if (dedupedToolCalls.length > 1) {
      attrs.parallel_tool_calls = true;
    }
  }

  injectSpan(
    buildClaudeReadableSpan({
      name: `${state.agentName}.agent`,
      traceId: state.traceId,
      spanId: state.agentSpanId,
      parentId: state.parentSpanId,
      startTimeHr: state.startTime,
      endTimeHr: hrTime(),
      attributes: attrs,
      statusCode: state.statusCode,
      errorMessage: state.errorMessage,
    }),
  );
}

function formatAgentOutput(state: QueryState): string {
  if (state.finalOutput !== undefined) {
    return stringifyOutputValue(state.finalOutput);
  }

  for (let index = state.outputMessages.length - 1; index >= 0; index -= 1) {
    const formatted = stringifyOutputValue(state.outputMessages[index]?.content);
    if (formatted) {
      return formatted;
    }
  }

  return "";
}

function stringifyOutputValue(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  if (Array.isArray(value)) {
    const parts = value
      .map((item) => stringifyOutputValue(item))
      .filter((part) => part.trim().length > 0);
    return parts.join("\n");
  }

  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (record.type === "thinking" || record.type === "tool_use") {
      return "";
    }
    if (record.type === "text" && typeof record.text === "string") {
      return record.text;
    }
    if ("content" in record) {
      return stringifyOutputValue(record.content);
    }
    return safeJson(record);
  }

  return String(value);
}

function createToolCall({
  id,
  toolName,
  args,
}: {
  id: string;
  toolName: string;
  args: unknown;
}): Record<string, unknown> {
  const parsedToolCall = ToolCallSchema.safeParse({
    type: "function",
    id,
    name: toolName,
    args: toSerializableValue(args),
  });
  if (parsedToolCall.success) {
    const normalizedToolCall = {
      ...(parsedToolCall.data as Record<string, unknown>),
    };
    delete normalizedToolCall.name;
    delete normalizedToolCall.args;
    return normalizedToolCall;
  }

  return {
    id,
    type: "function",
    function: {
      name: toolName,
      arguments: safeJson(args ?? {}),
    },
  };
}

function normalizeToolCall(block: unknown): Record<string, unknown> | null {
  if (!block || typeof block !== "object" || Array.isArray(block)) {
    return null;
  }

  const record = block as Record<string, unknown>;
  if (record.type !== "tool_use") {
    return null;
  }

  const toolName = record.name;
  if (typeof toolName !== "string" || !toolName) {
    return null;
  }

  return createToolCall({
    id: String(record.id ?? record.tool_use_id ?? ""),
    toolName,
    args: record.input ?? {},
  });
}

function addToolCall(state: QueryState, toolCall: Record<string, unknown>): void {
  state.toolCalls.push(toolCall);
}

function dedupeToolDefinitions(
  toolDefinitions: Record<string, unknown>[],
): Record<string, unknown>[] {
  const seen = new Set<string>();
  const deduped: Record<string, unknown>[] = [];

  for (const toolDefinition of toolDefinitions) {
    const functionPayload =
      toolDefinition.function &&
      typeof toolDefinition.function === "object" &&
      !Array.isArray(toolDefinition.function)
        ? (toolDefinition.function as Record<string, unknown>)
        : {};
    const name = functionPayload.name;
    if (typeof name !== "string" || !name) {
      deduped.push(toolDefinition);
      continue;
    }
    if (seen.has(name)) {
      continue;
    }
    seen.add(name);
    deduped.push(toolDefinition);
  }

  return deduped;
}

function dedupeToolCalls(
  toolCalls: Record<string, unknown>[],
): Record<string, unknown>[] {
  const seen = new Set<string>();
  const deduped: Record<string, unknown>[] = [];

  for (const toolCall of toolCalls) {
    const functionPayload =
      toolCall.function &&
      typeof toolCall.function === "object" &&
      !Array.isArray(toolCall.function)
        ? (toolCall.function as Record<string, unknown>)
        : {};
    const signature = safeJson([
      toolCall.id ?? "",
      functionPayload.name ?? "",
      functionPayload.arguments ?? "",
    ]);
    if (seen.has(signature)) {
      continue;
    }
    seen.add(signature);
    deduped.push(toolCall);
  }

  return deduped;
}

function updateSessionId(state: QueryState, rawSessionId: unknown): void {
  if (rawSessionId === undefined || rawSessionId === null) {
    return;
  }
  const sessionId = String(rawSessionId);
  if (sessionId) {
    state.sessionId = sessionId;
  }
}

function baseAttrs(
  entityName: string,
  entityPath: string,
  logType: string,
): Record<string, unknown> {
  return {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
  };
}

function buildClaudeReadableSpan(
  options: Parameters<typeof buildReadableSpan>[0],
): ReadableSpan {
  const span = buildReadableSpan(options) as ReadableSpan & {
    instrumentationLibrary?: {
      name: string;
      version?: string;
    };
  };

  span.instrumentationLibrary = {
    name: CLAUDE_AGENT_INSTRUMENTATION_NAME,
    version: PACKAGE_VERSION,
  };
  return span;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(toSerializableValue(value), (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return String(value);
  }
}

function toSerializableValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return undefined;
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => toSerializableValue(item))
      .filter((item) => item !== undefined);
  }
  if (typeof value === "object") {
    const normalizedObject: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, itemValue]) => {
      const normalizedValue = toSerializableValue(itemValue);
      if (normalizedValue !== undefined) {
        normalizedObject[key] = normalizedValue;
      }
    });
    return normalizedObject;
  }
  if (typeof value === "function" || typeof value === "symbol") {
    return undefined;
  }
  return String(value);
}

function coerceInteger(value: unknown): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return null;
  }
  return Math.trunc(numericValue);
}
