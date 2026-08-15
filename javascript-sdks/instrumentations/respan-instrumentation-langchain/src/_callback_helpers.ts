import { context, trace } from "@opentelemetry/api";
import { ensureSpanId, ensureTraceId } from "@respan/tracing";

export const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
export const STATUS_CODE_ATTR = "status_code";

export const LANGCHAIN_RUN_ID_ATTR = "langchain.run_id";
export const LANGCHAIN_PARENT_RUN_ID_ATTR = "langchain.parent_run_id";
export const LANGCHAIN_FRAMEWORK_ATTR = "langchain.framework";
export const LANGCHAIN_TAGS_ATTR = "langchain.tags";
export const LANGCHAIN_METADATA_ATTR = "langchain.metadata";
export const LANGCHAIN_SERIALIZED_ATTR = "langchain.serialized";

export const DIRECT_INPUT = "input";
export const DIRECT_OUTPUT = "output";
export const DIRECT_MODEL = "model";
export const DIRECT_PROMPT_TOKENS = "prompt_tokens";
export const DIRECT_COMPLETION_TOKENS = "completion_tokens";
export const DIRECT_TOTAL_REQUEST_TOKENS = "total_request_tokens";
export const DIRECT_TOOLS = "tools";

const JSON_CODE_FENCE_RE =
  /^\s*(?<fence>`{3,}|~{3,})[ \t]*(?<language>jsonc?)?[ \t]*\r?\n(?<body>.*?)(?:\r?\n)?\k<fence>\s*$/is;

const EMPTY_VALUES = new Set<unknown>([undefined, null, ""]);

export type HrTimeTuple = [number, number];
export type FrameworkName = "langchain" | "langgraph" | "langflow";
export interface LangChainCallbackConfig {
  callbacks?: unknown;
}

export type SpanAttributesRecord = Record<string, unknown>;

export interface RespanCallbackHandlerOptions {
  includeContent?: boolean;
  includeMetadata?: boolean;
  groupLangflowRootRuns?: boolean;
  maxCachedRuns?: number;
}

interface ActiveParent {
  traceId: string;
  spanId: string;
}

interface MessageLikeRecord extends Record<string, unknown> {
  _getType?: () => unknown;
  getType?: () => unknown;
}

export interface RunRecord {
  runId: string;
  traceId: string;
  spanId: string;
  parentRunId?: string;
  parentSpanId?: string;
  name: string;
  entityPath: string;
  workflowName: string;
  logType: string;
  spanKind: string;
  startTime: HrTimeTuple;
  inputValue?: unknown;
  serialized?: unknown;
  tags?: string[];
  metadata?: Record<string, unknown>;
  framework: FrameworkName;
  extraAttributes: SpanAttributesRecord;
  streamedTokens: string[];
}

export function generateTraceId(): string {
  return ensureTraceId();
}

export function runIdToHex(runId: unknown): string {
  if (runId === undefined || runId === null || runId === "") {
    return generateTraceId();
  }

  const value = String(runId);
  const compactUuid = value.replace(/-/g, "");
  if (/^[0-9a-f]{32}$/i.test(compactUuid)) {
    return compactUuid.toLowerCase();
  }
  return ensureTraceId(value);
}

export function deriveSpanId(...parts: unknown[]): string {
  const id = ensureSpanId(parts.map(String).join("|"));
  return /^0+$/.test(id) ? "0000000000000001" : id;
}

export function getActiveOtelParent(): ActiveParent | undefined {
  const spanContext = trace.getSpan(context.active())?.spanContext();
  if (!spanContext?.traceId || !spanContext?.spanId) {
    return undefined;
  }
  if (/^0+$/.test(spanContext.traceId) || /^0+$/.test(spanContext.spanId)) {
    return undefined;
  }
  return {
    traceId: spanContext.traceId,
    spanId: spanContext.spanId,
  };
}

export function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function toSerializableValue(
  value: unknown,
  seen = new WeakSet<object>(),
): unknown {
  if (
    value === undefined ||
    value === null ||
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
    return value.map((item) => toSerializableValue(item, seen));
  }
  if (typeof value === "object") {
    if (seen.has(value)) {
      return "[Circular]";
    }
    seen.add(value);

    const withJson = value as { toJSON?: () => unknown };
    if (typeof withJson.toJSON === "function") {
      try {
        return toSerializableValue(withJson.toJSON(), seen);
      } catch {
        // Fall through to enumerable properties.
      }
    }

    const output: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (typeof item === "function" || typeof item === "symbol" || item === undefined) {
        continue;
      }
      output[key] = toSerializableValue(item, seen);
    }
    return output;
  }
  return String(value);
}

export function safeJsonString(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function setIfPresent(
  attrs: SpanAttributesRecord,
  key: string,
  value: unknown,
): void {
  if (EMPTY_VALUES.has(value)) return;
  if (Array.isArray(value) && value.length === 0) return;
  attrs[key] = value;
}

function stripJsonCodeFence(value: string): string {
  const match = JSON_CODE_FENCE_RE.exec(value);
  if (!match?.groups) {
    return value;
  }

  const body = match.groups.body.trim();
  if (match.groups.language) {
    return body;
  }

  try {
    JSON.parse(body);
    return body;
  } catch {
    return value;
  }
}

export function normalizeOutputForLogging(value: unknown): unknown {
  if (typeof value === "string") {
    return stripJsonCodeFence(value);
  }
  if (Array.isArray(value)) {
    return value.map(normalizeOutputForLogging);
  }
  if (isPlainRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        normalizeOutputForLogging(item),
      ]),
    );
  }
  return value;
}

function normalizeRole(role: unknown): string {
  const normalized = String(role ?? "unknown").toLowerCase();
  switch (normalized) {
    case "human":
    case "chat":
      return "user";
    case "ai":
      return "assistant";
    case "function":
      return "tool";
    default:
      return normalized;
  }
}

function messageToDict(message: unknown): Record<string, unknown> {
  const raw = toSerializableValue(message);
  const rawRecord = isPlainRecord(raw) ? raw : {};
  const original: MessageLikeRecord = isPlainRecord(message)
    ? message
    : {};
  const role =
    rawRecord.role ??
    rawRecord.type ??
    original?.role ??
    original?.type ??
    (typeof original?._getType === "function" ? original._getType() : undefined) ??
    (typeof original?.getType === "function" ? original.getType() : undefined);

  const content = rawRecord.content ?? original?.content;
  const output: Record<string, unknown> = {
    role: normalizeRole(role),
    content: toSerializableValue(content),
  };

  for (const key of ["id", "name", "tool_call_id"]) {
    const item = rawRecord[key] ?? original?.[key];
    if (!EMPTY_VALUES.has(item)) {
      output[key] = toSerializableValue(item);
    }
  }

  const toolCalls = rawRecord.tool_calls ?? rawRecord.toolCalls ?? original?.tool_calls ?? original?.toolCalls;
  if (!EMPTY_VALUES.has(toolCalls)) {
    output.tool_calls = toSerializableValue(toolCalls);
  }

  const additionalKwargs =
    rawRecord.additional_kwargs ??
    rawRecord.additionalKwargs ??
    original?.additional_kwargs ??
    original?.additionalKwargs;
  if (isPlainRecord(additionalKwargs)) {
    for (const key of ["tool_calls", "toolCalls", "function_call", "functionCall"]) {
      if (additionalKwargs[key] !== undefined && output[key] === undefined) {
        output[key] = toSerializableValue(additionalKwargs[key]);
      }
    }
  }

  return output;
}

export function normalizeChatMessages(messages: unknown): Record<string, unknown>[][] {
  const conversations = Array.isArray(messages) ? messages : [messages];
  return conversations.map((conversation) => {
    const items = Array.isArray(conversation) ? conversation : [conversation];
    return items.map(messageToDict);
  });
}

export function extractName(
  serialized: unknown,
  fallback: string,
  explicitName?: unknown,
): string {
  if (typeof explicitName === "string" && explicitName.trim()) {
    return explicitName.trim();
  }
  if (typeof serialized === "string" && serialized.trim()) {
    return serialized.trim();
  }
  if (isPlainRecord(serialized)) {
    for (const key of ["name", "id", "type"]) {
      const value = serialized[key];
      if (typeof value === "string" && value.trim()) {
        return key === "id" && value.includes(".")
          ? value.split(".").at(-1) ?? fallback
          : value.trim();
      }
      if (Array.isArray(value) && value.length > 0) {
        return String(value[value.length - 1]);
      }
    }
    const kwargs = serialized.kwargs;
    if (isPlainRecord(kwargs)) {
      for (const key of ["name", "model", "modelName", "model_name", "repo_id"]) {
        const value = kwargs[key];
        if (typeof value === "string" && value.trim()) {
          return value.trim();
        }
      }
    }
  }
  return fallback;
}

export function detectFramework({
  serialized,
  tags,
  metadata,
  name,
}: {
  serialized?: unknown;
  tags?: string[];
  metadata?: Record<string, unknown>;
  name?: string;
}): FrameworkName {
  const haystack: string[] = [];
  haystack.push(...(tags ?? []).map((tag) => String(tag).toLowerCase()));
  if (metadata) {
    haystack.push(...Object.keys(metadata).map((key) => key.toLowerCase()));
    haystack.push(
      ...Object.values(metadata)
        .filter((value) => typeof value === "string")
        .map((value) => String(value).toLowerCase()),
    );
  }
  if (serialized !== undefined) {
    haystack.push(safeJsonString(toSerializableValue(serialized)).toLowerCase());
  }
  if (name) {
    haystack.push(name.toLowerCase());
  }

  const text = haystack.join(" ");
  if (text.includes("langflow")) return "langflow";
  if (text.includes("langgraph") || text.includes("__pregel") || text.includes("graph:")) {
    return "langgraph";
  }
  return "langchain";
}

function coerceInteger(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) {
    return Math.trunc(Number(value));
  }
  return undefined;
}

function firstDefined(...values: unknown[]): unknown {
  return values.find((value) => value !== undefined && value !== null);
}

export function extractUsage(response: unknown): {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
} {
  const payloads: unknown[] = [];
  const responseRecord = isPlainRecord(response) ? response : {};
  const llmOutput = responseRecord.llmOutput ?? responseRecord.llm_output;
  if (isPlainRecord(llmOutput)) {
    payloads.push(
      llmOutput.tokenUsage,
      llmOutput.token_usage,
      llmOutput.usage,
      llmOutput.usage_metadata,
      llmOutput.estimatedTokenUsage,
      llmOutput,
    );
  }
  payloads.push(responseRecord.usage, responseRecord.usage_metadata);

  const generations = responseRecord.generations;
  if (Array.isArray(generations) && generations.length > 0) {
    const firstBatch = Array.isArray(generations[0]) ? generations[0] : [generations[0]];
    const firstGeneration = firstBatch[0];
    const message = isPlainRecord(firstGeneration) ? firstGeneration.message : undefined;
    if (isPlainRecord(message)) {
      payloads.push(message.usage_metadata, message.usageMetadata);
      const responseMetadata = message.response_metadata ?? message.responseMetadata;
      if (isPlainRecord(responseMetadata)) {
        payloads.push(
          responseMetadata.tokenUsage,
          responseMetadata.token_usage,
          responseMetadata.usage,
          responseMetadata.usage_metadata,
          responseMetadata,
        );
      }
    }
  }

  for (const payload of payloads) {
    if (!isPlainRecord(payload)) continue;
    const promptTokens = coerceInteger(firstDefined(
      payload.prompt_tokens,
      payload.promptTokens,
      payload.input_tokens,
      payload.inputTokens,
    ));
    const completionTokens = coerceInteger(firstDefined(
      payload.completion_tokens,
      payload.completionTokens,
      payload.output_tokens,
      payload.outputTokens,
    ));
    let totalTokens = coerceInteger(firstDefined(
      payload.total_tokens,
      payload.totalTokens,
      payload.total,
    ));
    if (totalTokens === undefined && (promptTokens !== undefined || completionTokens !== undefined)) {
      totalTokens = (promptTokens ?? 0) + (completionTokens ?? 0);
    }
    if (promptTokens !== undefined || completionTokens !== undefined || totalTokens !== undefined) {
      const usage: {
        promptTokens?: number;
        completionTokens?: number;
        totalTokens?: number;
      } = {};
      if (promptTokens !== undefined) usage.promptTokens = promptTokens;
      if (completionTokens !== undefined) usage.completionTokens = completionTokens;
      if (totalTokens !== undefined) usage.totalTokens = totalTokens;
      return usage;
    }
  }

  return {};
}

export function extractModel(
  serialized?: unknown,
  response?: unknown,
  metadata?: Record<string, unknown>,
): string | undefined {
  const candidates: unknown[] = [];
  if (metadata) {
    candidates.push(metadata.ls_model_name, metadata.model, metadata.modelName, metadata.model_name);
  }
  if (isPlainRecord(serialized)) {
    const kwargs = serialized.kwargs;
    if (isPlainRecord(kwargs)) {
      candidates.push(kwargs.model, kwargs.modelName, kwargs.model_name, kwargs.repo_id);
    }
    candidates.push(serialized.model, serialized.modelName, serialized.model_name);
  }
  const responseRecord = isPlainRecord(response) ? response : {};
  const llmOutput = responseRecord.llmOutput ?? responseRecord.llm_output;
  if (isPlainRecord(llmOutput)) {
    candidates.push(llmOutput.model_name, llmOutput.modelName, llmOutput.model);
  }

  return candidates.find(
    (candidate): candidate is string => typeof candidate === "string" && candidate.trim().length > 0,
  );
}

function generationToMessage(generation: unknown): Record<string, unknown> {
  const generationRecord = isPlainRecord(generation) ? generation : {};
  const message = generationRecord.message;
  if (message !== undefined) {
    return messageToDict(message);
  }
  const text = generationRecord.text;
  if (text !== undefined) {
    return { role: "assistant", content: toSerializableValue(text) };
  }
  return { role: "assistant", content: toSerializableValue(generation) };
}

export function extractLlmOutput(response: unknown): {
  outputPayload: unknown;
  completionMessages: Record<string, unknown>[];
} {
  const responseRecord = isPlainRecord(response) ? response : {};
  const generations = responseRecord.generations;
  if (Array.isArray(generations)) {
    const normalizedBatches: Record<string, unknown>[][] = [];
    const completionMessages: Record<string, unknown>[] = [];
    for (const batch of generations) {
      const batchItems = Array.isArray(batch) ? batch : [batch];
      const normalizedBatch = batchItems.map(generationToMessage);
      normalizedBatches.push(normalizedBatch);
      completionMessages.push(...normalizedBatch);
    }
    return { outputPayload: normalizedBatches, completionMessages };
  }

  const serialized = toSerializableValue(response);
  const message =
    isPlainRecord(serialized) && serialized.content !== undefined
      ? serialized
      : { role: "assistant", content: serialized };
  return {
    outputPayload: serialized,
    completionMessages: [message as Record<string, unknown>],
  };
}

export function extractToolCallsFromMessages(
  messages: Record<string, unknown>[],
): Record<string, unknown>[] | undefined {
  const toolCalls: Record<string, unknown>[] = [];
  for (const message of messages) {
    const rawCalls = message.tool_calls ?? message.toolCalls;
    if (!Array.isArray(rawCalls)) continue;
    for (const rawCall of rawCalls) {
      if (!isPlainRecord(rawCall)) continue;
      const call: Record<string, unknown> = { ...rawCall };
      if (!call.type) call.type = "function";
      if (!call.id && (call.toolCallId || call.tool_call_id)) {
        call.id = call.toolCallId ?? call.tool_call_id;
      }
      if (!call.function && call.name) {
        call.function = {
          name: call.name,
          arguments: safeJsonString(call.args ?? call.arguments),
        };
      }
      toolCalls.push(toSerializableValue(call) as Record<string, unknown>);
    }
  }
  return toolCalls.length > 0 ? toolCalls : undefined;
}

export function extractToolNamesFromSerialized(serialized: unknown): string[] | undefined {
  const names = (extractTools({ serialized }) ?? [])
    .map((tool) => {
      const functionDefinition = tool.function;
      if (isPlainRecord(functionDefinition) && typeof functionDefinition.name === "string") {
        return functionDefinition.name;
      }
      return undefined;
    })
    .filter((name): name is string => typeof name === "string" && name.length > 0);
  return names.length > 0 ? names : undefined;
}

function collectToolCandidates(value: unknown, candidates: unknown[]): void {
  if (!isPlainRecord(value)) return;

  const rawTools = value.tools ?? value.functions;
  if (Array.isArray(rawTools)) {
    candidates.push(...rawTools);
  }

  for (const key of [
    "invocation_params",
    "invocationParams",
    "kwargs",
    "options",
    "config",
    "model_kwargs",
    "modelKwargs",
  ]) {
    collectToolCandidates(value[key], candidates);
  }
}

function normalizeToolDefinition(tool: unknown): Record<string, unknown> | undefined {
  const serializableTool = toSerializableValue(tool);
  if (!isPlainRecord(serializableTool)) return undefined;

  const functionDefinition = serializableTool.function;
  if (isPlainRecord(functionDefinition) && typeof functionDefinition.name === "string") {
    return {
      ...serializableTool,
      type: typeof serializableTool.type === "string" ? serializableTool.type : "function",
      function: functionDefinition,
    };
  }

  const name = serializableTool.name ?? serializableTool.toolName ?? serializableTool.id;
  if (typeof name !== "string" || !name.trim()) return undefined;

  const parameters =
    serializableTool.parameters ??
    serializableTool.schema ??
    serializableTool.args_schema ??
    serializableTool.argsSchema;

  const normalizedFunction: Record<string, unknown> = {
    name: name.trim(),
  };
  setIfPresent(normalizedFunction, "description", serializableTool.description);
  setIfPresent(normalizedFunction, "parameters", parameters);

  return {
    type: "function",
    function: normalizedFunction,
  };
}

export function extractTools({
  serialized,
  extraParams,
}: {
  serialized?: unknown;
  extraParams?: unknown;
}): Record<string, unknown>[] | undefined {
  const candidates: unknown[] = [];
  collectToolCandidates(extraParams, candidates);
  collectToolCandidates(serialized, candidates);

  const tools = candidates
    .map(normalizeToolDefinition)
    .filter((tool): tool is Record<string, unknown> => tool !== undefined);

  return tools.length > 0 ? tools : undefined;
}

export function normalizeTags(tags?: unknown): string[] | undefined {
  if (!Array.isArray(tags)) return undefined;
  return tags.map(String);
}

export function normalizeMetadata(metadata?: unknown): Record<string, unknown> | undefined {
  if (!isPlainRecord(metadata)) return undefined;
  return { ...metadata };
}

export function trimMap<K, V>(map: Map<K, V>, maxSize: number): void {
  while (map.size > maxSize) {
    const firstKey = map.keys().next().value;
    if (firstKey === undefined) return;
    map.delete(firstKey);
  }
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
