/**
 * Translate OpenInference spans → OpenLLMetry/Traceloop format.
 *
 * This SpanProcessor converts spans produced by OpenInference instrumentors
 * (Haystack, CrewAI, LangChain, Google ADK, etc.) into the Traceloop/OpenLLMetry
 * semantic conventions that the Respan backend expects.
 *
 * The mapping is the exact reverse of Arize's `openinference-instrumentation-openllmetry`
 * package which converts OpenLLMetry → OpenInference.
 * Translation stays additive on the live span; cleanup is reserved for export clones.
 *
 * Arize direction (OpenLLMetry → OI)          | Our reverse (OI → OpenLLMetry)
 * ---------------------------------------------|------------------------------------------
 * traceloop.span.kind → openinference.span.kind | openinference.span.kind → respan.log_type
 * traceloop.entity.input → input.value          | input.value → traceloop.entity.input
 * traceloop.entity.output → output.value        | output.value → traceloop.entity.output
 * gen_ai.prompt.N.* → llm.input_messages.N.*    | llm.input_messages.N.* → gen_ai.prompt.N.*
 * gen_ai.completion.N.* → llm.output_messages.N.*| llm.output_messages.N.* → gen_ai.completion.N.*
 * gen_ai.usage.input_tokens → llm.token_count.prompt     | llm.token_count.prompt → gen_ai.usage.input_tokens
 * gen_ai.usage.output_tokens → llm.token_count.completion | llm.token_count.completion → gen_ai.usage.output_tokens
 * llm.usage.total_tokens → llm.token_count.total         | llm.token_count.total → llm.usage.total_tokens
 * llm.usage.cache_read_input_tokens → llm.token_count.prompt_details.cache_read | (reverse)
 * gen_ai.request.model → llm.invocation_parameters.model | llm.invocation_parameters → gen_ai.request.*
 * gen_ai.request.temperature → llm.invocation_parameters.temperature | (reverse)
 * llm.request.functions → llm.tools             | llm.tools → llm.request.functions
 * gen_ai.system → llm.system                    | llm.system → gen_ai.system
 * gen_ai.provider.name → llm.provider           | llm.provider → gen_ai.provider.name
 */

import type { Context } from "@opentelemetry/api";
import type { SpanProcessor, ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  AGENT_NAME,
  INPUT_VALUE,
  LLM_INVOCATION_PARAMETERS,
  LLM_MODEL_NAME,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_TOTAL,
  LLM_TOOLS,
  OUTPUT_VALUE,
  SemanticConventions as OpenInferenceSemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { RespanSpanAttributes, RespanLogType } from "@respan/respan-sdk";

// ---------------------------------------------------------------------------
// Attribute keys imported from owning semantic-convention packages
// ---------------------------------------------------------------------------
const OI_SPAN_KIND = OpenInferenceSemanticConventions.OPENINFERENCE_SPAN_KIND;
const OI_LLM_MODEL_NAME = LLM_MODEL_NAME;
const OI_LLM_TOKEN_COUNT_PROMPT = LLM_TOKEN_COUNT_PROMPT;
const OI_LLM_TOKEN_COUNT_COMPLETION = LLM_TOKEN_COUNT_COMPLETION;
const RESPAN_LOG_TYPE = RespanSpanAttributes.RESPAN_LOG_TYPE;
const GEN_AI_SYSTEM = ATTR_GEN_AI_SYSTEM;
const GEN_AI_REQUEST_MODEL = ATTR_GEN_AI_REQUEST_MODEL;
const GEN_AI_USAGE_PROMPT_TOKENS = ATTR_GEN_AI_USAGE_PROMPT_TOKENS;
const GEN_AI_USAGE_COMPLETION_TOKENS = ATTR_GEN_AI_USAGE_COMPLETION_TOKENS;
const LLM_REQUEST_TYPE = RespanSpanAttributes.LLM_REQUEST_TYPE;

// ---------------------------------------------------------------------------
// OpenInference attribute keys (not in SDK — OI-specific, used only here)
// ---------------------------------------------------------------------------
const OI_INPUT_VALUE = INPUT_VALUE;
const OI_OUTPUT_VALUE = OUTPUT_VALUE;
const OI_LLM_PROVIDER = LLM_PROVIDER;
const OI_LLM_SYSTEM = LLM_SYSTEM;
const OI_LLM_INVOCATION_PARAMETERS = LLM_INVOCATION_PARAMETERS;
const OI_LLM_TOKEN_COUNT_TOTAL = LLM_TOKEN_COUNT_TOTAL;
const OI_LLM_TOKEN_COUNT_CACHE_READ = LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ;
const OI_LLM_TOOLS = LLM_TOOLS;
const OI_AGENT_NAME = AGENT_NAME;

// ---------------------------------------------------------------------------
// OpenLLMetry wire-format attribute keys (not in SDK — mapping targets only)
// ---------------------------------------------------------------------------
const TL_ENTITY_NAME = "traceloop.entity.name";
const TL_ENTITY_INPUT = "traceloop.entity.input";
const TL_ENTITY_OUTPUT = "traceloop.entity.output";
const TL_ENTITY_PATH = "traceloop.entity.path";
const TL_USAGE_INPUT_TOKENS = ATTR_GEN_AI_USAGE_INPUT_TOKENS;
const TL_USAGE_OUTPUT_TOKENS = ATTR_GEN_AI_USAGE_OUTPUT_TOKENS;
const TL_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens";
const TL_USAGE_CACHE_READ = "llm.usage.cache_read_input_tokens";
const TL_REQUEST_TEMPERATURE = ATTR_GEN_AI_REQUEST_TEMPERATURE;
const TL_REQUEST_TOP_P = ATTR_GEN_AI_REQUEST_TOP_P;
const TL_REQUEST_MAX_TOKENS = ATTR_GEN_AI_REQUEST_MAX_TOKENS;
const TL_TOP_K = "llm.top_k";
const TL_STOP_SEQUENCES = "llm.chat.stop_sequences";
const TL_REPETITION_PENALTY = "llm.request.repetition_penalty";
const TL_FREQUENCY_PENALTY = "llm.frequency_penalty";
const TL_PRESENCE_PENALTY = "llm.presence_penalty";
const TL_PROVIDER_NAME = ATTR_GEN_AI_PROVIDER_NAME;
const TL_REQUEST_FUNCTIONS = "llm.request.functions";
const CLAUDE_AGENT_SDK_SCOPE_NAME =
  "@arizeai/openinference-instrumentation-claude-agent-sdk";
const EMPTY_SCOPE_NAME = "";
const DIRECT_MODEL = "model";
const DIRECT_PROMPT_TOKENS = "prompt_tokens";
const DIRECT_COMPLETION_TOKENS = "completion_tokens";
const DIRECT_TOTAL_REQUEST_TOKENS = "total_request_tokens";

// ---------------------------------------------------------------------------
// Span kind mapping: OpenInference → Traceloop (reverse of Arize)
// ---------------------------------------------------------------------------
const OI_KIND_TO_TRACELOOP: Record<string, string> = {
  LLM: RespanLogType.TASK,
  CHAIN: RespanLogType.WORKFLOW,
  TOOL: RespanLogType.TOOL,
  AGENT: RespanLogType.AGENT,
  EMBEDDING: RespanLogType.TASK,
  RETRIEVER: RespanLogType.TASK,
  RERANKER: RespanLogType.TASK,
  GUARDRAIL: RespanLogType.TASK,
  EVALUATOR: RespanLogType.TASK,
  PROMPT: RespanLogType.TASK,
  UNKNOWN: RespanLogType.TASK,
};

const OI_KIND_TO_LOG_TYPE: Record<string, string> = {
  LLM: RespanLogType.CHAT,
  CHAIN: RespanLogType.WORKFLOW,
  TOOL: RespanLogType.TOOL,
  AGENT: RespanLogType.AGENT,
  EMBEDDING: RespanLogType.EMBEDDING,
  RETRIEVER: RespanLogType.TASK,
  RERANKER: RespanLogType.TASK,
  GUARDRAIL: RespanLogType.GUARDRAIL,
  EVALUATOR: RespanLogType.TASK,
  PROMPT: RespanLogType.TASK,
  UNKNOWN: RespanLogType.TASK,
};

const OI_LLM_REQUEST_KINDS: Record<string, string> = {
  LLM: RespanLogType.CHAT,
  EMBEDDING: RespanLogType.EMBEDDING,
};

const LLM_KINDS = new Set(["LLM", "EMBEDDING"]);

// Invocation parameter key → OpenLLMetry target attribute
const INVOCATION_PARAM_MAP: Record<string, string> = {
  model: GEN_AI_REQUEST_MODEL,
  temperature: TL_REQUEST_TEMPERATURE,
  top_p: TL_REQUEST_TOP_P,
  max_tokens: TL_REQUEST_MAX_TOKENS,
  max_output_tokens: TL_REQUEST_MAX_TOKENS,
  top_k: TL_TOP_K,
  stop_sequences: TL_STOP_SEQUENCES,
  stop: TL_STOP_SEQUENCES,
  repetition_penalty: TL_REPETITION_PENALTY,
  frequency_penalty: TL_FREQUENCY_PENALTY,
  presence_penalty: TL_PRESENCE_PENALTY,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function safeJsonStr(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function parseJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

/**
 * Convert OI indexed messages to OpenLLMetry gen_ai.prompt.N / gen_ai.completion.N.
 *
 * OI format:
 *   llm.input_messages.0.message.role = "user"
 *   llm.input_messages.0.message.content = "hello"
 *   llm.input_messages.0.message.tool_calls.0.tool_call.function.name = "get_weather"
 *
 * OpenLLMetry format:
 *   gen_ai.prompt.0.role = "user"
 *   gen_ai.prompt.0.content = "hello"
 *   gen_ai.prompt.0.tool_calls.0.function.name = "get_weather"
 */
function oiMessagesToOpenLLMetry(
  attrs: Record<string, any>,
  oiPrefix: string,
  genAiPrefix: string,
): void {
  const buckets = new Map<number, Map<string, any>>();

  for (const [key, val] of Object.entries(attrs)) {
    if (!key.startsWith(oiPrefix)) continue;
    const rest = key.slice(oiPrefix.length);
    const dotIdx = rest.indexOf(".");
    const idxStr = dotIdx === -1 ? rest : rest.slice(0, dotIdx);
    if (!/^\d+$/.test(idxStr)) continue;
    const idx = parseInt(idxStr, 10);
    const field = dotIdx === -1 ? "" : rest.slice(dotIdx + 1);
    if (!buckets.has(idx)) buckets.set(idx, new Map());
    buckets.get(idx)!.set(field, val);
  }

  const sortedIndices = [...buckets.keys()].sort((a, b) => a - b);

  for (const idx of sortedIndices) {
    const raw = buckets.get(idx)!;
    const target = `${genAiPrefix}.${idx}`;

    const role = raw.get("message.role");
    if (role) attrs[`${target}.role`] = role;

    const content = raw.get("message.content");
    if (content !== undefined) attrs[`${target}.content`] = content;
    else {
      const contentBlocks = new Map<number, Map<string, any>>();

      for (const [fieldKey, fieldVal] of raw) {
        if (!fieldKey.startsWith("message.contents.")) continue;
        const blockRest = fieldKey.slice("message.contents.".length);
        const blockDotIdx = blockRest.indexOf(".");
        if (blockDotIdx === -1) continue;

        const blockIdxStr = blockRest.slice(0, blockDotIdx);
        if (!/^\d+$/.test(blockIdxStr)) continue;

        const blockIdx = parseInt(blockIdxStr, 10);
        let blockField = blockRest.slice(blockDotIdx + 1);
        if (blockField.startsWith("message_content.")) {
          blockField = blockField.slice("message_content.".length);
        }

        if (!contentBlocks.has(blockIdx)) {
          contentBlocks.set(blockIdx, new Map());
        }
        contentBlocks.get(blockIdx)!.set(blockField, fieldVal);
      }

      if (contentBlocks.size > 0) {
        const orderedBlocks = [...contentBlocks.keys()]
          .sort((a, b) => a - b)
          .map((blockIdx) => Object.fromEntries(contentBlocks.get(blockIdx)!));

        const textParts = orderedBlocks
          .map((block) =>
            typeof block.text === "string" ? block.text : undefined
          )
          .filter((part): part is string => part !== undefined);

        attrs[`${target}.content`] =
          textParts.length > 0 && textParts.length === orderedBlocks.length
            ? textParts.join("\n")
            : safeJsonStr(orderedBlocks);
      }
    }

    // Tool calls
    for (const [fieldKey, fieldVal] of raw) {
      if (fieldKey.startsWith("message.tool_calls.")) {
        const tcRest = fieldKey.slice("message.tool_calls.".length);
        const tcDotIdx = tcRest.indexOf(".");
        if (tcDotIdx === -1) continue;
        const tcIdx = tcRest.slice(0, tcDotIdx);
        if (!/^\d+$/.test(tcIdx)) continue;
        let tcField = tcRest.slice(tcDotIdx + 1);
        if (tcField.startsWith("tool_call.")) tcField = tcField.slice("tool_call.".length);
        attrs[`${target}.tool_calls.${tcIdx}.${tcField}`] = fieldVal;
      }
    }

    // Function call fields
    const funcName = raw.get("message.function_call_name");
    if (funcName) attrs[`${target}.function_call.name`] = funcName;
    const funcArgs = raw.get("message.function_call_arguments_json");
    if (funcArgs) attrs[`${target}.function_call.arguments`] = funcArgs;

    // Finish reason
    const finishReason = raw.get("message.finish_reason");
    if (finishReason) attrs[`${target}.finish_reason`] = finishReason;
  }
}

function setDefault(attrs: Record<string, any>, key: string, value: any): void {
  if (attrs[key] === undefined) attrs[key] = value;
}

function firstDefined<T>(...values: Array<T | undefined>): T | undefined {
  for (const value of values) {
    if (value !== undefined) return value;
  }
  return undefined;
}

function getInstrumentationScopeName(span: ReadableSpan): string {
  return (
    ((span as any).instrumentationScope?.name as string | undefined) ??
    ((span as any).instrumentationLibrary?.name as string | undefined) ??
    ""
  );
}

function buildCleanedAttrs(
  attrs: Record<string, any>,
  exactKeys: Set<string>,
  prefixes: string[],
): Record<string, any> {
  return Object.fromEntries(
    Object.entries(attrs).filter(
      ([key]) =>
        !exactKeys.has(key) &&
        !prefixes.some((prefix) => key.startsWith(prefix)),
    ),
  );
}

const REDUNDANT_OI_EXACT_KEYS = new Set([
  OI_SPAN_KIND,
  OI_INPUT_VALUE,
  "input.mime_type",
  OI_OUTPUT_VALUE,
  "output.mime_type",
  OI_LLM_MODEL_NAME,
  OI_LLM_PROVIDER,
  OI_LLM_SYSTEM,
  OI_LLM_TOKEN_COUNT_PROMPT,
  OI_LLM_TOKEN_COUNT_COMPLETION,
  OI_LLM_TOKEN_COUNT_TOTAL,
  OI_LLM_TOKEN_COUNT_CACHE_READ,
  OI_AGENT_NAME,
]);

const REDUNDANT_OI_LLM_EXACT_KEYS = new Set([
  OI_LLM_INVOCATION_PARAMETERS,
  OI_LLM_TOOLS,
]);

const REDUNDANT_OI_LLM_PREFIXES = [
  "llm.input_messages.",
  "llm.output_messages.",
];

const REDUNDANT_OI_PREFIXES = [
  "llm.token_count.",
];

const REDUNDANT_OTEL_EXACT_KEYS = new Set([
  "otel.scope.name",
  "otel.scope.version",
]);

const REDUNDANT_OTEL_PREFIXES = [
  "process.",
  "host.",
  "telemetry.sdk.",
];

const OTEL_RESOURCE_NOISE_PREFIXES = [
  "process.",
  "host.",
  "telemetry.sdk.",
];

type CloneableReadableSpan = ReadableSpan & {
  _attributes?: Record<string, any>;
  resource?: { attributes?: Record<string, any> };
  instrumentationLibrary?: { name?: string; version?: string };
  instrumentationScope?: { name?: string; version?: string };
};

function getOpenInferenceKindUpper(
  attrs: Record<string, any> | undefined,
): string | null {
  const oiKind = attrs?.[OI_SPAN_KIND];
  if (oiKind === undefined) return null;
  return String(oiKind).toUpperCase();
}

function filterResourceAttributes(
  attrs: Record<string, any> | undefined,
): Record<string, any> | undefined {
  if (!attrs) return attrs;

  return Object.fromEntries(
    Object.entries(attrs).filter(
      ([key]) =>
        !OTEL_RESOURCE_NOISE_PREFIXES.some((prefix) => key.startsWith(prefix)),
    ),
  );
}

function buildExportAttrs(
  attrs: Record<string, any>,
  oiKindUpper: string,
): Record<string, any> {
  const oiExactKeys = new Set(REDUNDANT_OI_EXACT_KEYS);
  const oiPrefixes = [...REDUNDANT_OI_PREFIXES];

  if (LLM_KINDS.has(oiKindUpper)) {
    for (const key of REDUNDANT_OI_LLM_EXACT_KEYS) {
      oiExactKeys.add(key);
    }
    oiPrefixes.push(...REDUNDANT_OI_LLM_PREFIXES);
  }

  const withoutOiAttrs = buildCleanedAttrs(attrs, oiExactKeys, oiPrefixes);

  return buildCleanedAttrs(
    withoutOiAttrs,
    REDUNDANT_OTEL_EXACT_KEYS,
    REDUNDANT_OTEL_PREFIXES,
  );
}

function cloneSpan(span: ReadableSpan): CloneableReadableSpan {
  return Object.assign(
    Object.create(Object.getPrototypeOf(span)),
    span,
  ) as CloneableReadableSpan;
}

export function isOpenInferenceSpan(span: ReadableSpan): boolean {
  return getOpenInferenceKindUpper((span as any).attributes) !== null;
}

export function prepareOpenInferenceSpanForExport(
  span: ReadableSpan,
): ReadableSpan {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  const oiKindUpper = getOpenInferenceKindUpper(attrs);
  if (!attrs || !oiKindUpper) return span;

  const clonedSpan = cloneSpan(span);
  const cleanedAttrs = buildExportAttrs(attrs, oiKindUpper);
  (clonedSpan as any).attributes = cleanedAttrs;

  if ((span as any)._attributes && typeof (span as any)._attributes === "object") {
    clonedSpan._attributes = cleanedAttrs;
  }

  const resource = (span as CloneableReadableSpan).resource;
  if (resource) {
    clonedSpan.resource = {
      ...resource,
      attributes: filterResourceAttributes(resource.attributes) ?? {},
    };
  }

  const instrumentationLibrary = (span as CloneableReadableSpan).instrumentationLibrary;
  if (instrumentationLibrary) {
    clonedSpan.instrumentationLibrary = {
      ...instrumentationLibrary,
      name: EMPTY_SCOPE_NAME,
      version: EMPTY_SCOPE_NAME,
    };
  }

  const instrumentationScope = (span as CloneableReadableSpan).instrumentationScope;
  if (instrumentationScope) {
    clonedSpan.instrumentationScope = {
      ...instrumentationScope,
      name: EMPTY_SCOPE_NAME,
      version: EMPTY_SCOPE_NAME,
    };
  }

  return clonedSpan;
}

export function translateOpenInferenceSpan(span: ReadableSpan): void {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs) return;

  const oiKindUpper = getOpenInferenceKindUpper(attrs);
  if (!oiKindUpper) return;

  const instrumentationScopeName = getInstrumentationScopeName(span);

  // Do not set traceloop.span.kind for translated OpenInference spans.
  // In Respan's composite processor that attribute is reserved for
  // user-decorated spans and would incorrectly force auto spans to root.
  setDefault(attrs, RESPAN_LOG_TYPE, OI_KIND_TO_LOG_TYPE[oiKindUpper] ?? "task");

  if (OI_LLM_REQUEST_KINDS[oiKindUpper]) {
    setDefault(attrs, LLM_REQUEST_TYPE, OI_LLM_REQUEST_KINDS[oiKindUpper]);
  }

  // Claude Agent SDK currently emits AGENT spans with exact token counts but the
  // current backend only preserves those typed token fields for chat/completion-
  // shaped spans. Mark these specific Claude Agent SDK spans as chat requests for
  // ingestion so token/model fields survive without affecting other agent SDKs.
  if (
    oiKindUpper === "AGENT" &&
    instrumentationScopeName === CLAUDE_AGENT_SDK_SCOPE_NAME
  ) {
    setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.CHAT);
  }

  // --- Entity name ---
  const entityName = attrs[OI_AGENT_NAME] ?? span.name;
  setDefault(attrs, TL_ENTITY_NAME, entityName);

  // --- Entity path ---
  setDefault(attrs, TL_ENTITY_PATH, OI_KIND_TO_TRACELOOP[oiKindUpper] !== "workflow" ? span.name : "");

  // --- Input / output ---
  if (attrs[OI_INPUT_VALUE] !== undefined) {
    setDefault(attrs, TL_ENTITY_INPUT, safeJsonStr(attrs[OI_INPUT_VALUE]));
  }
  if (attrs[OI_OUTPUT_VALUE] !== undefined) {
    setDefault(attrs, TL_ENTITY_OUTPUT, safeJsonStr(attrs[OI_OUTPUT_VALUE]));
  }

  // --- Model name ---
  if (attrs[OI_LLM_MODEL_NAME] !== undefined) {
    setDefault(attrs, GEN_AI_REQUEST_MODEL, attrs[OI_LLM_MODEL_NAME]);
  }

  // --- System / provider ---
  if (attrs[OI_LLM_SYSTEM] !== undefined) {
    setDefault(attrs, GEN_AI_SYSTEM, String(attrs[OI_LLM_SYSTEM]).toLowerCase());
  }
  if (attrs[OI_LLM_PROVIDER] !== undefined) {
    setDefault(attrs, TL_PROVIDER_NAME, String(attrs[OI_LLM_PROVIDER]).toLowerCase());
    setDefault(attrs, GEN_AI_SYSTEM, String(attrs[OI_LLM_PROVIDER]).toLowerCase());
  }

  // --- Token counts ---
  if (attrs[OI_LLM_TOKEN_COUNT_PROMPT] !== undefined) {
    setDefault(attrs, GEN_AI_USAGE_PROMPT_TOKENS, attrs[OI_LLM_TOKEN_COUNT_PROMPT]);
    setDefault(attrs, TL_USAGE_INPUT_TOKENS, attrs[OI_LLM_TOKEN_COUNT_PROMPT]);
  }
  if (attrs[OI_LLM_TOKEN_COUNT_COMPLETION] !== undefined) {
    setDefault(attrs, GEN_AI_USAGE_COMPLETION_TOKENS, attrs[OI_LLM_TOKEN_COUNT_COMPLETION]);
    setDefault(attrs, TL_USAGE_OUTPUT_TOKENS, attrs[OI_LLM_TOKEN_COUNT_COMPLETION]);
  }
  if (attrs[OI_LLM_TOKEN_COUNT_TOTAL] !== undefined) {
    setDefault(attrs, TL_USAGE_TOTAL_TOKENS, attrs[OI_LLM_TOKEN_COUNT_TOTAL]);
  }
  if (attrs[OI_LLM_TOKEN_COUNT_CACHE_READ] !== undefined) {
    setDefault(attrs, TL_USAGE_CACHE_READ, attrs[OI_LLM_TOKEN_COUNT_CACHE_READ]);
  }

  // Direct overrides for Respan ingestion. These ensure non-chat OpenInference
  // spans like AGENT still populate typed model/token columns even though the
  // backend's generic span path only auto-promotes input/output.
  const directModel = firstDefined(
    attrs[GEN_AI_REQUEST_MODEL],
    attrs[OI_LLM_MODEL_NAME],
  );
  if (directModel !== undefined) {
    setDefault(attrs, DIRECT_MODEL, directModel);
  }

  const directPromptTokens = firstDefined(
    attrs[GEN_AI_USAGE_PROMPT_TOKENS],
    attrs[TL_USAGE_INPUT_TOKENS],
    attrs[OI_LLM_TOKEN_COUNT_PROMPT],
  );
  if (directPromptTokens !== undefined) {
    setDefault(attrs, DIRECT_PROMPT_TOKENS, directPromptTokens);
  }

  const directCompletionTokens = firstDefined(
    attrs[GEN_AI_USAGE_COMPLETION_TOKENS],
    attrs[TL_USAGE_OUTPUT_TOKENS],
    attrs[OI_LLM_TOKEN_COUNT_COMPLETION],
  );
  if (directCompletionTokens !== undefined) {
    setDefault(attrs, DIRECT_COMPLETION_TOKENS, directCompletionTokens);
  }

  const directTotalRequestTokens = firstDefined(
    attrs[TL_USAGE_TOTAL_TOKENS],
    attrs[OI_LLM_TOKEN_COUNT_TOTAL],
    directPromptTokens !== undefined && directCompletionTokens !== undefined
      ? Number(directPromptTokens) + Number(directCompletionTokens)
      : undefined,
  );
  if (directTotalRequestTokens !== undefined) {
    setDefault(attrs, DIRECT_TOTAL_REQUEST_TOKENS, directTotalRequestTokens);
  }

  // --- LLM-specific: messages, invocation params, tools ---
  if (LLM_KINDS.has(oiKindUpper)) {
    oiMessagesToOpenLLMetry(attrs, "llm.input_messages.", "gen_ai.prompt");
    oiMessagesToOpenLLMetry(attrs, "llm.output_messages.", "gen_ai.completion");

    const invParamsRaw = attrs[OI_LLM_INVOCATION_PARAMETERS];
    if (invParamsRaw) {
      const params = parseJson(invParamsRaw);
      if (params && typeof params === "object" && !Array.isArray(params)) {
        for (const [key, val] of Object.entries(params as Record<string, any>)) {
          const targetAttr = INVOCATION_PARAM_MAP[key];
          if (targetAttr) setDefault(attrs, targetAttr, val);
        }
      }
    }

    if (attrs[OI_LLM_TOOLS] !== undefined) {
      setDefault(attrs, TL_REQUEST_FUNCTIONS, attrs[OI_LLM_TOOLS]);
    }
  }
}

// ---------------------------------------------------------------------------
// Main processor
// ---------------------------------------------------------------------------

/**
 * SpanProcessor that translates OpenInference attributes to OpenLLMetry/Traceloop.
 *
 * Detects OI spans by `openinference.span.kind` and enriches them with the
 * Traceloop attributes the Respan backend expects. All mappings are the exact
 * reverse of Arize's openinference-instrumentation-openllmetry.
 *
 * Translation is additive on the live span. Export-only cleanup happens in
 * `prepareOpenInferenceSpanForExport()` so downstream callbacks can still
 * inspect the original OpenInference attributes when needed.
 */
export class OpenInferenceTranslator implements SpanProcessor {
  onStart(_span: Span, _parentContext: Context): void {
    // no-op
  }

  onEnd(span: ReadableSpan): void {
    translateOpenInferenceSpan(span);
  }

  async shutdown(): Promise<void> {}
  async forceFlush(): Promise<void> {}
}
