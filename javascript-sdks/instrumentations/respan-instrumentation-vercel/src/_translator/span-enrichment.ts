import {
  AI_RESPONSE_MS_TO_FINISH,
  AI_PREFIX,
  AI_MODEL_PROVIDER,
  AI_TELEMETRY_METADATA_PREFIX,
  AI_USAGE_CACHED_INPUT_TOKENS,
  AI_USAGE_COMPLETION_TOKENS,
  AI_USAGE_INPUT_TOKENS,
  AI_USAGE_OUTPUT_TOKENS,
  AI_USAGE_PROMPT_TOKENS,
  AI_USAGE_TOTAL_TOKENS,
  CUSTOMER_EMAIL,
  CUSTOMER_ID,
  CUSTOMER_NAME,
  GEN_AI_SYSTEM,
  GEN_AI_PROVIDER_NAME,
  GEN_AI_RESPONSE_FINISH_REASONS,
  GEN_AI_RESPONSE_ID,
  GEN_AI_USAGE_COST,
  GEN_AI_USAGE_GENERATION_TIME,
  GEN_AI_USAGE_INPUT_TOKENS,
  GEN_AI_USAGE_OUTPUT_TOKENS,
  GEN_AI_USAGE_TTFT,
  GEN_AI_USAGE_TYPE,
  GEN_AI_REQUEST_MODEL,
  GEN_AI_USAGE_WARNINGS,
  GEN_AI_USAGE_COMPLETION_TOKENS,
  GEN_AI_USAGE_PROMPT_TOKENS,
  LLM_USAGE_CACHE_READ_INPUT_TOKENS,
  LLM_USAGE_TOTAL_TOKENS,
  RESPAN_SPAN_HANDOFFS,
  RESPAN_SPAN_TOOL_CALLS,
  RESPAN_SPAN_TOOLS,
  SESSION_ID,
  THREAD_ID,
  TRACE_GROUP_ID,
  metadataKey,
  normalizeModel,
  setDefault,
  type SpanAttributes,
} from "./shared.js";

export function enrichMetadata(attrs: SpanAttributes): void {
  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith(AI_TELEMETRY_METADATA_PREFIX)) {
      continue;
    }

    const cleanKey = key.slice(AI_TELEMETRY_METADATA_PREFIX.length);
    switch (cleanKey) {
      case "customer_identifier":
        setDefault(attrs, CUSTOMER_ID, String(value));
        break;
      case "customer_email":
        setDefault(attrs, CUSTOMER_EMAIL, String(value));
        break;
      case "customer_name":
        setDefault(attrs, CUSTOMER_NAME, String(value));
        break;
      case "session_identifier":
        setDefault(attrs, SESSION_ID, String(value));
        break;
      case "thread_identifier":
        setDefault(attrs, THREAD_ID, String(value));
        break;
      case "trace_group_identifier":
        setDefault(attrs, TRACE_GROUP_ID, String(value));
        break;
      case "customer_params": {
        // customer_params is a JSON-stringified object (Vercel telemetry
        // metadata values must be flat scalars, so users serialize the object).
        // Documented shape uses `email` / `name` (matching the Customer columns
        // in the UI); accept the legacy `customer_email` / `customer_name`
        // aliases too so older integrations keep working.
        try {
          const parsed = typeof value === "string" ? JSON.parse(value) : value;
          if (parsed?.customer_identifier) setDefault(attrs, CUSTOMER_ID, parsed.customer_identifier);
          const email = parsed?.email ?? parsed?.customer_email;
          if (email) setDefault(attrs, CUSTOMER_EMAIL, email);
          const name = parsed?.name ?? parsed?.customer_name;
          if (name) setDefault(attrs, CUSTOMER_NAME, name);
        } catch {
          // Ignore malformed customer_params metadata.
        }
        break;
      }
      case "prompt_unit_price":
        setDefault(attrs, metadataKey("prompt_unit_price"), String(value));
        break;
      case "completion_unit_price":
        setDefault(attrs, metadataKey("completion_unit_price"), String(value));
        break;
      case "userId":
        setDefault(attrs, CUSTOMER_ID, String(value));
        setDefault(attrs, metadataKey(cleanKey), String(value ?? ""));
        break;
      default:
        setDefault(attrs, metadataKey(cleanKey), String(value ?? ""));
        break;
    }
  }
}

export function enrichModel(attrs: SpanAttributes, modelId: unknown): void {
  if (!modelId) {
    return;
  }

  const model = normalizeModel(String(modelId));
  setDefault(attrs, GEN_AI_REQUEST_MODEL, model);
}

function normalizeSystem(system: unknown): string | undefined {
  if (!system) {
    return undefined;
  }

  const value = String(system).trim().toLowerCase();
  if (!value) {
    return undefined;
  }

  if (value.includes("openai")) return "openai";
  if (value.includes("anthropic")) return "anthropic";
  if (value.includes("google") || value.includes("gemini")) return "google";
  if (value.includes("bedrock")) return "bedrock";
  if (value.includes("azure")) return "azure";
  if (value.includes("mistral")) return "mistral";
  if (value.includes("cohere")) return "cohere";
  if (value.includes("groq")) return "groq";
  if (value.includes("xai")) return "xai";
  if (value.includes("deepseek")) return "deepseek";

  return value.split(/[.:/]/, 1)[0] || value;
}

export function enrichSystem(attrs: SpanAttributes): void {
  const system = normalizeSystem(attrs[GEN_AI_SYSTEM] ?? attrs[GEN_AI_PROVIDER_NAME] ?? attrs[AI_MODEL_PROVIDER]);
  if (system) {
    setDefault(attrs, GEN_AI_SYSTEM, system);
  }
}

function numberAttr(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }

  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

export function enrichTokens(attrs: SpanAttributes): void {
  const inputTokens =
    attrs[GEN_AI_USAGE_INPUT_TOKENS] ??
    attrs[GEN_AI_USAGE_PROMPT_TOKENS] ??
    attrs[AI_USAGE_INPUT_TOKENS] ??
    attrs[AI_USAGE_PROMPT_TOKENS];
  const outputTokens =
    attrs[GEN_AI_USAGE_OUTPUT_TOKENS] ??
    attrs[GEN_AI_USAGE_COMPLETION_TOKENS] ??
    attrs[AI_USAGE_OUTPUT_TOKENS] ??
    attrs[AI_USAGE_COMPLETION_TOKENS];
  const totalTokens = attrs[LLM_USAGE_TOTAL_TOKENS] ?? attrs[AI_USAGE_TOTAL_TOKENS];
  const cacheReadInputTokens = attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] ?? attrs[AI_USAGE_CACHED_INPUT_TOKENS];

  const promptTokens = numberAttr(inputTokens);
  const completionTokens = numberAttr(outputTokens);

  if (promptTokens !== undefined) {
    setDefault(attrs, GEN_AI_USAGE_INPUT_TOKENS, promptTokens);
    setDefault(attrs, GEN_AI_USAGE_PROMPT_TOKENS, promptTokens);
  }
  if (completionTokens !== undefined) {
    setDefault(attrs, GEN_AI_USAGE_OUTPUT_TOKENS, completionTokens);
    setDefault(attrs, GEN_AI_USAGE_COMPLETION_TOKENS, completionTokens);
  }

  const resolvedTotalTokens = numberAttr(totalTokens) ?? (
    promptTokens !== undefined && completionTokens !== undefined
      ? promptTokens + completionTokens
      : undefined
  );
  if (resolvedTotalTokens !== undefined) {
    setDefault(attrs, LLM_USAGE_TOTAL_TOKENS, resolvedTotalTokens);
  }

  const resolvedCacheReadInputTokens = numberAttr(cacheReadInputTokens);
  if (resolvedCacheReadInputTokens !== undefined) {
    setDefault(attrs, LLM_USAGE_CACHE_READ_INPUT_TOKENS, resolvedCacheReadInputTokens);
  }
}

export function enrichPerformanceMetrics(attrs: SpanAttributes, spanName: string): void {
  // Streaming is a first-class promoted attribute (llm.is_streaming), not metadata.
  setDefault(attrs, "llm.is_streaming", spanName.toLowerCase().includes("stream"));

  const msToFinish = attrs[AI_RESPONSE_MS_TO_FINISH];
  if (msToFinish !== undefined) {
    setDefault(attrs, metadataKey("time_to_first_token"), String(Number(msToFinish) / 1000));
  }

  const cost = attrs[GEN_AI_USAGE_COST];
  if (cost !== undefined) {
    setDefault(attrs, metadataKey("cost"), String(cost));
  }

  const ttft = attrs[GEN_AI_USAGE_TTFT];
  if (ttft !== undefined) {
    setDefault(attrs, metadataKey("ttft"), String(ttft));
  }

  const generationTime = attrs[GEN_AI_USAGE_GENERATION_TIME];
  if (generationTime !== undefined) {
    setDefault(attrs, metadataKey("generation_time"), String(generationTime));
  }

  const warnings = attrs[GEN_AI_USAGE_WARNINGS];
  if (warnings !== undefined) {
    setDefault(attrs, metadataKey("warnings"), String(warnings));
  }

  const responseType = attrs[GEN_AI_USAGE_TYPE];
  if (responseType !== undefined) {
    setDefault(attrs, metadataKey("type"), String(responseType));
  }
}

const NON_CONTRACT_ATTRS_TO_STRIP = [
  "operation.name",
  GEN_AI_RESPONSE_FINISH_REASONS,
  GEN_AI_RESPONSE_ID,
  "service.name",
  "telemetry.sdk.language",
  "telemetry.sdk.name",
  "telemetry.sdk.version",
  "process.pid",
  "process.executable.name",
  "process.executable.path",
  "process.command_args",
  "process.runtime.version",
  "process.runtime.name",
  "process.runtime.description",
  "process.command",
  "process.owner",
  "host.name",
  "host.arch",
  "host.id",
  "otel.scope.name",
  "otel.scope.version",
  "next.span_name",
  "next.span_type",
  "http.url",
  "http.method",
  "net.peer.name",
];

const OFF_CONTRACT_ALIAS_ATTRS_TO_STRIP = [
  "tools",
  "tool_calls",
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  RESPAN_SPAN_TOOLS,
  RESPAN_SPAN_TOOL_CALLS,
  RESPAN_SPAN_HANDOFFS,
];

export function stripRedundantAttrs(attrs: SpanAttributes): void {
  for (const key of NON_CONTRACT_ATTRS_TO_STRIP) {
    delete attrs[key];
  }

  for (const key of OFF_CONTRACT_ALIAS_ATTRS_TO_STRIP) {
    delete attrs[key];
  }

  for (const key of Object.keys(attrs)) {
    if (key.startsWith(AI_PREFIX)) {
      delete attrs[key];
    }
  }
}
