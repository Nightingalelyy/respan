import type { Context, Span } from "@opentelemetry/api";
import type {
  ReadableSpan,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_REQUEST_TOP_K,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";
import { RespanSpanAttributes, RespanLogType } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  capturedAttributes,
  captureBlocked,
  captureLanes,
  guardSpan,
  guardExportClone,
  safeDescriptorValue,
  type CaptureOptions,
} from "./_capture.js";

type Attributes = Record<string, any>;
const LOG_TYPES: Record<string, string> = {
  [OpenInferenceSpanKind.LLM]: RespanLogType.CHAT,
  [OpenInferenceSpanKind.CHAIN]: RespanLogType.WORKFLOW,
  [OpenInferenceSpanKind.AGENT]: RespanLogType.AGENT,
  [OpenInferenceSpanKind.TOOL]: RespanLogType.TOOL,
  [OpenInferenceSpanKind.EMBEDDING]: RespanLogType.EMBEDDING,
  [OpenInferenceSpanKind.GUARDRAIL]: RespanLogType.GUARDRAIL,
};
function json(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  try {
    return JSON.stringify(value);
  } catch {
    return undefined;
  }
}
function parse(value: unknown): any {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}
function asInput(value: unknown): string | undefined {
  return typeof value === "string" ? value : json(value);
}
function record(value: unknown): Attributes {
  const result = parse(value);
  return result && typeof result === "object" && !Array.isArray(result)
    ? result
    : {};
}
function indexed(attrs: Attributes, prefix: string): Map<number, Attributes> {
  const result = new Map<number, Attributes>();
  for (const key of Object.keys(attrs)) {
    if (!key.startsWith(`${prefix}.`)) continue;
    const rest = key.slice(prefix.length + 1);
    const dot = rest.indexOf(".");
    if (dot < 0 || !/^\d+$/.test(rest.slice(0, dot))) continue;
    const index = Number(rest.slice(0, dot));
    const item = result.get(index) ?? {};
    item[rest.slice(dot + 1)] = attrs[key];
    result.set(index, item);
  }
  return new Map([...result].sort(([left], [right]) => left - right));
}
function messages(attrs: Attributes, prefix: string): Attributes[] {
  return [...indexed(attrs, prefix).values()].map((raw) => {
    const message: Attributes = {};
    for (const [source, target] of [
      [SemanticConventions.MESSAGE_ROLE, "role"],
      [SemanticConventions.MESSAGE_NAME, "name"],
      [SemanticConventions.MESSAGE_CONTENT, "content"],
      [SemanticConventions.MESSAGE_TOOL_CALL_ID, "tool_call_id"],
    ])
      if (raw[source] !== undefined) message[target] = raw[source];
    const blocks = [
      ...indexed(raw, SemanticConventions.MESSAGE_CONTENTS).values(),
    ].map((block) => {
      const result: Attributes = {};
      for (const [key, value] of Object.entries(block)) {
        const field = key.startsWith("message_content.")
          ? key.slice("message_content.".length)
          : key;
        if (
          key ===
            `${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}` ||
          field === "image.url"
        ) {
          result.type = "image_url";
          result.image_url = { url: value };
        } else result[field] = value;
      }
      return result;
    });
    if (message.content === undefined && blocks.length) {
      message.content = blocks.every(
        (block) => block.type === "text" && typeof block.text === "string",
      )
        ? blocks.map((block) => block.text).join("\n")
        : blocks;
    }
    const calls = [
      ...indexed(raw, SemanticConventions.MESSAGE_TOOL_CALLS).values(),
    ].map((call) => {
      const invocation: Attributes = { type: "function", function: {} };
      if (call[SemanticConventions.TOOL_CALL_ID] !== undefined)
        invocation.id = call[SemanticConventions.TOOL_CALL_ID];
      if (call[SemanticConventions.TOOL_CALL_FUNCTION_NAME] !== undefined)
        invocation.function.name =
          call[SemanticConventions.TOOL_CALL_FUNCTION_NAME];
      if (
        call[SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON] !==
        undefined
      )
        invocation.function.arguments =
          call[SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON];
      if (call[SemanticConventions.TOOL_CALL_REASONING_SIGNATURE] !== undefined)
        invocation.reasoning_signature =
          call[SemanticConventions.TOOL_CALL_REASONING_SIGNATURE];
      return invocation;
    });
    if (calls.length) message.tool_calls = calls;
    if (raw[SemanticConventions.MESSAGE_FUNCTION_CALL_NAME] !== undefined)
      message.function_call = {
        name: raw[SemanticConventions.MESSAGE_FUNCTION_CALL_NAME],
        arguments:
          raw[SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON],
      };
    return message;
  });
}
function tools(attrs: Attributes): unknown[] | undefined {
  const indexedTools = [
    ...indexed(attrs, SemanticConventions.LLM_TOOLS).values(),
  ]
    .map((tool) => parse(tool[SemanticConventions.TOOL_JSON_SCHEMA]))
    .filter((tool) => tool !== undefined);
  if (indexedTools.length) return indexedTools;
  const direct = parse(attrs[SemanticConventions.LLM_TOOLS]);
  return Array.isArray(direct) ? direct : undefined;
}
export function isOpenInferenceSpan(span: ReadableSpan): boolean {
  return (
    typeof safeDescriptorValue(
      span.attributes,
      SemanticConventions.OPENINFERENCE_SPAN_KIND,
    ) === "string"
  );
}
/** Add canonical fields while retaining the original producer attributes. */
export function translateOpenInferenceSpan(span: ReadableSpan): void {
  if (captureBlocked(span)) return;
  const raw = (capturedAttributes(span) ?? span.attributes) as Attributes;
  const kindValue = safeDescriptorValue(
    raw,
    SemanticConventions.OPENINFERENCE_SPAN_KIND,
  );
  if (typeof kindValue !== "string") return;
  const kind = kindValue.toUpperCase();
  const lanes = captureLanes(span);
  const canonical: Attributes = {};
  const put = (key: string, value: unknown) => {
    if (value !== undefined) canonical[key] = value;
  };
  put(
    RespanSpanAttributes.RESPAN_LOG_TYPE,
    LOG_TYPES[kind] ?? RespanLogType.TASK,
  );
  put(RespanSpanAttributes.RESPAN_LOG_METHOD, "ts_tracing");
  put(
    SpanAttributes.TRACELOOP_ENTITY_NAME,
    raw[SemanticConventions.AGENT_NAME] ??
      raw[SemanticConventions.TOOL_NAME] ??
      span.name,
  );
  put(
    SpanAttributes.TRACELOOP_ENTITY_PATH,
    raw[SpanAttributes.TRACELOOP_ENTITY_PATH] ?? span.name,
  );
  const llm = kind === OpenInferenceSpanKind.LLM;
  const embedding = kind === OpenInferenceSpanKind.EMBEDDING;
  if (llm || embedding) {
    put(SpanAttributes.LLM_REQUEST_TYPE, embedding ? "embedding" : "chat");
    put(
      ATTR_GEN_AI_REQUEST_MODEL,
      raw[SemanticConventions.LLM_REQUEST_MODEL_NAME] ??
        raw[SemanticConventions.LLM_MODEL_NAME] ??
        raw[SemanticConventions.EMBEDDING_MODEL_NAME],
    );
    put(
      ATTR_GEN_AI_RESPONSE_MODEL,
      raw[SemanticConventions.LLM_RESPONSE_MODEL_NAME],
    );
    if (typeof raw[SemanticConventions.LLM_PROVIDER] === "string")
      put(
        ATTR_GEN_AI_PROVIDER_NAME,
        raw[SemanticConventions.LLM_PROVIDER].toLowerCase(),
      );
    if (
      typeof (
        raw[SemanticConventions.LLM_SYSTEM] ??
        raw[SemanticConventions.LLM_PROVIDER]
      ) === "string"
    )
      put(
        ATTR_GEN_AI_SYSTEM,
        (
          raw[SemanticConventions.LLM_SYSTEM] ??
          raw[SemanticConventions.LLM_PROVIDER]
        ).toLowerCase(),
      );
    for (const [source, targets] of [
      [
        SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
        [ATTR_GEN_AI_USAGE_INPUT_TOKENS, ATTR_GEN_AI_USAGE_PROMPT_TOKENS],
      ],
      [
        SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
        [ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, ATTR_GEN_AI_USAGE_COMPLETION_TOKENS],
      ],
      [
        SemanticConventions.LLM_TOKEN_COUNT_TOTAL,
        [SpanAttributes.LLM_USAGE_TOTAL_TOKENS],
      ],
      [
        SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
        [ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS],
      ],
      [
        SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
        [ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS],
      ],
      [
        SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
        [ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS],
      ],
    ] as const) {
      const value = raw[source];
      if (typeof value === "number" && Number.isFinite(value))
        for (const target of targets) put(target, value);
    }
  }
  const inputMessages =
    llm && lanes.inputs
      ? messages(raw, SemanticConventions.LLM_INPUT_MESSAGES)
      : [];
  const outputMessages =
    llm && lanes.outputs
      ? messages(raw, SemanticConventions.LLM_OUTPUT_MESSAGES)
      : [];
  const definitions = lanes.inputs ? tools(raw) : undefined;
  if (lanes.inputs) {
    const source = raw[SemanticConventions.INPUT_VALUE];
    if (kind === OpenInferenceSpanKind.TOOL && source !== undefined)
      put(
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        json({
          name: canonical[SpanAttributes.TRACELOOP_ENTITY_NAME],
          arguments: parse(source),
        }),
      );
    else
      put(
        SpanAttributes.TRACELOOP_ENTITY_INPUT,
        inputMessages.length ? json(inputMessages) : asInput(source),
      );
    if (llm && definitions !== undefined)
      put(SpanAttributes.LLM_REQUEST_FUNCTIONS, json(definitions));
    if (llm) {
      const parameters = record(
        raw[SemanticConventions.LLM_INVOCATION_PARAMETERS],
      );
      for (const [sourceKey, target] of [
        ["model", ATTR_GEN_AI_REQUEST_MODEL],
        ["temperature", ATTR_GEN_AI_REQUEST_TEMPERATURE],
        ["top_p", ATTR_GEN_AI_REQUEST_TOP_P],
        ["max_tokens", ATTR_GEN_AI_REQUEST_MAX_TOKENS],
        ["max_output_tokens", ATTR_GEN_AI_REQUEST_MAX_TOKENS],
        ["top_k", ATTR_GEN_AI_REQUEST_TOP_K],
        ["stop", SpanAttributes.LLM_CHAT_STOP_SEQUENCES],
        ["stop_sequences", SpanAttributes.LLM_CHAT_STOP_SEQUENCES],
        ["frequency_penalty", SpanAttributes.LLM_FREQUENCY_PENALTY],
        ["presence_penalty", SpanAttributes.LLM_PRESENCE_PENALTY],
      ])
        if (
          parameters[sourceKey] !== undefined &&
          canonical[target] === undefined
        )
          put(target, parameters[sourceKey]);
    }
  }
  if (lanes.outputs)
    put(
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      outputMessages.length
        ? json(outputMessages)
        : asInput(raw[SemanticConventions.OUTPUT_VALUE]),
    );
  if (kind === OpenInferenceSpanKind.TOOL)
    put(
      ATTR_GEN_AI_TOOL_CALL_ID,
      raw[SemanticConventions.TOOL_CALL_ID] ?? raw[SemanticConventions.TOOL_ID],
    );
  if (embedding) {
    const values = [
      ...indexed(raw, SemanticConventions.EMBEDDING_EMBEDDINGS).values(),
    ];
    const texts = values
      .filter(
        (value) => value[SemanticConventions.EMBEDDING_TEXT] !== undefined,
      )
      .map((value) => value[SemanticConventions.EMBEDDING_TEXT]);
    const vectors = values
      .filter(
        (value) => value[SemanticConventions.EMBEDDING_VECTOR] !== undefined,
      )
      .map((value) => value[SemanticConventions.EMBEDDING_VECTOR]);
    if (
      lanes.inputs &&
      texts.length &&
      canonical[SpanAttributes.TRACELOOP_ENTITY_INPUT] === undefined
    )
      put(SpanAttributes.TRACELOOP_ENTITY_INPUT, json(texts));
    if (lanes.outputs && vectors.length)
      put(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json(vectors));
  }
  if (
    kind === OpenInferenceSpanKind.RETRIEVER &&
    lanes.outputs &&
    canonical[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] === undefined
  ) {
    const documents = [
      ...indexed(raw, SemanticConventions.RETRIEVAL_DOCUMENTS).values(),
    ].map((document) => {
      const result: Attributes = {};
      for (const [key, value] of Object.entries(document))
        result[
          key.startsWith("document.") ? key.slice("document.".length) : key
        ] =
          key === SemanticConventions.DOCUMENT_METADATA ? parse(value) : value;
      return result;
    });
    if (documents.length)
      put(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json(documents));
  }
  if (lanes.inputs && lanes.outputs) {
    const metadata = {
      ...record(raw[RespanSpanAttributes.RESPAN_METADATA]),
      ...record(raw[SemanticConventions.METADATA]),
    };
    if (!llm && definitions !== undefined) metadata.tools = definitions;
    const tokenDetails: Attributes = {};
    for (const key of [
      SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
      SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
    ])
      if (typeof raw[key] === "number") tokenDetails[key] = raw[key];
    if (Object.keys(tokenDetails).length) metadata.token_details = tokenDetails;
    for (const key of [
      SemanticConventions.TAG_TAGS,
      SemanticConventions.PROMPT_TEMPLATE_TEMPLATE,
      SemanticConventions.PROMPT_TEMPLATE_VARIABLES,
      SemanticConventions.PROMPT_TEMPLATE_VERSION,
    ])
      if (raw[key] !== undefined) metadata[key] = parse(raw[key]);
    if (Object.keys(metadata).length)
      put(RespanSpanAttributes.RESPAN_METADATA, json(metadata));
    put(
      RespanSpanAttributes.RESPAN_SESSION_ID,
      raw[SemanticConventions.SESSION_ID],
    );
    put(
      RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID,
      raw[SemanticConventions.USER_ID],
    );
  }
  // Aggregates and status/usage survive the native 128-attribute budget. Optional
  // indexed projections follow those canonical fields, without arbitrary caps.
  const attrs = span.attributes as Attributes;
  for (const [key, value] of Object.entries(canonical)) attrs[key] = value;
  for (const [prefix, values] of [
    [ATTR_GEN_AI_PROMPT, inputMessages],
    [ATTR_GEN_AI_COMPLETION, outputMessages],
  ] as const)
    values.forEach((message, index) => {
      for (const [key, value] of Object.entries(message))
        attrs[`${prefix}.${index}.${key}`] =
          typeof value === "object" ? json(value) : value;
    });
}
function vendorKey(key: string): boolean {
  return (
    key === SemanticConventions.OPENINFERENCE_SPAN_KIND ||
    key === SemanticConventions.INPUT_VALUE ||
    key === SemanticConventions.INPUT_MIME_TYPE ||
    key === SemanticConventions.OUTPUT_VALUE ||
    key === SemanticConventions.OUTPUT_MIME_TYPE ||
    key === SemanticConventions.METADATA ||
    key === SemanticConventions.SESSION_ID ||
    key === SemanticConventions.USER_ID ||
    key === SemanticConventions.TAG_TAGS ||
    key === SemanticConventions.AGENT_NAME ||
    key.startsWith("embedding.") ||
    key.startsWith("retrieval.") ||
    key.startsWith("reranker.") ||
    key.startsWith("tool.") ||
    key.startsWith(`${SemanticConventions.LLM_INPUT_MESSAGES}.`) ||
    key.startsWith(`${SemanticConventions.LLM_OUTPUT_MESSAGES}.`) ||
    key.startsWith("llm.token_count.") ||
    key.startsWith(`${SemanticConventions.LLM_TOOLS}.`) ||
    key === SemanticConventions.LLM_TOOLS ||
    key === SemanticConventions.LLM_INVOCATION_PARAMETERS ||
    key === SemanticConventions.LLM_MODEL_NAME ||
    key === SemanticConventions.LLM_REQUEST_MODEL_NAME ||
    key === SemanticConventions.LLM_RESPONSE_MODEL_NAME ||
    key === SemanticConventions.LLM_PROVIDER ||
    key === SemanticConventions.LLM_SYSTEM ||
    key.startsWith("llm.prompt_template")
  );
}
/** Export cleanup clones the native span; producer-readable attributes remain. */
export function prepareOpenInferenceSpanForExport(
  span: ReadableSpan,
): ReadableSpan {
  if (!isOpenInferenceSpan(span) && !captureBlocked(span)) return span;
  const attrs: Attributes = {};
  const lanes = captureLanes(span);
  for (const key of Object.keys(span.attributes)) {
    if (
      vendorKey(key) ||
      [
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "tools",
        "tool_calls",
      ].includes(key)
    )
      continue;
    const value = safeDescriptorValue(span.attributes, key);
    if (value !== undefined) attrs[key] = value;
  }
  const clone = Object.create(Object.getPrototypeOf(span));
  for (const key of Object.keys(span)) {
    if (["attributes", "status", "events"].includes(key)) continue;
    const descriptor = Object.getOwnPropertyDescriptor(span, key);
    if (descriptor) Object.defineProperty(clone, key, descriptor);
  }
  Object.defineProperties(clone, {
    attributes: {
      value: attrs,
      enumerable: true,
      writable: true,
      configurable: true,
    },
    status: {
      writable: true,
      configurable: true,
      value:
        lanes.inputs && lanes.outputs
          ? span.status
          : { code: span.status.code },
      enumerable: true,
    },
    events: {
      writable: true,
      configurable: true,
      value: lanes.inputs && lanes.outputs ? span.events : [],
      enumerable: true,
    },
  });
  if (captureBlocked(span)) {
    Object.defineProperty(clone, "attributes", {
      value: { [RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN]: true },
      enumerable: true,
      configurable: true,
      writable: true,
    });
  }
  guardExportClone(clone, span);
  return clone;
}
/** Use directly as an OTel SpanProcessor, or through the Respan plugin. */
export class OpenInferenceTranslator implements SpanProcessor {
  private readonly options: CaptureOptions;
  constructor(options: CaptureOptions = {}) {
    this.options = { ...options };
  }
  onStart(span: Span, parentContext: Context): void {
    guardSpan(span, parentContext, this.options);
  }
  onEnd(span: ReadableSpan): void {
    translateOpenInferenceSpan(span);
  }
  prepareForExport(span: ReadableSpan): ReadableSpan {
    return prepareOpenInferenceSpanForExport(span);
  }
  async shutdown(): Promise<void> {}
  async forceFlush(): Promise<void> {}
}
