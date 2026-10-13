import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_REQUEST_TOP_K,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  dataValue,
  protect,
  type Capture,
  type CaptureOptions,
} from "./_capture.js";
import {
  extractToolCalls,
  extractTools,
  extractUsage,
  normalizeInputMessages,
  requestPayloadFromCall,
  safeJson,
  toSerializableValue,
  type VertexAIRequestPayload,
} from "./_translator.js";
import {
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
} from "./_constants.js";
export interface VertexOperation {
  span: Span;
  ctx: Context;
  capture: Capture;
  payload?: VertexAIRequestPayload;
  embedding: boolean;
  generation: boolean;
  nativeInput?: unknown;
  ended: boolean;
}
export function startOperation(
  instance: any,
  args: any[],
  options: CaptureOptions,
  genai: boolean,
  embedding: boolean,
  chat: boolean,
  generation = false,
): VertexOperation {
  const parent = context.active();
  const request = args[0];
  const candidate = genai
    ? dataValue(request, "model")
    : (dataValue(instance, "model") ?? dataValue(instance, "resourcePath"));
  const model = typeof candidate === "string" ? candidate : undefined;
  const span = trace
    .getTracer(INSTRUMENTATION_LIBRARY_NAME, PACKAGE_VERSION)
    .startSpan(
      embedding
        ? "vertexai.embeddings"
        : generation
          ? "vertexai.generate"
          : VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
      {
        kind: SpanKind.CLIENT,
        attributes: {
          [RespanSpanAttributes.RESPAN_LOG_TYPE]: embedding
            ? RespanLogType.EMBEDDING
            : generation
              ? RespanLogType.GENERATION
              : RespanLogType.CHAT,
          [SpanAttributes.LLM_REQUEST_TYPE]: embedding
            ? "embedding"
            : generation
              ? "generation"
              : "chat",
          [SpanAttributes.TRACELOOP_ENTITY_NAME]: embedding
            ? "vertexai.embeddings"
            : VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
          [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
          [ATTR_GEN_AI_SYSTEM]: "google",
          ...(model ? { [ATTR_GEN_AI_REQUEST_MODEL]: model } : {}),
        },
      },
      parent,
    );
  const capture = protect(span, parent, options);
  const op: VertexOperation = {
    span,
    ctx: trace.setSpan(parent, span),
    capture,
    embedding,
    generation,
    ended: false,
  };
  if (capture.inputs)
    try {
      if (generation) {
        const nativeInput: Record<string, unknown> = {};
        for (const key of Object.keys(request ?? {})) {
          if (key === "config") continue;
          nativeInput[key] = toSerializableValue(request[key]);
        }
        if (request?.config) {
          const config: Record<string, unknown> = {};
          for (const key of Object.keys(request.config)) {
            if (["httpOptions", "abortSignal"].includes(key)) continue;
            config[key] = toSerializableValue(request.config[key]);
          }
          nativeInput.config = config;
        }
        op.nativeInput = nativeInput;
      }
      if (genai) {
        const config = dataValue(request, "config") ?? {};
        op.payload = {
          model,
          contents: toSerializableValue(dataValue(request, "contents")),
          generationConfig: {
            temperature: dataValue(config, "temperature"),
            maxOutputTokens: dataValue(config, "maxOutputTokens"),
            topP: dataValue(config, "topP"),
            topK: dataValue(config, "topK"),
          },
          systemInstruction: toSerializableValue(config.systemInstruction),
          tools: toSerializableValue(config.tools),
        };
      } else {
        op.payload = requestPayloadFromCall(
          instance,
          [toSerializableValue(args[0]), ...args.slice(1)],
          { isChatMethod: chat },
        );
        if (chat && Array.isArray(dataValue(instance, "historyInternal"))) {
          const history = toSerializableValue(
            dataValue(instance, "historyInternal"),
          );
          const next: any = op.payload.contents;
          op.payload.contents = [
            ...history,
            ...(typeof next === "string"
              ? [{ role: "user", parts: [{ text: next }] }]
              : Array.isArray(next)
                ? [{ role: "user", parts: next }]
                : (next?.contents ?? [next])),
          ];
        }
        op.payload = toSerializableValue(op.payload);
      }
    } catch {}
  return op;
}
function inputMessages(payload: VertexAIRequestPayload): any[] {
  const messages = normalizeInputMessages(
    payload.contents,
    payload.systemInstruction,
  );
  return messages;
}
function requestOptions(payload: VertexAIRequestPayload): Record<string, any> {
  const attrs: Record<string, any> = {};
  for (const [field, key] of [
    ["temperature", ATTR_GEN_AI_REQUEST_TEMPERATURE],
    ["maxOutputTokens", ATTR_GEN_AI_REQUEST_MAX_TOKENS],
    ["topP", ATTR_GEN_AI_REQUEST_TOP_P],
    ["topK", ATTR_GEN_AI_REQUEST_TOP_K],
  ]) {
    const value = dataValue(payload.generationConfig, field);
    if (typeof value === "number" && Number.isFinite(value)) attrs[key] = value;
  }
  return attrs;
}
function responseAttrs(
  original: any,
  outputs: boolean,
  embedding = false,
  generation = false,
): Record<string, any> {
  const attrs: Record<string, any> = {};
  let response = original;
  if (Array.isArray(original)) {
    const candidates = new Map<number, any>();
    let usageMetadata: any, modelVersion: any, responseId: any;
    for (const chunk of original) {
      usageMetadata = dataValue(chunk, "usageMetadata") ?? usageMetadata;
      modelVersion = dataValue(chunk, "modelVersion") ?? modelVersion;
      responseId = dataValue(chunk, "responseId") ?? responseId;
      if (outputs)
        for (const c of chunk.candidates ?? []) {
          const index = c.index ?? 0;
          let candidate = candidates.get(index);
          if (!candidate) {
            candidate = { index, content: { parts: [] } };
            candidates.set(index, candidate);
          }
          candidate.content.parts.push(
            ...toSerializableValue(c.content?.parts ?? []),
          );
          if (c.finishReason !== undefined)
            candidate.finishReason = c.finishReason;
        }
    }
    response = {
      usageMetadata,
      modelVersion,
      responseId,
      candidates: [...candidates.values()],
    };
  }
  const model = dataValue(response, "modelVersion"),
    id = dataValue(response, "responseId");
  if (typeof model === "string") attrs[ATTR_GEN_AI_RESPONSE_MODEL] = model;
  if (typeof id === "string") attrs[ATTR_GEN_AI_RESPONSE_ID] = id;
  const nativeUsage = dataValue(response, "usageMetadata");
  for (const [field, key] of [
    ["promptTokenCount", ATTR_GEN_AI_USAGE_INPUT_TOKENS],
    ["promptTokenCount", ATTR_GEN_AI_USAGE_PROMPT_TOKENS],
    ["candidatesTokenCount", ATTR_GEN_AI_USAGE_OUTPUT_TOKENS],
    ["candidatesTokenCount", ATTR_GEN_AI_USAGE_COMPLETION_TOKENS],
    ["totalTokenCount", SpanAttributes.LLM_USAGE_TOTAL_TOKENS],
    ["thoughtsTokenCount", ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS],
    ["cachedContentTokenCount", ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS],
  ]) {
    const value = dataValue(nativeUsage, field);
    if (typeof value === "number" && Number.isFinite(value) && value >= 0)
      attrs[key] = value;
  }
  if (embedding) {
    const embeddings = dataValue(response, "embeddings");
    if (Array.isArray(embeddings) && embeddings.length) {
      let tokens = 0,
        complete = true;
      for (let i = 0; i < embeddings.length; i++) {
        const count = dataValue(
          dataValue(dataValue(embeddings, String(i)), "statistics"),
          "tokenCount",
        );
        if (typeof count !== "number" || !Number.isFinite(count) || count < 0) {
          complete = false;
          break;
        }
        tokens += count;
      }
      if (complete) {
        attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = tokens;
        attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = tokens;
      }
    }
  }
  if (outputs) {
    if (generation)
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
        toSerializableValue(response),
      );
    else if (embedding)
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
        toSerializableValue(response?.embeddings ?? response),
      );
    else {
      const candidates = Array.isArray(response?.candidates)
        ? response.candidates
        : [];
      const messages = candidates.map((candidate: any) => ({
        role: "assistant",
        content: toSerializableValue(candidate.content?.parts ?? []),
        ...(candidate.finishReason !== undefined
          ? { finish_reason: candidate.finishReason }
          : {}),
        ...(candidate.index !== undefined ? { index: candidate.index } : {}),
      }));
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(messages);
      if (messages.length) {
        attrs[`${ATTR_GEN_AI_COMPLETION}.0.role`] = "assistant";
        attrs[`${ATTR_GEN_AI_COMPLETION}.0.content`] = safeJson(
          messages[0].content,
        );
      }
      const calls = extractToolCalls(response);
      if (calls.length)
        attrs[`${ATTR_GEN_AI_COMPLETION}.0.tool_calls`] = safeJson(calls);
    }
  }
  return attrs;
}
export function buildGenerateContentAttrs(opts: {
  requestPayload: VertexAIRequestPayload;
  responseOrChunks?: unknown;
  spanName?: string;
}): Record<string, any> {
  const attrs: Record<string, any> = {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [SpanAttributes.LLM_REQUEST_TYPE]: "chat",
    [ATTR_GEN_AI_SYSTEM]: "google",
    [SpanAttributes.TRACELOOP_ENTITY_NAME]:
      opts.spanName ?? VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
  };
  const p = opts.requestPayload;
  if (p.model) attrs[ATTR_GEN_AI_REQUEST_MODEL] = p.model;
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMessages(p));
  const tools = extractTools(p.tools);
  if (tools.length)
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
  Object.assign(attrs, requestOptions(p));
  if (opts.responseOrChunks !== undefined)
    Object.assign(attrs, responseAttrs(opts.responseOrChunks, true));
  return attrs;
}
export function finishOperation(
  op: VertexOperation,
  response?: any,
  error?: unknown,
  failed = false,
): void {
  if (op.ended) return;
  op.ended = true;
  op.capture.observe();
  try {
    const attrs: Record<string, any> = {};
    if (op.capture.inputs && op.generation)
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(op.nativeInput);
    else if (op.capture.inputs && op.payload) {
      if (op.embedding)
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
          op.payload.contents,
        );
      else {
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
          inputMessages(op.payload),
        );
        const tools = extractTools(op.payload.tools);
        if (tools.length)
          attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
      }
    }
    if (
      !failed &&
      response !== undefined &&
      op.span.isRecording() &&
      op.span.spanContext().traceFlags & 1
    )
      Object.assign(
        attrs,
        responseAttrs(
          response,
          op.capture.outputs,
          op.embedding,
          op.generation,
        ),
      );
    if (op.capture.inputs && op.payload)
      Object.assign(attrs, requestOptions(op.payload));
    op.span.setAttributes(attrs);
    if (op.capture.inputs && op.payload && !op.embedding && !op.generation)
      inputMessages(op.payload).forEach((message, index) => {
        op.span.setAttribute(
          `${ATTR_GEN_AI_PROMPT}.${index}.role`,
          message.role,
        );
        op.span.setAttribute(
          `${ATTR_GEN_AI_PROMPT}.${index}.content`,
          typeof message.content === "string"
            ? message.content
            : safeJson(message.content),
        );
        if (message.tool_calls?.length)
          op.span.setAttribute(
            `${ATTR_GEN_AI_PROMPT}.${index}.tool_calls`,
            safeJson(message.tool_calls),
          );
      });
    if (failed) {
      op.span.setStatus({
        code: SpanStatusCode.ERROR,
        ...(op.capture.inputs && op.capture.outputs
          ? { message: error instanceof Error ? error.message : String(error) }
          : {}),
      });
      if (op.capture.inputs && op.capture.outputs)
        op.span.recordException(
          error instanceof Error ? error : new Error(String(error)),
        );
      const status =
        dataValue(error, "status") ?? dataValue(error, "statusCode");
      if (typeof status === "number" && Number.isFinite(status))
        op.span.setAttribute("status_code", status);
    }
  } catch {
  } finally {
    op.span.end();
    op.payload = undefined;
    op.nativeInput = undefined;
  }
}
