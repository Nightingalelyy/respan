import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  RESPAN_SPAN_ATTRIBUTES_MAP,
  RespanSpanAttributes,
} from "@respan/respan-sdk";
import { data, json, primitive, snapshot, string, type Row } from "./_data.js";

export function logType(row: Row): string {
  const attrs = data(row, "span_attributes");
  const type = string(data(attrs, "type"));
  if (type === "llm" && data(attrs, "name") === "Embedding") return "embedding";
  if (type === "llm") return "chat";
  if (type === "function") return "tool";
  if (["workflow", "agent", "task", "tool", "embedding"].includes(type ?? ""))
    return type!;
  if (["score", "classifier", "review"].includes(type ?? ""))
    return "guardrail";
  return "task";
}

export function safeAttributes(row: Row): Row {
  const attrs = data(row, "span_attributes");
  const metadata = data(row, "metadata");
  const identity: Row = {};
  for (const key of ["id", "span_id", "root_span_id"]) {
    const value = data(row, key);
    if (primitive(value))
      identity[`braintrust_${key === "id" ? "row_id" : key}`] = value;
  }
  const name = string(data(attrs, "name")) ?? "braintrust.span";
  const result: Row = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType(row),
    [RespanSpanAttributes.RESPAN_METADATA]: json(identity),
  };
  if (["chat", "embedding"].includes(logType(row))) {
    const model =
      string(data(attrs, "model")) ?? string(data(metadata, "model"));
    const provider =
      string(data(attrs, "provider")) ?? string(data(metadata, "provider"));
    if (model !== undefined) result[SpanAttributes.LLM_REQUEST_MODEL] = model;
    if (provider !== undefined) result[SpanAttributes.LLM_SYSTEM] = provider;
    result[SpanAttributes.LLM_REQUEST_TYPE] = logType(row);
  }
  return result;
}

export function usage(row: Row): Row {
  const metrics = data(row, "metrics");
  const result: Row = {};
  const number = (...keys: string[]): number | undefined => {
    for (const key of keys) {
      const value = data(metrics, key);
      if (typeof value === "number" && Number.isInteger(value) && value >= 0)
        return value;
    }
    return undefined;
  };
  const input = number("prompt_tokens", "input_tokens");
  const output = number("completion_tokens", "output_tokens");
  const total = number("tokens", "total_tokens");
  const cache = number("cache_read_input_tokens");
  if (input !== undefined) {
    result[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input;
    result[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = input;
  }
  if (output !== undefined) {
    result[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output;
    result[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = output;
  }
  if (total !== undefined)
    result[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total;
  if (cache !== undefined)
    result[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache;
  return result;
}

/** Applied only after admission. Canonical fields precede optional indexed fields. */
export function attributes(
  row: Row,
  propagated: Row,
  includeRaw: boolean,
): Row {
  const result = safeAttributes(row);
  const type = logType(row);
  const llm = type === "chat" || type === "embedding";
  const metadata: Row = {
    ...snapshot(data(propagated, "metadata")),
    ...snapshot(row.metadata),
    ...JSON.parse(result[RespanSpanAttributes.RESPAN_METADATA]),
  };
  const spanType = data(row.span_attributes, "type");
  if (spanType !== undefined) metadata.braintrust_span_type = spanType;
  for (const key of ["tags", "scores", "expected", "context", "metrics"]) {
    if (row[key] !== undefined) metadata[`braintrust_${key}`] = row[key];
  }
  if (includeRaw && row.span_attributes !== undefined)
    metadata.braintrust_span_attributes = row.span_attributes;
  result[RespanSpanAttributes.RESPAN_METADATA] = json(metadata);
  for (const key of Object.keys(RESPAN_SPAN_ATTRIBUTES_MAP)) {
    if (key === "metadata") continue;
    const value = data(propagated, key);
    if (primitive(value)) result[RESPAN_SPAN_ATTRIBUTES_MAP[key]] = value;
  }
  const workflow =
    string(data(data(propagated, "metadata"), "workflow_name")) ??
    string(data(propagated, "trace_group_identifier"));
  if (workflow !== undefined)
    result[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow;
  if (row.input !== undefined) {
    result[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json(
      type === "tool"
        ? {
            name: result[SpanAttributes.TRACELOOP_ENTITY_NAME],
            arguments:
              data(row.input, "arguments") !== undefined
                ? row.input.arguments
                : row.input,
          }
        : row.input,
    );
  }
  if (row.output !== undefined)
    result[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json(row.output);
  if (llm) Object.assign(result, usage(row));
  if (type === "tool") {
    const callId =
      string(data(row.span_attributes, "tool_call_id")) ??
      string(data(row.metadata, "tool_call_id")) ??
      string(data(row.input, "tool_call_id"));
    if (callId !== undefined) result[ATTR_GEN_AI_TOOL_CALL_ID] = callId;
  }
  if (type !== "chat") return result;
  const toolDefinitions =
    row.span_attributes?.tools ??
    row.metadata?.tools ??
    data(row.input, "tools") ??
    row.span_attributes?.functions ??
    row.metadata?.functions;
  if (toolDefinitions !== undefined)
    result[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json(toolDefinitions);
  const output = messages(row.output, true);
  output.forEach((message, index) => {
    if (message.role !== undefined)
      result[`${SpanAttributes.LLM_COMPLETIONS}.${index}.role`] = message.role;
    if (message.content !== undefined)
      result[`${SpanAttributes.LLM_COMPLETIONS}.${index}.content`] =
        typeof message.content === "string"
          ? message.content
          : json(message.content);
    if (message.tool_calls !== undefined)
      result[`${SpanAttributes.LLM_COMPLETIONS}.${index}.tool_calls`] = json(
        message.tool_calls,
      );
  });
  // Indexed prompts are optional: the complete canonical JSON stays within the default 128-attribute budget.
  messages(row.input).forEach((message, index) => {
    if (message.role !== undefined)
      result[`${SpanAttributes.LLM_PROMPTS}.${index}.role`] = message.role;
    if (message.content !== undefined)
      result[`${SpanAttributes.LLM_PROMPTS}.${index}.content`] =
        typeof message.content === "string"
          ? message.content
          : json(message.content);
    if (message.tool_calls !== undefined)
      result[`${SpanAttributes.LLM_PROMPTS}.${index}.tool_calls`] = json(
        message.tool_calls,
      );
  });
  return result;
}

function messages(value: any, output = false): Row[] {
  if (typeof value === "string")
    return [{ role: output ? "assistant" : "user", content: value }];
  if (!value || typeof value !== "object") return [];
  if (Array.isArray(value))
    return value.flatMap((item) => messages(item, output));
  if (typeof value.role === "string") return [value];
  if (value.message !== undefined) return messages(value.message, output);
  if (value.messages !== undefined) return messages(value.messages, output);
  if (value.choices !== undefined) return messages(value.choices, output);
  if (output && (value.content !== undefined || value.tool_calls !== undefined))
    return [{ ...value, role: "assistant" }];
  return [];
}

export function recordSnapshot(row: Row): Row {
  const result: Row = {};
  for (const key of [
    "id",
    "span_id",
    "root_span_id",
    "span_parents",
    "created",
    "_parent_id",
    "_is_merge",
    "_merge_paths",
    "span_attributes",
    "input",
    "output",
    "metadata",
    "metrics",
    "error",
    "tags",
    "scores",
    "expected",
    "context",
  ]) {
    const value = snapshot(data(row, key));
    if (value !== undefined) result[key] = value;
  }
  return result;
}
