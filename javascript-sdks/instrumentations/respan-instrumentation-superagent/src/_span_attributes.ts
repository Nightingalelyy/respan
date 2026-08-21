import {
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  GUARD_METHOD,
  SUPERAGENT_INSTRUMENTATION_NAME,
  SUPERAGENT_METADATA_CLASSIFICATION,
  SUPERAGENT_METADATA_INTEGRATION,
  SUPERAGENT_METADATA_METHOD,
  SUPERAGENT_METADATA_MODEL,
  SUPERAGENT_METADATA_REDACT_FINDINGS,
} from "./_constants.js";
import {
  extractModel,
  extractPrimaryInput,
  getAttr,
  normalizeCallInput,
  safeJsonStringify,
} from "./_serialization.js";

export type SuperagentSpanAttributeValue = string | number | boolean | string[];

export type SuperagentSpanAttributes = Record<
  string,
  SuperagentSpanAttributeValue
>;

export interface BuildSuperagentSpanAttributesOptions {
  methodName: string;
  args: unknown[];
  result?: unknown;
  error?: unknown;
  workflowName?: string;
}

export type BuildSuperagentModelSpanAttributesOptions =
  BuildSuperagentSpanAttributesOptions;

function operationLogType(methodName: string): RespanLogType {
  return methodName === GUARD_METHOD ? RespanLogType.GUARDRAIL : RespanLogType.TOOL;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function addResultMetadata(
  attrs: SuperagentSpanAttributes,
  methodName: string,
  result: unknown,
): void {
  if (methodName === GUARD_METHOD) {
    const classification = getAttr(result, "classification");
    if (typeof classification === "string" && classification.length > 0) {
      attrs[SUPERAGENT_METADATA_CLASSIFICATION] = classification;
      attrs[RespanSpanAttributes.RESPAN_METADATA_TRIGGERED] =
        classification === "block";
    }

    attrs[RespanSpanAttributes.RESPAN_METADATA_GUARDRAIL_NAME] =
      "superagent.guard";
    return;
  }

  if (methodName === "redact") {
    const findings = getAttr(result, "findings");
    if (findings !== undefined && findings !== null) {
      attrs[SUPERAGENT_METADATA_REDACT_FINDINGS] = safeJsonStringify(findings);
    }
  }
}

function tokenCount(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return undefined;
  }
  return Math.trunc(value);
}

function addUsageAttributes(
  attrs: SuperagentSpanAttributes,
  result: unknown,
): void {
  const usage = getAttr(result, "usage");
  const inputTokens = tokenCount(
    getAttr(usage, "promptTokens") ?? getAttr(usage, "inputTokens"),
  );
  const outputTokens = tokenCount(
    getAttr(usage, "completionTokens") ?? getAttr(usage, "outputTokens"),
  );
  const totalTokens =
    tokenCount(getAttr(usage, "totalTokens")) ??
    (inputTokens !== undefined || outputTokens !== undefined
      ? (inputTokens ?? 0) + (outputTokens ?? 0)
      : undefined);

  if (inputTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = inputTokens;
  }
  if (outputTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = outputTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = outputTokens;
  }
  if (totalTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
}

function modelProvider(model: string): string {
  const provider = model.split("/", 1)[0]?.toLowerCase();
  if (provider === "openai-compatible") {
    return "openai";
  }
  return provider || "superagent";
}

function messageContent(value: unknown): string {
  return typeof value === "string" ? value : safeJsonStringify(value);
}

export function buildSuperagentModelSpanAttributes({
  methodName,
  args,
  result,
  error,
  workflowName,
}: BuildSuperagentModelSpanAttributesOptions): SuperagentSpanAttributes {
  const operationName = `superagent.${methodName}.model`;
  const input =
    extractPrimaryInput(methodName, args) ?? normalizeCallInput(methodName, args);
  const inputContent = messageContent(input);
  const attrs: SuperagentSpanAttributes = {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: operationName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: operationName,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJsonStringify([
      { role: "user", content: input },
    ]),
    [SpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
    "gen_ai.prompt.0.role": "user",
    "gen_ai.prompt.0.content": inputContent,
  };

  if (workflowName) {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
  }

  const model = extractModel(args);
  if (model) {
    attrs[ATTR_GEN_AI_SYSTEM] = modelProvider(model);
    attrs[ATTR_GEN_AI_REQUEST_MODEL] = model;
  }

  if (error !== undefined) {
    const output = { error: errorMessage(error) };
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStringify([
      { role: "assistant", content: output },
    ]);
    attrs["gen_ai.completion.0.role"] = "assistant";
    attrs["gen_ai.completion.0.content"] = safeJsonStringify(output);
    return attrs;
  }

  if (result !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStringify([
      { role: "assistant", content: result },
    ]);
    attrs["gen_ai.completion.0.role"] = "assistant";
    attrs["gen_ai.completion.0.content"] = messageContent(result);
    addUsageAttributes(attrs, result);
  }

  return attrs;
}

export function buildSuperagentSpanAttributes({
  methodName,
  args,
  result,
  error,
  workflowName,
}: BuildSuperagentSpanAttributesOptions): SuperagentSpanAttributes {
  const operationName = `superagent.${methodName}`;
  const attrs: SuperagentSpanAttributes = {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: operationLogType(methodName),
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: operationName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: operationName,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJsonStringify(
      normalizeCallInput(methodName, args),
    ),
    [SUPERAGENT_METADATA_INTEGRATION]: SUPERAGENT_INSTRUMENTATION_NAME,
    [SUPERAGENT_METADATA_METHOD]: methodName,
  };

  if (workflowName) {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
  }

  const model = extractModel(args);
  if (model) {
    attrs[SUPERAGENT_METADATA_MODEL] = model;
  }

  if (error !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStringify({
      error: errorMessage(error),
    });
    return attrs;
  }

  if (result !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStringify(result);
    addResultMetadata(attrs, methodName, result);
  }

  return attrs;
}
