/**
 * Translate Vercel AI SDK spans -> Traceloop/OpenLLMetry format.
 *
 * The Vercel AI SDK emits OTEL spans with its own attribute schema (ai.model.id,
 * ai.prompt.messages, ai.response.text, etc.). This SpanProcessor enriches those
 * spans with the Traceloop/GenAI semantic conventions the Respan backend expects.
 *
 * Two-phase enrichment:
 * - onStart(): Sets RESPAN_LOG_TYPE so the span passes CompositeProcessor filtering
 * - onEnd():   Full attribute translation (model, messages, tokens, metadata, etc.)
 */

import type { Context } from "@opentelemetry/api";
import type { ReadableSpan, Span, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { RespanLogType } from "@respan/respan-sdk";
import { VERCEL_PARENT_SPANS, VERCEL_SPAN_CONFIG, VERCEL_STRUCTURAL_LLM_PARENT_SPANS } from "./constants/index.js";
import {
  formatCompletionOutput,
  formatPromptInput,
  formatToolInput,
  formatToolOutput,
  parseToolChoice,
  parseToolsValue,
} from "./_translator/messages.js";
import {
  AI_AGENT_ID,
  AI_MODEL_ID,
  AI_PREFIX,
  AI_TOOL_CALL_NAME,
  GEN_AI_REQUEST_MODEL,
  LLM_REQUEST_TYPE,
  RESPAN_INTERNAL_DROP_SPAN,
  RESPAN_INTERNAL_SPAN_NAME_DETAIL,
  RESPAN_INTERNAL_SPAN_NAME_KIND,
  RESPAN_LOG_TYPE,
  RESPAN_METADATA_AGENT_NAME,
  TL_SPAN_KIND,
  TL_ENTITY_INPUT,
  TL_ENTITY_OUTPUT,
  TL_REQUEST_FUNCTIONS,
  formatEmbeddingInput,
  formatEmbeddingOutput,
  isVercelAISpan,
  metadataKey,
  resolveLogType,
  safeJsonStr,
  setDefault,
} from "./_translator/shared.js";
import { enrichMetadata, enrichModel, enrichPerformanceMetrics, enrichTokens, stripRedundantAttrs } from "./_translator/span-enrichment.js";

function setSemanticLlmSpanNameHint(attrs: Record<string, any>): void {
  setDefault(attrs, RESPAN_INTERNAL_SPAN_NAME_KIND, "llm");
  setDefault(attrs, RESPAN_INTERNAL_SPAN_NAME_DETAIL, attrs[GEN_AI_REQUEST_MODEL]);
}

function setSemanticEmbeddingSpanNameHint(attrs: Record<string, any>): void {
  setDefault(attrs, RESPAN_INTERNAL_SPAN_NAME_KIND, "embedding");
  delete attrs[RESPAN_INTERNAL_SPAN_NAME_DETAIL];
}

function setSemanticNamedSpanNameHint(
  attrs: Record<string, any>,
  kind: string,
  detail: unknown
): void {
  setDefault(attrs, RESPAN_INTERNAL_SPAN_NAME_KIND, kind);
  setDefault(attrs, RESPAN_INTERNAL_SPAN_NAME_DETAIL, detail);
}

/**
 * SpanProcessor that translates Vercel AI SDK attributes to Traceloop/OpenLLMetry.
 *
 * Phase 1 (onStart): Sets RESPAN_LOG_TYPE so CompositeProcessor lets the span through.
 * Phase 2 (onEnd):   Full attribute enrichment -- model, messages, tokens, metadata,
 *                     tools, performance metrics, environment, etc.
 */
export class VercelAITranslator implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    const writableSpan = span as any;
    const name: string = writableSpan.name ?? "";
    if (!name.startsWith(AI_PREFIX)) {
      return;
    }

    if (VERCEL_STRUCTURAL_LLM_PARENT_SPANS.has(name)) {
      writableSpan.setAttribute(RESPAN_INTERNAL_DROP_SPAN, true);
      return;
    }

    const config = VERCEL_SPAN_CONFIG[name];
    if (config) {
      writableSpan.setAttribute(RESPAN_LOG_TYPE, config.logType);
      return;
    }

    const parentLogType = VERCEL_PARENT_SPANS[name];
    if (parentLogType !== undefined) {
      writableSpan.setAttribute(RESPAN_LOG_TYPE, parentLogType);
      return;
    }

    writableSpan.setAttribute(RESPAN_LOG_TYPE, RespanLogType.TASK);
  }

  onEnd(span: ReadableSpan): void {
    const attrs = (span as any).attributes as Record<string, any> | undefined;
    if (!attrs || !isVercelAISpan(span)) {
      return;
    }

    if (attrs[RESPAN_INTERNAL_DROP_SPAN] === true || attrs[RESPAN_INTERNAL_DROP_SPAN] === "true") {
      stripRedundantAttrs(attrs);
      return;
    }

    const name = span.name;
    const config = VERCEL_SPAN_CONFIG[name];
    const parentLogType = VERCEL_PARENT_SPANS[name];
    const logType = resolveLogType(name, attrs);

    // Embedding spans (span-contract.md): input = embedded text, output = the
    // embedding vector(s). Vercel's synthetic ai.usage.tokens is intentionally
    // NOT surfaced as a token count; it is stripped after promotion.
    if (
      logType === RespanLogType.EMBEDDING ||
      config?.logType === RespanLogType.EMBEDDING ||
      parentLogType === RespanLogType.EMBEDDING
    ) {
      setSemanticEmbeddingSpanNameHint(attrs);
      const embInput = formatEmbeddingInput(attrs);
      if (embInput) setDefault(attrs, TL_ENTITY_INPUT, embInput);
      const embOutput = formatEmbeddingOutput(attrs);
      if (embOutput) setDefault(attrs, TL_ENTITY_OUTPUT, embOutput);
    }

    enrichMetadata(attrs);
    delete attrs[TL_SPAN_KIND];

    if (parentLogType !== undefined && !config) {
      setDefault(attrs, RESPAN_LOG_TYPE, logType);
      stripRedundantAttrs(attrs);
      return;
    }

    attrs[RESPAN_LOG_TYPE] = logType;

    if (config) {
      // Do NOT set traceloop.span.kind for auto-emitted Vercel SDK spans.
      // In the Respan composite processor `traceloop.span.kind` is reserved
      // for user-decorated spans (withWorkflow / withTask / withAgent) and
      // setting it on auto spans flattens the tree and misclassifies LLM spans.

      if (config.isLLM) {
        setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.CHAT);

        enrichModel(attrs, attrs[AI_MODEL_ID]);
        setSemanticLlmSpanNameHint(attrs);

        const input = formatPromptInput(attrs);
        if (input) {
          setDefault(attrs, TL_ENTITY_INPUT, input);
        }

        const output = formatCompletionOutput(attrs);
        if (output) {
          setDefault(attrs, TL_ENTITY_OUTPUT, output);
        }

        enrichTokens(attrs);

        const toolsValue = parseToolsValue(attrs);
        if (toolsValue) {
          attrs[TL_REQUEST_FUNCTIONS] = safeJsonStr(toolsValue);
        }

        const toolChoice = parseToolChoice(attrs);
        if (toolChoice) {
          setDefault(attrs, metadataKey("tool_choice"), toolChoice);
        }

        enrichPerformanceMetrics(attrs, name);
      }

      if (config.logType === RespanLogType.EMBEDDING || logType === RespanLogType.EMBEDDING) {
        // input/output/tokens are mapped in the up-front embedding block.
        setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.EMBEDDING);
        enrichModel(attrs, attrs[AI_MODEL_ID]);
      }

      if (config.logType === RespanLogType.TOOL || logType === RespanLogType.TOOL) {
        setSemanticNamedSpanNameHint(attrs, "tool", attrs[AI_TOOL_CALL_NAME] ?? name.split(".").at(-1));

        const toolInput = formatToolInput(attrs);
        if (toolInput) {
          setDefault(attrs, TL_ENTITY_INPUT, toolInput);
        }

        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) {
          setDefault(attrs, TL_ENTITY_OUTPUT, toolOutput);
        }
      }

      if (config.logType === RespanLogType.AGENT || logType === RespanLogType.AGENT) {
        const agentName = attrs["ai.agent.name"] ?? attrs[AI_AGENT_ID] ?? name;
        setSemanticNamedSpanNameHint(attrs, "agent", agentName);
        setDefault(attrs, RESPAN_METADATA_AGENT_NAME, String(agentName));
      }
    } else {
      if (logType === RespanLogType.TEXT) {
        enrichModel(attrs, attrs[AI_MODEL_ID]);
        setSemanticLlmSpanNameHint(attrs);

        enrichTokens(attrs);

        setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.CHAT);

        const input = formatPromptInput(attrs);
        if (input) {
          setDefault(attrs, TL_ENTITY_INPUT, input);
        }

        const output = formatCompletionOutput(attrs);
        if (output) {
          setDefault(attrs, TL_ENTITY_OUTPUT, output);
        }

        enrichPerformanceMetrics(attrs, name);
      }

      if (logType === RespanLogType.EMBEDDING) {
        // input/output/tokens are mapped in the up-front embedding block.
        setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.EMBEDDING);
        enrichModel(attrs, attrs[AI_MODEL_ID]);
      }

      if (logType === RespanLogType.TOOL) {
        setSemanticNamedSpanNameHint(attrs, "tool", attrs[AI_TOOL_CALL_NAME] ?? name.split(".").at(-1));

        const toolInput = formatToolInput(attrs);
        if (toolInput) {
          setDefault(attrs, TL_ENTITY_INPUT, toolInput);
        }

        const toolOutput = formatToolOutput(attrs);
        if (toolOutput) {
          setDefault(attrs, TL_ENTITY_OUTPUT, toolOutput);
        }
      }
    }

    stripRedundantAttrs(attrs);
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }
}
