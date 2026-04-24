/**
 * Translate Vercel AI SDK spans → Traceloop/OpenLLMetry format.
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
import { VERCEL_PARENT_SPANS, VERCEL_SPAN_CONFIG } from "./constants/index.js";
import { formatCompletionOutput, formatPromptInput, formatToolInput, formatToolOutput, parseToolChoice, parseToolsValue } from "./_translator/messages.js";
import {
  AI_AGENT_ID,
  AI_MODEL_ID,
  AI_PREFIX,
  GEN_AI_REQUEST_MODEL,
  LLM_REQUEST_TYPE,
  RESPAN_LOG_TYPE,
  RESPAN_METADATA_AGENT_NAME,
  RESPAN_SPAN_TOOLS,
  TL_ENTITY_INPUT,
  TL_ENTITY_OUTPUT,
  TL_REQUEST_FUNCTIONS,
  TL_SPAN_KIND,
  isVercelAISpan,
  metadataKey,
  normalizeModel,
  resolveLogType,
  safeJsonStr,
  setDefault,
} from "./_translator/shared.js";
import { enrichMetadata, enrichPerformanceMetrics, enrichTokens, stripRedundantAttrs } from "./_translator/span-enrichment.js";

/**
 * SpanProcessor that translates Vercel AI SDK attributes to Traceloop/OpenLLMetry.
 *
 * Phase 1 (onStart): Sets RESPAN_LOG_TYPE so CompositeProcessor lets the span through.
 * Phase 2 (onEnd):   Full attribute enrichment — model, messages, tokens, metadata,
 *                     tools, performance metrics, environment, etc.
 */
export class VercelAITranslator implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    const writableSpan = span as any;
    const name: string = writableSpan.name ?? "";
    if (!name.startsWith(AI_PREFIX)) {
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

    const name = span.name;
    const config = VERCEL_SPAN_CONFIG[name];
    const parentLogType = VERCEL_PARENT_SPANS[name];
    const logType = resolveLogType(name, attrs);

    enrichMetadata(attrs);

    if (parentLogType !== undefined && !config) {
      setDefault(attrs, RESPAN_LOG_TYPE, logType);
      stripRedundantAttrs(attrs);
      return;
    }

    attrs[RESPAN_LOG_TYPE] = logType;

    if (config) {
      setDefault(attrs, TL_SPAN_KIND, config.kind);

      if (config.isLLM) {
        setDefault(attrs, LLM_REQUEST_TYPE, RespanLogType.CHAT);

        const modelId = attrs[AI_MODEL_ID];
        if (modelId) {
          setDefault(attrs, GEN_AI_REQUEST_MODEL, normalizeModel(String(modelId)));
        }

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
          const tools = safeJsonStr(toolsValue);
          attrs[RESPAN_SPAN_TOOLS] = tools;
          attrs[TL_REQUEST_FUNCTIONS] = tools;
          attrs.tools = toolsValue;
        }

        const toolChoice = parseToolChoice(attrs);
        if (toolChoice) {
          setDefault(attrs, metadataKey("tool_choice"), toolChoice);
        }

        enrichPerformanceMetrics(attrs, name);
      }

      if (config.logType === RespanLogType.TOOL || logType === RespanLogType.TOOL) {
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
        setDefault(attrs, RESPAN_METADATA_AGENT_NAME, String(agentName));
      }
    } else {
      if (logType === RespanLogType.TEXT || logType === RespanLogType.EMBEDDING) {
        const modelId = attrs[AI_MODEL_ID];
        if (modelId) {
          setDefault(attrs, GEN_AI_REQUEST_MODEL, normalizeModel(String(modelId)));
        }

        enrichTokens(attrs);

        if (logType === RespanLogType.TEXT) {
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
      }

      if (logType === RespanLogType.TOOL) {
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
