import { Context } from "@opentelemetry/api";
import {
  SpanProcessor,
  ReadableSpan,
} from "@opentelemetry/sdk-trace-base";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  RESPAN_SPAN_ATTRIBUTES_MAP,
  RespanSpanAttributes,
} from "@respan/respan-sdk";
import { MultiProcessorManager } from "./manager.js";
import { getEntityPath, getPropagatedAttributes } from "../utils/context.js";

// ── OpenInference span enrichment ──────────────────────────────────────────

/** Map OI span kinds → Traceloop span kinds */
const OI_KIND_TO_TRACELOOP: Record<string, string> = {
  LLM: "task",
  CHAIN: "workflow",
  TOOL: "tool",
  AGENT: "agent",
  EMBEDDING: "task",
  RETRIEVER: "task",
  RERANKER: "task",
  GUARDRAIL: "task",
  EVALUATOR: "task",
};

/** OI span kinds that imply an LLM request type */
const OI_LLM_REQUEST_KINDS: Record<string, string> = {
  LLM: "chat",
  EMBEDDING: "embedding",
};

/** Map OI span kinds → respan.entity.log_type values */
const OI_LOG_TYPE: Record<string, string> = {
  LLM: "chat",
  CHAIN: "workflow",
  TOOL: "tool",
  AGENT: "agent",
  EMBEDDING: "embedding",
  RETRIEVER: "task",
  RERANKER: "task",
  GUARDRAIL: "guardrail",
  EVALUATOR: "task",
};

/**
 * Build Traceloop/GenAI enrichment attributes for an OpenInference span.
 * Returns only the attributes that need to be *added*; callers merge them
 * on top of the original span attributes.
 */
function getOIEnrichmentAttrs(span: ReadableSpan): Record<string, any> {
  const attrs: Record<string, any> = {};
  const oiKind = String(span.attributes["openinference.span.kind"] ?? "");

  if (OI_KIND_TO_TRACELOOP[oiKind]) {
    attrs[SpanAttributes.TRACELOOP_SPAN_KIND] = OI_KIND_TO_TRACELOOP[oiKind];
  }
  if (OI_LLM_REQUEST_KINDS[oiKind]) {
    attrs["llm.request.type"] = OI_LLM_REQUEST_KINDS[oiKind];
  }
  if (OI_LOG_TYPE[oiKind]) {
    attrs["respan.entity.log_type"] = OI_LOG_TYPE[oiKind];
  }

  // Bridge OI semantic attrs → Traceloop/GenAI equivalents
  if (span.attributes["input.value"] !== undefined)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = span.attributes["input.value"];
  if (span.attributes["output.value"] !== undefined)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = span.attributes["output.value"];
  if (span.attributes["llm.model_name"] !== undefined)
    attrs["gen_ai.request.model"] = span.attributes["llm.model_name"];
  if (span.attributes["llm.token_count.prompt"] !== undefined)
    attrs["gen_ai.usage.prompt_tokens"] = span.attributes["llm.token_count.prompt"];
  if (span.attributes["llm.token_count.completion"] !== undefined)
    attrs["gen_ai.usage.completion_tokens"] = span.attributes["llm.token_count.completion"];

  // Entity name / path
  attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = span.name;
  if (OI_KIND_TO_TRACELOOP[oiKind] !== "workflow") {
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = span.name;
  }

  return attrs;
}

// ── Composite processor ────────────────────────────────────────────────────

/**
 * Composite processor that combines filtering with multi-processor routing.
 *
 * Flow:
 * 1. Filter spans (keep only user-decorated spans and their children)
 * 2. Apply postprocess callback if configured
 * 3. Route filtered spans to appropriate processors
 *
 * This ensures only meaningful spans are routed to processors.
 */
export class RespanCompositeProcessor implements SpanProcessor {
  private readonly _processorManager: MultiProcessorManager;
  private readonly _postprocessCallback?: (span: ReadableSpan) => void;

  constructor(
    processorManager: MultiProcessorManager,
    postprocessCallback?: (span: ReadableSpan) => void
  ) {
    this._processorManager = processorManager;
    this._postprocessCallback = postprocessCallback;
  }

  onStart(span: ReadableSpan, parentContext: Context): void {
    // Check if this span is being created within an entity context
    // If so, add the entityPath attribute so it gets preserved by our filtering
    const entityPath = getEntityPath(parentContext);
    if (entityPath && !span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]) {
      // This is an auto-instrumentation span within an entity context
      // Add the entityPath attribute so it doesn't get filtered out
      console.debug(
        `[Respan Debug] Adding entityPath to auto-instrumentation span: ${span.name} (entityPath: ${entityPath})`
      );

      // We need to cast to any to set attributes during onStart
      (span as any).setAttribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entityPath);
    }

    // Apply propagated attributes (customer_identifier, thread_identifier, etc.)
    const propagated = getPropagatedAttributes(parentContext);
    if (propagated) {
      for (const [key, value] of Object.entries(propagated)) {
        if (value === undefined || value === null) continue;
        const attrKey = RESPAN_SPAN_ATTRIBUTES_MAP[key];
        if (!attrKey) continue;

        if (key === "metadata" && typeof value === "object") {
          for (const [mk, mv] of Object.entries(value as Record<string, any>)) {
            (span as any).setAttribute(
              `${RespanSpanAttributes.RESPAN_METADATA}.${mk}`,
              typeof mv === "string" ? mv : JSON.stringify(mv)
            );
          }
        } else if (key === "prompt" && typeof value === "object") {
          (span as any).setAttribute(attrKey, JSON.stringify(value));
        } else {
          (span as any).setAttribute(attrKey, value as any);
        }
      }
    }

    // Forward to processor manager
    this._processorManager.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    const spanKind = span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND];
    const entityPath = span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH];

    // Apply postprocess callback if provided
    if (this._postprocessCallback) {
      try {
        this._postprocessCallback(span);
      } catch (error) {
        console.error("[Respan] Error in span postprocess callback:", error);
      }
    }

    // Check if this is an LLM instrumentation span (OpenAI, Anthropic, etc.)
    // These have gen_ai.* or llm.* attributes
    const isLLMSpan =
      span.attributes['gen_ai.system'] !== undefined ||
      span.attributes['llm.system'] !== undefined ||
      span.attributes['gen_ai.request.model'] !== undefined ||
      span.name.includes('anthropic.messages') ||
      span.name.includes('openai.chat') ||
      span.name.includes('chat.completions');

    // Filter: only process spans that are user-decorated, within entity context, or LLM calls
    if (spanKind) {
      // This is a user-decorated span (withWorkflow, withTask, etc.) - make it a root span
      console.debug(
        `[Respan Debug] Processing user-decorated span as root: ${span.name} (kind: ${spanKind})`
      );

      // Create a wrapper that makes the span appear as a root span
      const rootSpan = Object.create(Object.getPrototypeOf(span));
      Object.assign(rootSpan, span);

      // Override the parentSpanId to make it a root span
      Object.defineProperty(rootSpan, 'parentSpanId', {
        value: undefined,
        writable: false,
        configurable: true,
        enumerable: true
      });

      // Route to processors
      this._processorManager.onEnd(rootSpan);
    } else if (entityPath && entityPath !== "") {
      // This span doesn't have a kind but has entityPath - it's a child span within a withEntity context
      // Keep it as a normal child span (preserve parent relationships)
      console.debug(
        `[Respan Debug] Processing child span within entity context: ${span.name} (entityPath: ${entityPath})`
      );

      // Route to processors
      this._processorManager.onEnd(span);
    } else if (isLLMSpan) {
      // This is an LLM instrumentation span - keep it!
      console.debug(
        `[Respan Debug] Processing LLM instrumentation span: ${span.name}`
      );

      // Route to processors
      this._processorManager.onEnd(span);
    } else if (span.attributes["openinference.span.kind"] !== undefined) {
      // OpenInference span — enrich with Traceloop/GenAI attrs, then route
      console.debug(
        `[Respan Debug] Processing OpenInference span: ${span.name} (kind: ${span.attributes["openinference.span.kind"]})`
      );

      const enrichmentAttrs = getOIEnrichmentAttrs(span);
      const enrichedSpan = Object.create(Object.getPrototypeOf(span));
      Object.assign(enrichedSpan, span);
      Object.defineProperty(enrichedSpan, "attributes", {
        value: { ...span.attributes, ...enrichmentAttrs },
        writable: false,
        configurable: true,
        enumerable: true,
      });

      this._processorManager.onEnd(enrichedSpan);
    } else if (span.attributes["respan.entity.log_type"] !== undefined) {
      // Enriched Respan span (from an instrumentation plugin)
      console.debug(
        `[Respan Debug] Processing enriched Respan span: ${span.name}`
      );
      this._processorManager.onEnd(span);
    } else {
      // This span has none of the above - it's pure auto-instrumentation noise (HTTP calls, etc.)
      console.debug(
        `[Respan Debug] Filtering out auto-instrumentation span: ${span.name}`
      );
    }
  }

  async shutdown(): Promise<void> {
    await this._processorManager.shutdown();
  }

  async forceFlush(): Promise<void> {
    await this._processorManager.forceFlush();
  }

  /**
   * Get the entity path from context
   */
  // Removed - now using imported getEntityPath function

  /**
   * Get the processor manager (for adding new processors)
   */
  public getProcessorManager(): MultiProcessorManager {
    return this._processorManager;
  }
}
