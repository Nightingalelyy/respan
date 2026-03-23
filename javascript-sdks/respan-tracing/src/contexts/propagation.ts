import { context } from "@opentelemetry/api";
import type { RespanParams } from "@respan/respan-sdk";
import {
  PROPAGATED_ATTRIBUTES_KEY,
  getPropagatedAttributes,
} from "../utils/context.js";

/**
 * Run a function within a context that propagates Respan attributes to all
 * spans created within its scope.
 *
 * Attributes are merged into every span at creation time via
 * `RespanCompositeProcessor.onStart()`. Nested calls merge attributes
 * (inner wins). Metadata dicts are merged, not replaced.
 *
 * @param attrs - Respan attributes to propagate (customer_identifier,
 *   thread_identifier, metadata, prompt, environment, etc.)
 * @param fn - The function to execute within the propagation scope
 * @returns The result of `fn`
 *
 * @example
 * ```typescript
 * await propagateAttributes(
 *   { customer_identifier: "user_123", thread_identifier: "conv_abc" },
 *   async () => {
 *     await Runner.run(agent, "Hello");
 *   }
 * );
 * ```
 */
export function propagateAttributes<T>(
  attrs: Partial<RespanParams>,
  fn: () => T
): T {
  // Merge with any already-active attributes (supports nesting)
  const parent = getPropagatedAttributes() ?? {};
  const merged: Partial<RespanParams> = { ...parent };

  for (const [key, value] of Object.entries(attrs)) {
    if (key === "metadata" && typeof value === "object" && value !== null) {
      // Merge metadata dicts instead of replacing
      merged.metadata = { ...(merged.metadata ?? {}), ...value };
    } else {
      (merged as any)[key] = value;
    }
  }

  const ctx = context.active().setValue(PROPAGATED_ATTRIBUTES_KEY, merged);
  return context.with(ctx, fn);
}
