/**
 * Respan instrumentation plugin for the Anthropic SDK.
 *
 * Monkey-patches `messages.create()` on the Anthropic client prototype
 * to emit OTEL spans with GenAI attributes for both standard and streaming
 * message responses.
 *
 * ```typescript
 * import { Respan } from "@respan/respan";
 * import { AnthropicInstrumentor } from "@respan/instrumentation-anthropic";
 *
 * const respan = new Respan({
 *   instrumentations: [new AnthropicInstrumentor()],
 * });
 * await respan.initialize();
 * ```
 */

import { loadAnthropicConstructors } from "./_helpers.js";
import { patchMessagesPrototype, type PatchedMessagesTarget } from "./_streaming.js";

export class AnthropicInstrumentor {
  public readonly name = "anthropic";
  private static readonly _sharedState = {
    activeInstances: 0,
    patchedTargets: [] as PatchedMessagesTarget[],
  };

  private _isInstrumented = false;

  async activate(): Promise<void> {
    if (this._isInstrumented) return;

    const anthropicConstructors = await loadAnthropicConstructors();
    if (anthropicConstructors.length === 0) {
      console.warn(
        "[Respan] Failed to activate Anthropic instrumentation — @anthropic-ai/sdk not found",
      );
      return;
    }

    const sharedState = AnthropicInstrumentor._sharedState;

    try {
      for (const Anthropic of anthropicConstructors) {
        const tempClient = new Anthropic({ apiKey: "sk-placeholder" });
        const messagesProto = Object.getPrototypeOf(tempClient.messages);

        if (
          !messagesProto ||
          typeof messagesProto.create !== "function" ||
          sharedState.patchedTargets.some((target) => target.messagesPrototype === messagesProto)
        ) {
          continue;
        }

        const patchedTarget = patchMessagesPrototype(messagesProto);
        if (patchedTarget) {
          sharedState.patchedTargets.push(patchedTarget);
        }
      }

      if (sharedState.patchedTargets.length === 0) {
        console.warn(
          "[Respan] Failed to activate Anthropic instrumentation — no compatible Messages prototypes found",
        );
        return;
      }

      sharedState.activeInstances += 1;
      this._isInstrumented = true;
    } catch (err) {
      console.warn("[Respan] Failed to activate Anthropic instrumentation:", err);
    }
  }

  deactivate(): void {
    if (!this._isInstrumented) return;

    const sharedState = AnthropicInstrumentor._sharedState;
    sharedState.activeInstances = Math.max(0, sharedState.activeInstances - 1);
    this._isInstrumented = false;

    if (sharedState.activeInstances > 0 || sharedState.patchedTargets.length === 0) return;

    try {
      for (const patchedTarget of sharedState.patchedTargets) {
        patchedTarget.messagesPrototype.create = patchedTarget.originalCreate;
      }
    } catch {
      /* ignore */
    }

    sharedState.patchedTargets = [];
  }
}
