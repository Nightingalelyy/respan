/** Native Codex turns and items in the active Respan OpenTelemetry pipeline. */
import { context, createContextKey } from "@opentelemetry/api";
import {
  createCodexTurnState,
  fail,
  finishCodexTurn,
  trackCodexEvent,
  trackCodexRunResult,
  type CodexEmitterOptions,
  type CodexTurnState,
} from "./_otel_emitter.js";

type Fn = (...args: any[]) => any;
type Prototype = Record<string, any>;
export interface CodexSDKModule {
  Thread?: { prototype?: object };
}
export interface CodexSDKInstrumentorOptions extends CodexEmitterOptions {
  sdkModule?: CodexSDKModule;
}
interface Patch {
  prototype: Prototype;
  originals: Map<string, Fn>;
  wrappers: Map<string, Fn>;
  owners: Map<CodexSDKInstrumentor, CodexEmitterOptions>;
}
const patches = new WeakMap<object, Patch>();
const CURRENT_TURN = createContextKey("respan.codex-sdk.turn");
const wrappedIterators = new WeakSet<object>();

export class CodexSDKInstrumentor {
  public readonly name = "codex-sdk";
  private readonly sdkModule?: CodexSDKModule;
  private readonly options: CodexEmitterOptions;
  private patch?: Patch;
  private pending?: Promise<void>;
  private generation = 0;
  constructor({ sdkModule, ...options }: CodexSDKInstrumentorOptions = {}) {
    this.sdkModule = sdkModule;
    this.options = options;
  }
  activate(): Promise<void> {
    if (this.patch) return Promise.resolve();
    if (this.pending) return this.pending;
    const generation = this.generation;
    const pending = this.activateInternal(generation).finally(() => {
      if (this.pending === pending) this.pending = undefined;
    });
    this.pending = pending;
    return pending;
  }
  private async activateInternal(generation: number): Promise<void> {
    const sdk = this.sdkModule ?? (await import("@openai/codex-sdk"));
    if (generation !== this.generation) return;
    const prototype = sdk.Thread?.prototype as Prototype | undefined;
    if (
      !prototype ||
      typeof prototype.run !== "function" ||
      typeof prototype.runStreamed !== "function"
    )
      throw new Error(
        "CodexSDKInstrumentor requires Thread.prototype.run and runStreamed.",
      );
    let patch = patches.get(prototype);
    if (!patch) {
      patch = {
        prototype,
        originals: new Map(),
        wrappers: new Map(),
        owners: new Map(),
      };
      for (const key of ["run", "runStreamed", "runStreamedInternal"]) {
        if (typeof prototype[key] !== "function") continue;
        patch.originals.set(key, prototype[key]);
      }
      install(patch);
      patches.set(prototype, patch);
    }
    patch.owners.set(this, this.options);
    this.patch = patch;
  }
  deactivate(): void {
    this.generation++;
    const patch = this.patch;
    this.patch = undefined;
    if (!patch) return;
    patch.owners.delete(this);
    if (patch.owners.size) return;
    for (const [key, original] of patch.originals)
      if (patch.prototype[key] === patch.wrappers.get(key))
        patch.prototype[key] = original;
    patches.delete(patch.prototype);
  }
  isActive(): boolean {
    return Boolean(this.patch);
  }
}
function install(patch: Patch): void {
  const originalInternal = patch.originals.get("runStreamedInternal");
  if (originalInternal) {
    const wrapper = function (this: unknown, ...args: unknown[]) {
      const iterator = originalInternal.apply(this, args);
      const state = context.active().getValue(CURRENT_TURN) as
        CodexTurnState | undefined;
      if (state) wrapIterator(iterator, state, this, false);
      return iterator;
    };
    patch.wrappers.set("runStreamedInternal", wrapper);
    patch.prototype.runStreamedInternal = wrapper;
  }
  for (const key of ["run", "runStreamed"]) {
    const original = patch.originals.get(key)!;
    const wrapper = async function (this: unknown, ...args: unknown[]) {
      const options = [...patch.owners.values()][0] ?? {};
      const traceContent = [...patch.owners.values()].every(
        (owner) => owner.traceContent !== false,
      );
      const state = createCodexTurnState(args[0], this, {
        ...options,
        traceContent,
      });
      state.context = state.context.setValue(CURRENT_TURN, state);
      try {
        const result = await context.with(state.context, () =>
          original.apply(this, args),
        );
        if (key === "run") {
          context.with(state.context, () => trackCodexRunResult(state, result));
          context.with(state.context, () => finishCodexTurn(state, this));
        } else if (result?.events && typeof result.events.next === "function")
          wrapIterator(result.events, state, this, true);
        else context.with(state.context, () => finishCodexTurn(state, this));
        return result;
      } catch (error) {
        context.with(state.context, () => {
          fail(state.span, state.guard, error);
          finishCodexTurn(state, this);
        });
        throw error;
      }
    };
    patch.wrappers.set(key, wrapper);
    patch.prototype[key] = wrapper;
  }
}
function wrapIterator(
  iterator: any,
  state: CodexTurnState,
  thread: unknown,
  finish: boolean,
): void {
  if (wrappedIterators.has(iterator)) {
    // The public streaming call owns completion; the shared internal iterator
    // already owns event observation. Mark it without replacing its identity.
    if (finish) iterator[FINISH_TURN] = true;
    return;
  }
  wrappedIterators.add(iterator);
  if (finish) iterator[FINISH_TURN] = true;
  for (const key of ["next", "return", "throw"]) {
    const original = iterator[key];
    if (typeof original !== "function") continue;
    iterator[key] = function (this: unknown, ...args: unknown[]) {
      state.guard.observe();
      return context.with(state.context, async () => {
        try {
          const result = await original.apply(this, args);
          if (!result.done && key === "next")
            trackCodexEvent(state, result.value);
          if (result.done && iterator[FINISH_TURN])
            finishCodexTurn(state, thread);
          return result;
        } catch (error) {
          fail(state.span, state.guard, error);
          if (iterator[FINISH_TURN]) finishCodexTurn(state, thread);
          throw error;
        }
      });
    };
  }
}
const FINISH_TURN = Symbol("respan.codex-sdk.finish");
