import { AsyncLocalStorage } from "node:async_hooks";
import { context } from "@opentelemetry/api";
import {
  createCursorRunState,
  emitCursorToolExecution,
  emitFinalCursorRunSpans,
  markCursorRunError,
  recordCursorRunResult,
  registerCursorToolDefinitions,
  trackCursorCallback,
  trackCursorMessage,
  type CursorRunState,
} from "./_otel_emitter.js";
type AnyRecord = Record<string, any>;
type AnyFunction = (...args: any[]) => any;
export interface CursorSDKInstrumentorOptions {
  agentName?: string;
  sdkModule: Record<string, unknown>;
  traceContent?: boolean;
}
interface Patch {
  target: AnyRecord;
  key: string;
  original: AnyFunction;
  wrapped: AnyFunction;
  run?: CursorRunState;
}
interface Shared {
  owners: Set<CursorSDKInstrumentor>;
  patches: Patch[];
  agents: WeakMap<object, AnyRecord | undefined>;
  knownAgents: Set<WeakRef<AnyRecord>>;
  patchedAgents: WeakSet<object>;
  runs: WeakMap<object, CursorRunState>;
  prompt: AsyncLocalStorage<CursorRunState>;
}
const modules = new WeakMap<object, Shared>();
export class CursorSDKInstrumentor {
  readonly name = "cursor-sdk";
  private shared?: Shared;
  readonly options: CursorSDKInstrumentorOptions;
  constructor(options: CursorSDKInstrumentorOptions) {
    this.options = options;
  }
  isActive(): boolean {
    return !!this.shared;
  }
  activate(): void {
    if (this.shared) return;
    const Agent = this.options.sdkModule.Agent as AnyRecord;
    if (!Agent || (typeof Agent !== "object" && typeof Agent !== "function"))
      throw new Error(
        "CursorSDKInstrumentor requires sdkModule.Agent from @cursor/sdk.",
      );
    let shared = modules.get(Agent);
    if (!shared) {
      shared = {
        owners: new Set(),
        patches: [],
        agents: new WeakMap(),
        knownAgents: new Set(),
        patchedAgents: new WeakSet(),
        runs: new WeakMap(),
        prompt: new AsyncLocalStorage(),
      };
      modules.set(Agent, shared);
    }
    this.shared = shared;
    shared.owners.add(this);
    if (shared.owners.size !== 1) return;
    for (const reference of shared.knownAgents) {
      const agent = reference.deref();
      if (agent) wrapAgent(shared, agent, shared.agents.get(agent));
      else shared.knownAgents.delete(reference);
    }
    for (const key of ["create", "resume"])
      patch(
        shared,
        Agent,
        key,
        (original) =>
          function (this: unknown, ...args: any[]) {
            const value = original.apply(this, args);
            return settle(value, (agent: AnyRecord) => {
              if (shared!.owners.size)
                wrapAgent(shared!, agent, key === "create" ? args[0] : args[1]);
              return agent;
            });
          },
      );
    patch(
      shared,
      Agent,
      "prompt",
      (original) =>
        function (this: unknown, ...args: any[]) {
          if (!shared!.owners.size || shared!.prompt.getStore())
            return original.apply(this, args);
          const state = createState(shared!, args[0], undefined, args[1]);
          return shared!.prompt.run(state, () =>
            context.with(state.capture.ctx, () => {
              let result: any;
              try {
                result = original.apply(this, [
                  args[0],
                  wrapOptions(state, args[1]),
                  ...args.slice(2),
                ]);
              } catch (error) {
                markCursorRunError(state, error);
                finish(shared!, state);
                throw error;
              }
              return settle(
                result,
                (value: unknown) => {
                  recordCursorRunResult(state, value);
                  finish(shared!, state);
                  return value;
                },
                (error: unknown) => {
                  markCursorRunError(state, error);
                  finish(shared!, state);
                  throw error;
                },
              );
            }),
          );
        },
    );
    patch(
      shared,
      Agent,
      "getRun",
      (original) =>
        function (this: unknown, ...args: any[]) {
          const value = original.apply(this, args);
          return settle(value, (run: AnyRecord) => {
            if (shared!.owners.size) {
              const state = createState(shared!, undefined);
              recordCursorRunResult(state, run);
              wrapRun(shared!, run, state);
            }
            return run;
          });
        },
    );
  }
  deactivate(): void {
    const shared = this.shared;
    if (!shared) return;
    this.shared = undefined;
    shared.owners.delete(this);
    if (shared.owners.size) return;
    // Existing runs drain through their own handles; new calls are unpatched immediately.
    for (const entry of [...shared.patches].reverse())
      if (!entry.run) restore(entry);
    shared.patches = shared.patches.filter((entry) => entry.run);
    shared.patchedAgents = new WeakSet();
  }
}
function createState(
  shared: Shared,
  message?: unknown,
  agent?: AnyRecord,
  options?: AnyRecord,
): CursorRunState {
  const owners = [...shared.owners];
  return createCursorRunState(
    owners.find((owner) => owner.options.agentName)?.options.agentName ??
      "cursor-agent",
    owners.every((owner) => owner.options.traceContent !== false),
    message,
    agent,
    options,
  );
}
function patch(
  shared: Shared,
  target: AnyRecord,
  key: string,
  factory: (original: AnyFunction) => AnyFunction,
  run?: CursorRunState,
): void {
  const original = target[key];
  if (typeof original !== "function") return;
  const wrapped = factory(original);
  target[key] = wrapped;
  shared.patches.push({ target, key, original, wrapped, run });
}
function restore(entry: Patch): void {
  if (entry.target[entry.key] === entry.wrapped)
    entry.target[entry.key] = entry.original;
}
function finish(shared: Shared, state: CursorRunState): void {
  emitFinalCursorRunSpans(state);
  const owned = shared.patches.filter((entry) => entry.run === state);
  for (const entry of owned.reverse()) restore(entry);
  shared.patches = shared.patches.filter((entry) => entry.run !== state);
}
function wrapAgent(
  shared: Shared,
  agent: AnyRecord,
  options?: AnyRecord,
): void {
  if (!agent || typeof agent !== "object" || shared.patchedAgents.has(agent))
    return;
  if (!shared.agents.has(agent)) shared.knownAgents.add(new WeakRef(agent));
  shared.agents.set(agent, options);
  shared.patchedAgents.add(agent);
  patch(
    shared,
    agent,
    "send",
    (original) =>
      function (this: AnyRecord, ...args: any[]) {
        if (!shared.owners.size && !shared.prompt.getStore())
          return original.apply(this, args);
        const prompt = shared.prompt.getStore();
        const state =
          prompt ??
          createState(
            shared,
            args[0],
            this,
            args[1] ?? shared.agents.get(this),
          );
        if (!prompt)
          registerCursorToolDefinitions(
            state,
            args[1] ?? shared.agents.get(this),
          );
        return context.with(state.capture.ctx, () => {
          let value: any;
          try {
            const inherited = shared.agents.get(this);
            const supplied = args[1];
            const effective =
              supplied?.local?.customTools !== undefined ||
              !inherited?.local?.customTools
                ? supplied
                : {
                    ...supplied,
                    local: {
                      ...supplied?.local,
                      customTools: inherited.local.customTools,
                    },
                  };
            value = original.apply(this, [
              args[0],
              wrapOptions(state, effective),
              ...args.slice(2),
            ]);
          } catch (error) {
            markCursorRunError(state, error);
            finish(shared, state);
            throw error;
          }
          return settle(
            value,
            (run: AnyRecord) => {
              recordCursorRunResult(state, run);
              wrapRun(shared, run, state, !!prompt);
              return run;
            },
            (error: unknown) => {
              markCursorRunError(state, error);
              finish(shared, state);
              throw error;
            },
          );
        });
      },
  );
}
function wrapRun(
  shared: Shared,
  run: AnyRecord,
  state: CursorRunState,
  prompt = false,
): void {
  if (!run || typeof run !== "object" || shared.runs.has(run)) return;
  shared.runs.set(run, state);
  patch(
    shared,
    run,
    "stream",
    (original) =>
      function (this: AnyRecord, ...args: any[]) {
        // Keep the native generator object and all of its helper methods.
        const iterator = original.apply(this, args);
        if (!iterator || typeof iterator.next !== "function") return iterator;
        const next = iterator.next;
        iterator.next = function (...values: any[]) {
          return context
            .with(state.capture.ctx, () => next.apply(this, values))
            .then(
              (item: IteratorResult<unknown>) => {
                if (!item.done) trackCursorMessage(state, item.value);
                else {
                  recordCursorRunResult(state, run);
                  if (!prompt) finish(shared, state);
                }
                return item;
              },
              (error: unknown) => {
                markCursorRunError(state, error);
                if (!prompt) finish(shared, state);
                throw error;
              },
            );
        };
        // An early consumer break leaves a still-running native run open for wait/cancel.
        return iterator;
      },
    state,
  );
  for (const key of ["wait", "conversation", "cancel"])
    patch(
      shared,
      run,
      key,
      (original) =>
        function (this: AnyRecord, ...args: any[]) {
          const value = context.with(state.capture.ctx, () =>
            original.apply(this, args),
          );
          return settle(
            value,
            (result: unknown) => {
              recordCursorRunResult(state, key === "wait" ? result : run);
              if (!prompt && (key === "wait" || run.status !== "running"))
                finish(shared, state);
              return result;
            },
            (error: unknown) => {
              markCursorRunError(state, error);
              if (!prompt) finish(shared, state);
              throw error;
            },
          );
        },
      state,
    );
}
function wrapOptions(
  state: CursorRunState,
  options?: AnyRecord,
): AnyRecord | undefined {
  const wrapped = { ...options };
  options ??= {};
  for (const key of ["onStep", "onDelta"]) {
    const callback = options[key];
    if (key !== "onDelta" && typeof callback !== "function") continue;
    wrapped[key] = function (this: unknown, ...args: unknown[]) {
      trackCursorCallback(state, key, args[0]);
      return typeof callback === "function"
        ? callback.apply(this, args)
        : undefined;
    };
  }
  const custom = options.local?.customTools;
  if (custom) {
    const tools: AnyRecord = {};
    for (const [name, tool] of Object.entries(custom) as [
      string,
      AnyRecord,
    ][]) {
      if (typeof tool?.execute !== "function") {
        tools[name] = tool;
        continue;
      }
      tools[name] = {
        ...tool,
        execute: function (this: unknown, ...args: any[]) {
          let result: any;
          try {
            result = tool.execute.apply(tool, args);
          } catch (error) {
            emitCursorToolExecution(
              state,
              name,
              args[0],
              args[1],
              undefined,
              error,
              true,
            );
            throw error;
          }
          if (!result || typeof result.then !== "function") {
            emitCursorToolExecution(state, name, args[0], args[1], result);
            return result;
          }
          return settle(
            result,
            (value: unknown) => {
              emitCursorToolExecution(state, name, args[0], args[1], value);
              return value;
            },
            (error: unknown) => {
              emitCursorToolExecution(
                state,
                name,
                args[0],
                args[1],
                undefined,
                error,
                true,
              );
              throw error;
            },
          );
        },
      };
    }
    wrapped.local = { ...options.local, customTools: tools };
  }
  return wrapped;
}

function settle(
  value: any,
  fulfilled: (value: any) => any,
  rejected?: (error: unknown) => any,
): any {
  if (value && typeof value.then === "function")
    return value.then(fulfilled, rejected);
  return fulfilled(value);
}
