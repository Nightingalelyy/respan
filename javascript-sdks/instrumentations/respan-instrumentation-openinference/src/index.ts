import { trace } from "@opentelemetry/api";
import {
  registerSpanTransformer,
  type RespanSpanTransformer,
  type SpanTransformerRegistration,
} from "@respan/tracing";
import { OpenInferenceTranslator } from "./_translator.js";
import { guardSpan, type CaptureOptions } from "./_capture.js";

export {
  OpenInferenceTranslator,
  isOpenInferenceSpan,
  prepareOpenInferenceSpanForExport,
  translateOpenInferenceSpan,
} from "./_translator.js";
export interface OpenInferenceInstrumentorOptions extends CaptureOptions {}
type Constructor = new (...args: any[]) => any;
interface NativeOwner {
  instance: any;
  count: number;
  modules: Set<unknown>;
  processorRegistration?: SpanTransformerRegistration;
  pending?: Promise<void>;
}
interface SharedState {
  owners: Map<object, OpenInferenceInstrumentorOptions>;
  native: Map<Constructor, NativeOwner>;
  transformer: RespanSpanTransformer;
}
const SHARED = Symbol.for("respan.instrumentation.openinference.v2");
function shared(): SharedState {
  const global = globalThis as any;
  if (!global[SHARED]) {
    const state = { owners: new Map(), native: new Map() } as SharedState;
    const translator = new OpenInferenceTranslator();
    state.transformer = {
      onStart(span, parent) {
        const options: OpenInferenceInstrumentorOptions = {};
        for (const owner of state.owners.values())
          for (const key of [
            "traceContent",
            "recordInputs",
            "recordOutputs",
          ] as const)
            if (owner[key] === false) options[key] = false;
        guardSpan(span, parent, options);
      },
      onEnd: translator.onEnd.bind(translator),
      prepareForExport: translator.prepareForExport.bind(translator),
    };
    Object.defineProperty(global, SHARED, { value: state });
  }
  return global[SHARED];
}
/**
 * Adapt an upstream OTel/OpenInference instrumentor, or route released
 * `@arizeai/openinference-core` helpers with the no-argument constructor.
 * Initialize the Respan runtime before activating this plugin.
 */
export class OpenInferenceInstrumentor {
  readonly name: string;
  private readonly instrumentorClass?: Constructor;
  private readonly sdkModule?: unknown;
  private readonly options: OpenInferenceInstrumentorOptions;
  private registration?: SpanTransformerRegistration;
  private nativeOwner?: NativeOwner;
  private active = false;
  private activation?: Promise<void>;
  private generation = 0;

  constructor(options?: OpenInferenceInstrumentorOptions);
  constructor(
    instrumentorClass: Constructor,
    sdkModule?: unknown,
    options?: OpenInferenceInstrumentorOptions,
  );
  constructor(
    instrumentorOrOptions?: Constructor | OpenInferenceInstrumentorOptions,
    sdkModule?: unknown,
    options: OpenInferenceInstrumentorOptions = {},
  ) {
    this.instrumentorClass =
      typeof instrumentorOrOptions === "function"
        ? instrumentorOrOptions
        : undefined;
    this.sdkModule = sdkModule;
    this.options = this.instrumentorClass
      ? { ...options }
      : {
          ...(instrumentorOrOptions as
            OpenInferenceInstrumentorOptions | undefined),
        };
    this.name = this.instrumentorClass
      ? `openinference-${this.instrumentorClass.name || "unknown"}`
      : "openinference";
  }
  activate(): void | Promise<void> {
    if (this.active) return;
    if (this.activation) return this.activation;
    const generation = ++this.generation;
    const state = shared();
    state.owners.set(this, this.options);
    try {
      this.registration = registerSpanTransformer(
        "@respan/instrumentation-openinference",
        state.transformer,
      );
      if (this.instrumentorClass) {
        let owner = state.native.get(this.instrumentorClass);
        if (!owner) {
          const instance = new this.instrumentorClass();
          owner = { instance, count: 1, modules: new Set() };
          this.nativeOwner = owner;
          state.native.set(this.instrumentorClass, owner);
          instance.setTracerProvider?.(trace.getTracerProvider());
          let result: unknown;
          if (typeof instance.instrument === "function")
            result = instance.instrument({
              tracerProvider: trace.getTracerProvider(),
            });
          else if (typeof instance.enable === "function")
            result = instance.enable();
          else if (
            typeof instance.onStart === "function" &&
            typeof instance.onEnd === "function"
          ) {
            owner.processorRegistration = registerSpanTransformer(
              `!@respan/openinference-producer/${this.instrumentorClass.name}`,
              {
                onStart: instance.onStart.bind(instance),
                onEnd: instance.onEnd.bind(instance),
                dispose: () => {
                  void instance.shutdown?.();
                },
              },
            );
          } else if (typeof instance.manuallyInstrument !== "function")
            throw new TypeError(
              "Expected an OpenInference/OTel instrumentor or SpanProcessor.",
            );
          if (result && typeof (result as Promise<void>).then === "function")
            owner.pending = Promise.resolve(result).then(() => {});
        } else {
          owner.count++;
          this.nativeOwner = owner;
        }
        if (
          this.sdkModule &&
          !owner.modules.has(this.sdkModule) &&
          typeof owner.instance.manuallyInstrument === "function"
        ) {
          owner.modules.add(this.sdkModule);
          if (owner.pending)
            owner.pending = owner.pending.then(() =>
              owner!.instance.manuallyInstrument(this.sdkModule),
            );
          else {
            const result = owner.instance.manuallyInstrument(this.sdkModule);
            if (result && typeof result.then === "function")
              owner.pending = Promise.resolve(result);
          }
        }
      }
      if (this.nativeOwner?.pending) {
        const pending = this.nativeOwner.pending.then(
          () => {
            if (generation === this.generation) this.active = true;
          },
          (error) => {
            if (generation === this.generation) this.release();
            throw error;
          },
        );
        this.activation = pending;
        void pending
          .finally(() => {
            if (this.activation === pending) this.activation = undefined;
          })
          .catch(() => {});
        return pending;
      }
      this.active = true;
    } catch (error) {
      this.release();
      throw error;
    }
  }
  deactivate(): void {
    this.release();
  }
  isActive(): boolean {
    return this.active;
  }
  private release(): void {
    ++this.generation;
    this.activation = undefined;
    const state = shared();
    state.owners.delete(this);
    this.registration?.unregister();
    this.registration = undefined;
    const owner = this.nativeOwner;
    this.nativeOwner = undefined;
    if (owner && --owner.count === 0) {
      owner.processorRegistration?.unregister();
      const disable = () => {
        if (owner.count !== 0) return;
        if (
          this.instrumentorClass &&
          state.native.get(this.instrumentorClass) === owner
        )
          state.native.delete(this.instrumentorClass);
        if (typeof owner.instance.disable === "function")
          owner.instance.disable();
        else owner.instance.uninstrument?.();
      };
      if (owner.pending) void owner.pending.then(disable, disable);
      else disable();
    }
    this.active = false;
  }
}
