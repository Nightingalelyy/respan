import { context } from "@opentelemetry/api";
import { dataValue, type CaptureOptions } from "./_capture.js";
import {
  startOperation,
  finishOperation,
  type Operation,
} from "./_span_emitter.js";
import { wrapStreamingResult } from "./_streaming.js";
import type { PatchedResourceTarget, TogetherOperationSpec } from "./_types.js";

export function instrumentCreateResult(
  result: any,
  op: Operation,
  streaming: boolean,
): any {
  if (!result || typeof result !== "object") {
    finishOperation(op, result);
    return result;
  }
  const complete = (value: any) => {
    if (streaming) return wrapStreamingResult(value, op);
    finishOperation(op, value);
    return value;
  };
  const failed = (error: unknown) => {
    finishOperation(op, undefined, error, true);
  };
  let parsedStarted = false;
  const parse =
    typeof result.parse === "function" ? result.parse.bind(result) : undefined;
  if (parse)
    result.parse = function (...args: any[]) {
      parsedStarted = true;
      const promise = parse(...args);
      promise
        .then((value: any) => {
          complete(value);
        }, failed)
        .catch(() => {});
      return promise;
    };
  else if (typeof result.then === "function")
    result
      .then((value: any) => {
        complete(value);
      }, failed)
      .catch(() => {});
  else complete(result);
  const asResponse =
    typeof result.asResponse === "function"
      ? result.asResponse.bind(result)
      : undefined;
  if (asResponse)
    result.asResponse = function (...args: any[]) {
      const rawOnly = !parsedStarted;
      const promise = asResponse(...args);
      promise
        .then((value: any) => {
          if (rawOnly && !parsedStarted) {
            if (typeof value?.status === "number")
              op.span.setAttribute("status_code", value.status);
            finishOperation(op);
          }
        }, failed)
        .catch(() => {});
      return promise;
    };
  return result;
}
export function patchResourceMethod(
  prototype: Record<string, any>,
  spec: TogetherOperationSpec,
  options: () => CaptureOptions = () => ({}),
): PatchedResourceTarget | null {
  if (typeof prototype?.[spec.method] !== "function") return null;
  const original = prototype[spec.method];
  const wrapped = function (this: any, ...args: any[]) {
    let op: Operation;
    try {
      op = startOperation(spec, args[0], options());
    } catch {
      return original.apply(this, args);
    }
    try {
      const streaming = dataValue(args[0], "stream") === true;
      const result = context.with(op.ctx, () => original.apply(this, args));
      return instrumentCreateResult(result, op, streaming);
    } catch (error) {
      finishOperation(op, undefined, error, true);
      throw error;
    }
  };
  prototype[spec.method] = wrapped;
  return { prototype, method: spec.method, original, wrapped };
}
