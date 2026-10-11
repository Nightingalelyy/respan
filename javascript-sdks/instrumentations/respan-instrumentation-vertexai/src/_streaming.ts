import { context } from "@opentelemetry/api";
import { finishOperation, type VertexOperation } from "./_otel_emitter.js";
import { toSerializableValue } from "./_translator.js";
import { dataValue } from "./_capture.js";
const seen = new WeakSet<object>();
export function wrapStream(stream: any, op: VertexOperation): any {
  if (!stream || typeof stream !== "object" || seen.has(stream)) return stream;
  const original = stream[Symbol.asyncIterator];
  if (typeof original !== "function") return stream;
  seen.add(stream);
  const candidates = new Map<number, any>();
  let usageMetadata: any, modelVersion: any, responseId: any;
  function observe(chunk: any): void {
    op.capture.observe();
    const usage = dataValue(chunk, "usageMetadata");
    if (usage) usageMetadata = usage;
    modelVersion = dataValue(chunk, "modelVersion") ?? modelVersion;
    responseId = dataValue(chunk, "responseId") ?? responseId;
    if (!op.capture.outputs) {
      candidates.clear();
      return;
    }
    for (const candidate of chunk?.candidates ?? []) {
      const index = typeof candidate.index === "number" ? candidate.index : 0;
      let entry = candidates.get(index);
      if (!entry) {
        entry = {
          index,
          content: { role: candidate.content?.role ?? "model", parts: [] },
        };
        candidates.set(index, entry);
      }
      entry.content.parts.push(
        ...toSerializableValue(candidate.content?.parts ?? []),
      );
      if (candidate.finishReason !== undefined)
        entry.finishReason = candidate.finishReason;
    }
  }
  const response = () => ({
    usageMetadata,
    modelVersion,
    responseId,
    candidates: [...candidates.values()],
  });
  const wrappedIterators = new WeakSet<object>();
  stream[Symbol.asyncIterator] = function (...args: any[]) {
    const iterator = context.with(op.ctx, () => original.apply(this, args));
    if (wrappedIterators.has(iterator)) return iterator;
    wrappedIterators.add(iterator);
    const next = iterator.next.bind(iterator),
      ret =
        typeof iterator.return === "function"
          ? iterator.return.bind(iterator)
          : undefined,
      thr =
        typeof iterator.throw === "function"
          ? iterator.throw.bind(iterator)
          : undefined;
    iterator.next = async function (...args: any[]) {
      try {
        const value = await context.with(op.ctx, () => next(...args));
        if (value.done) finishOperation(op, response());
        else observe(value.value);
        return value;
      } catch (error) {
        finishOperation(op, undefined, error, true);
        throw error;
      }
    };
    if (ret)
      iterator.return = async function (...args: any[]) {
        try {
          const value = await context.with(op.ctx, () => ret(...args));
          finishOperation(op, response());
          return value;
        } catch (error) {
          finishOperation(op, undefined, error, true);
          throw error;
        }
      };
    if (thr)
      iterator.throw = async function (...args: any[]) {
        try {
          return await context.with(op.ctx, () => thr(...args));
        } catch (error) {
          finishOperation(op, undefined, error, true);
          throw error;
        } finally {
          if (!op.ended) finishOperation(op, response());
        }
      };
    return iterator;
  };
  return stream;
}
