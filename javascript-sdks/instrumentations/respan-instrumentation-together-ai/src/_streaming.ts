import { context } from "@opentelemetry/api";
import { finishOperation, type Operation } from "./_span_emitter.js";
import { STREAM_INSTRUMENTED } from "./_constants.js";
import { dataValue } from "./_capture.js";

export function wrapStreamingResult(stream: any, op: Operation): any {
  if (!stream || typeof stream !== "object" || stream[STREAM_INSTRUMENTED])
    return stream;
  const iteratorMethod = stream[Symbol.asyncIterator];
  if (typeof iteratorMethod !== "function") {
    finishOperation(op, stream);
    return stream;
  }
  Object.defineProperty(stream, STREAM_INSTRUMENTED, {
    value: true,
    configurable: true,
  });
  const choices = new Map<number, any>();
  let usage: any, model: string | undefined, id: string | undefined;
  const raw: any[] = [];
  const observe = (chunk: any) => {
    op.capture.observe();
    const nextUsage = dataValue(chunk, "usage");
    if (nextUsage) usage = nextUsage;
    const nextModel = dataValue(chunk, "model"),
      nextId = dataValue(chunk, "id");
    if (typeof nextModel === "string") model = nextModel;
    if (typeof nextId === "string") id = nextId;
    if (!op.capture.outputs) {
      choices.clear();
      raw.length = 0;
      return;
    }
    if (op.spec.kind !== "chat" && op.spec.kind !== "text") {
      raw.push(chunk);
      return;
    }
    for (const choice of chunk?.choices ?? []) {
      const index = typeof choice.index === "number" ? choice.index : 0;
      let entry = choices.get(index);
      if (!entry) {
        entry = {
          index,
          message: { role: "assistant", content: "" },
          text: "",
          calls: new Map(),
        };
        choices.set(index, entry);
      }
      if (choice.finish_reason !== undefined)
        entry.finish_reason = choice.finish_reason;
      const delta = choice.delta ?? {};
      if (delta.role) entry.message.role = delta.role;
      for (const key of ["content", "reasoning", "reasoning_content"])
        if (typeof delta[key] === "string")
          entry.message[key] = (entry.message[key] ?? "") + delta[key];
      if (typeof choice.text === "string") entry.text += choice.text;
      for (const tool of delta.tool_calls ?? []) {
        const slot = typeof tool.index === "number" ? tool.index : 0;
        let call = entry.calls.get(slot);
        if (!call) {
          call = { type: "function", function: { name: "", arguments: "" } };
          entry.calls.set(slot, call);
        }
        if (typeof tool.id === "string") call.id = (call.id ?? "") + tool.id;
        if (tool.type) call.type = tool.type;
        if (typeof tool.function?.name === "string")
          call.function.name += tool.function.name;
        if (typeof tool.function?.arguments === "string")
          call.function.arguments += tool.function.arguments;
      }
    }
    if (op.spec.kind === "text" && chunk?.token?.text) {
      let entry = choices.get(0);
      if (!entry) {
        entry = {
          index: 0,
          text: "",
          message: { role: "assistant", content: "" },
          calls: new Map(),
        };
        choices.set(0, entry);
      }
      entry.text += chunk.token.text;
    }
  };
  const response = () => ({
    model,
    id,
    usage,
    choices: [...choices.values()]
      .sort((a, b) => a.index - b.index)
      .map((entry) => {
        const { calls, ...value } = entry;
        if (calls.size) value.message.tool_calls = [...calls.values()];
        return value;
      }),
    ...(raw.length ? { chunks: raw } : {}),
  });
  const wrappedIterators = new WeakSet<object>();
  stream[Symbol.asyncIterator] = function (...args: any[]) {
    const iterator = context.with(op.ctx, () =>
      iteratorMethod.apply(this, args),
    );
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
        const result = await context.with(op.ctx, () => next(...args));
        if (result.done) finishOperation(op, response());
        else observe(result.value);
        return result;
      } catch (error) {
        finishOperation(op, undefined, error, true);
        throw error;
      }
    };
    if (ret)
      iterator.return = async function (...args: any[]) {
        try {
          const result = await context.with(op.ctx, () => ret(...args));
          finishOperation(op, response());
          return result;
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
