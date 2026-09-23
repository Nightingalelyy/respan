import { AsyncLocalStorage } from "node:async_hooks";
import { OpenAIChatCompletionsModel, OpenAIResponsesModel } from "@openai/agents";

const streaming = new AsyncLocalStorage<boolean>();
const patches: Array<{ prototype: any; original: any; wrapped: any }> = [];

export function isStreaming(): boolean {
  return streaming.getStore() === true;
}

export function installStreamPatches(): void {
  if (patches.length) return;
  for (const model of [OpenAIChatCompletionsModel, OpenAIResponsesModel]) {
    const prototype = model.prototype as any;
    const original = prototype.getStreamedResponse;
    if (typeof original !== "function") continue;
    const wrapped = function (this: any, ...args: any[]) {
      const source = original.apply(this, args);
      return {
        [Symbol.asyncIterator]() { return this; },
        next(value?: any) { return streaming.run(true, () => source.next(value)); },
        return(value?: any) { return streaming.run(true, () => source.return(value)); },
        throw(error?: any) { return streaming.run(true, () => source.throw(error)); },
      };
    };
    prototype.getStreamedResponse = wrapped;
    patches.push({ prototype, original, wrapped });
  }
}

export function removeStreamPatches(): void {
  for (const { prototype, original, wrapped } of patches.splice(0)) {
    if (prototype.getStreamedResponse === wrapped) prototype.getStreamedResponse = original;
  }
}
