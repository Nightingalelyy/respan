import { hrTime } from "@opentelemetry/core";
import {
  STREAM_INSTRUMENTED,
  TOOL_USE_JSON_BUFFER_KEY,
} from "./_helpers.js";
import { emitErrorSpan, emitSuccessSpan } from "./_span_emitter.js";
import { emitToolSpansFromMessages } from "./_span_emitter.js";

export interface StreamState {
  message: any;
  usage: Record<string, any>;
  stopReason: string | null;
  stopSequence: string | null;
  contentBlocks: Map<number, any>;
}

function cloneContentBlock(block: any): any {
  if (!block || typeof block !== "object") return block;
  if (Array.isArray(block)) return block.map((entry) => cloneContentBlock(entry));
  return { ...block };
}

export function createStreamState(): StreamState {
  return {
    message: null,
    usage: {},
    stopReason: null,
    stopSequence: null,
    contentBlocks: new Map<number, any>(),
  };
}

export function buildMessageFromStreamState(
  state: StreamState,
  kwargs: Record<string, any>,
): any {
  const content = Array.from(state.contentBlocks.entries())
    .sort((left, right) => left[0] - right[0])
    .map(([, block]) => {
      const normalized = cloneContentBlock(block);
      if (normalized && typeof normalized === "object") {
        const jsonBuffer = normalized[TOOL_USE_JSON_BUFFER_KEY];
        delete normalized[TOOL_USE_JSON_BUFFER_KEY];

        if (
          normalized.type === "tool_use" &&
          typeof normalized.input === "string" &&
          typeof jsonBuffer === "string"
        ) {
          try {
            normalized.input = jsonBuffer.trim() ? JSON.parse(jsonBuffer) : {};
          } catch {
            normalized.input = jsonBuffer;
          }
        }
      }
      return normalized;
    });

  return {
    ...(state.message ?? {}),
    model: state.message?.model ?? kwargs.model,
    content,
    usage: state.usage,
    stop_reason: state.stopReason ?? state.message?.stop_reason ?? null,
    stop_sequence: state.stopSequence ?? state.message?.stop_sequence ?? null,
  };
}

export function updateStreamState(state: StreamState, event: any): void {
  if (!event || typeof event !== "object") return;

  if (event.type === "message_start") {
    state.message = { ...(event.message ?? {}) };
    state.usage = { ...(event.message?.usage ?? {}) };

    if (Array.isArray(event.message?.content)) {
      for (const [index, block] of event.message.content.entries()) {
        state.contentBlocks.set(index, cloneContentBlock(block));
      }
    }
    return;
  }

  if (event.type === "content_block_start") {
    state.contentBlocks.set(event.index, cloneContentBlock(event.content_block));
    return;
  }

  if (event.type === "content_block_delta") {
    const existingBlock = state.contentBlocks.get(event.index) ?? {};
    const delta = event.delta ?? {};

    if (delta.type === "text_delta") {
      existingBlock.type ??= "text";
      existingBlock.text = `${existingBlock.text ?? ""}${delta.text ?? ""}`;
      state.contentBlocks.set(event.index, existingBlock);
      return;
    }

    if (delta.type === "input_json_delta") {
      existingBlock.type ??= "tool_use";
      const nextBuffer =
        `${existingBlock[TOOL_USE_JSON_BUFFER_KEY] ?? ""}${delta.partial_json ?? ""}`;
      existingBlock[TOOL_USE_JSON_BUFFER_KEY] = nextBuffer;

      try {
        existingBlock.input = nextBuffer.trim() ? JSON.parse(nextBuffer) : {};
      } catch {
        existingBlock.input = nextBuffer;
      }

      state.contentBlocks.set(event.index, existingBlock);
    }
    return;
  }

  if (event.type === "message_delta") {
    state.stopReason = event.delta?.stop_reason ?? state.stopReason;
    state.stopSequence = event.delta?.stop_sequence ?? state.stopSequence;
    if (event.usage && typeof event.usage === "object") {
      state.usage = { ...state.usage, ...event.usage };
    }
  }
}

export function wrapStreamingCreateResult(
  streamResult: any,
  kwargs: Record<string, any>,
  startTime: [number, number],
): any {
  if (
    !streamResult ||
    typeof streamResult !== "object" ||
    streamResult[STREAM_INSTRUMENTED]
  ) {
    return streamResult;
  }

  Object.defineProperty(streamResult, STREAM_INSTRUMENTED, {
    value: true,
    configurable: true,
    enumerable: false,
  });

  const state = createStreamState();
  let hasEmitted = false;

  const emitFinalSpan = (error?: unknown) => {
    if (hasEmitted) return;
    hasEmitted = true;

    if (error) {
      emitErrorSpan(kwargs, startTime, error);
      return;
    }

    emitSuccessSpan(kwargs, startTime, buildMessageFromStreamState(state, kwargs));
  };

  const originalAsyncIterator = streamResult[Symbol.asyncIterator]?.bind(streamResult);
  if (typeof originalAsyncIterator !== "function") {
    emitSuccessSpan(kwargs, startTime, streamResult);
    return streamResult;
  }

  streamResult[Symbol.asyncIterator] = function () {
    const iterator = originalAsyncIterator();

    return {
      async next(...args: any[]) {
        try {
          const result = await iterator.next(...args);
          if (result.done) {
            emitFinalSpan();
          } else {
            updateStreamState(state, result.value);
          }
          return result;
        } catch (err) {
          emitFinalSpan(err);
          throw err;
        }
      },

      async return(value?: any) {
        try {
          const result = typeof iterator.return === "function"
            ? await iterator.return(value)
            : { done: true, value };
          emitFinalSpan();
          return result;
        } catch (err) {
          emitFinalSpan(err);
          throw err;
        }
      },

      async throw(err?: any) {
        emitFinalSpan(err);
        if (typeof iterator.throw === "function") {
          return iterator.throw(err);
        }
        throw err;
      },

      [Symbol.asyncIterator]() {
        return this;
      },
    };
  };

  return streamResult;
}

export function instrumentCreateResult(
  result: any,
  kwargs: Record<string, any>,
  startTime: [number, number],
): any {
  if (!result || typeof result !== "object") {
    return result;
  }

  let hasHandled = false;

  const handleSuccess = (value: any) => {
    if (kwargs?.stream === true) {
      return wrapStreamingCreateResult(value, kwargs, startTime);
    }

    if (!hasHandled) {
      hasHandled = true;
      emitSuccessSpan(kwargs, startTime, value);
    }
    return value;
  };

  const handleError = (err: unknown) => {
    if (hasHandled) return;
    hasHandled = true;
    emitErrorSpan(kwargs, startTime, err);
  };

  const originalThen = typeof result.then === "function" ? result.then.bind(result) : null;
  if (originalThen) {
    result.then = function (onfulfilled?: any, onrejected?: any) {
      return originalThen(
        (value: any) => {
          const instrumentedValue = handleSuccess(value);
          return onfulfilled ? onfulfilled(instrumentedValue) : instrumentedValue;
        },
        (reason: any) => {
          handleError(reason);
          if (onrejected) {
            return onrejected(reason);
          }
          throw reason;
        },
      );
    };
  }

  const originalCatch = typeof result.catch === "function" ? result.catch.bind(result) : null;
  if (originalCatch) {
    result.catch = function (onrejected?: any) {
      return originalCatch((reason: any) => {
        handleError(reason);
        if (onrejected) {
          return onrejected(reason);
        }
        throw reason;
      });
    };
  }

  const originalWithResponse =
    typeof result.withResponse === "function" ? result.withResponse.bind(result) : null;
  if (originalWithResponse) {
    result.withResponse = async function () {
      try {
        const response = await originalWithResponse();
        return {
          ...response,
          data: handleSuccess(response.data),
        };
      } catch (err) {
        handleError(err);
        throw err;
      }
    };
  }

  return result;
}

export interface PatchedMessagesTarget {
  messagesPrototype: any;
  originalCreate: any;
}

export function patchMessagesPrototype(messagesPrototype: any): PatchedMessagesTarget | null {
  if (!messagesPrototype || typeof messagesPrototype.create !== "function") {
    return null;
  }

  const patchedTarget = {
    messagesPrototype,
    originalCreate: messagesPrototype.create,
  };

  messagesPrototype.create = function (
    this: any,
    body: any,
    options?: any,
  ) {
    const startTime = hrTime();
    try {
      emitToolSpansFromMessages(body?.messages);
      const result = patchedTarget.originalCreate.call(this, body, options);
      return instrumentCreateResult(result, body, startTime);
    } catch (err: any) {
      emitErrorSpan(body, startTime, err);
      throw err;
    }
  };

  return patchedTarget;
}
