import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  OpenInferenceInstrumentor,
  OpenInferenceTranslator,
} from "../dist/index.js";

const OI_SPAN_KIND = "openinference.span.kind";
const CLAUDE_AGENT_SDK_SCOPE_NAME =
  "@arizeai/openinference-instrumentation-claude-agent-sdk";

function makeSpan({
  name = "test-span",
  attributes = {},
  resourceAttributes = {},
  instrumentationScopeName = "test-scope",
} = {}) {
  return {
    name,
    attributes: { ...attributes },
    _attributes: { ...attributes },
    resource: { attributes: { ...resourceAttributes } },
    instrumentationScope: {
      name: instrumentationScopeName,
      version: "1.0.0",
    },
    instrumentationLibrary: {
      name: instrumentationScopeName,
      version: "1.0.0",
    },
  };
}

function resetTracerProvider(provider) {
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  if (provider) {
    trace.setGlobalTracerProvider(provider);
  }
}

function createFakeTracerProvider(processor) {
  return {
    activeSpanProcessor: processor,
    getTracer() {
      return {
        startSpan() {
          throw new Error("startSpan should not be called in this test");
        },
      };
    },
  };
}

class FakeInstrumentor {
  setTracerProvider(tracerProvider) {
    this.tracerProvider = tracerProvider;
  }

  instrument({ tracerProvider } = {}) {
    this.instrumentedWith = tracerProvider;
  }

  uninstrument() {
    this.uninstrumented = true;
  }
}

test("translator promotes Claude Agent SDK token fields and cleans redundant attributes", () => {
  const translator = new OpenInferenceTranslator();
  const span = makeSpan({
    name: "ClaudeAgent.query",
    instrumentationScopeName: CLAUDE_AGENT_SDK_SCOPE_NAME,
    attributes: {
      [OI_SPAN_KIND]: "AGENT",
      "input.value": "Explain tracing",
      "input.mime_type": "text/plain",
      "output.value": "Tracing adds spans",
      "output.mime_type": "text/plain",
      "llm.model_name": "claude-sonnet-4-6",
      "llm.provider": "Anthropic",
      "llm.system": "Anthropic",
      "llm.token_count.prompt": 3,
      "llm.token_count.completion": 97,
      "llm.token_count.total": 100,
      "process.pid": 1234,
    },
  });

  translator.onEnd(span);

  assert.equal(span.attributes["respan.entity.log_type"], "agent");
  assert.equal(span.attributes["llm.request.type"], "chat");
  assert.equal(span.attributes["gen_ai.request.model"], "claude-sonnet-4-6");
  assert.equal(span.attributes["gen_ai.system"], "anthropic");
  assert.equal(span.attributes["gen_ai.provider.name"], "anthropic");
  assert.equal(span.attributes.model, "claude-sonnet-4-6");
  assert.equal(span.attributes.prompt_tokens, 3);
  assert.equal(span.attributes.completion_tokens, 97);
  assert.equal(span.attributes.total_request_tokens, 100);
  assert.equal(span.attributes["traceloop.entity.input"], "Explain tracing");
  assert.equal(span.attributes["traceloop.entity.output"], "Tracing adds spans");
  assert.equal(span.attributes[OI_SPAN_KIND], undefined);
  assert.equal(span.attributes["llm.token_count.prompt"], undefined);
  assert.equal(span.attributes["input.mime_type"], undefined);
  assert.equal(span.attributes["process.pid"], undefined);
  assert.strictEqual(span._attributes, span.attributes);
});

test("translator rebuilds message content from text blocks", () => {
  const translator = new OpenInferenceTranslator();
  const span = makeSpan({
    attributes: {
      [OI_SPAN_KIND]: "LLM",
      "llm.input_messages.0.message.role": "user",
      "llm.input_messages.0.message.contents.0.message_content.text": "Hello",
      "llm.input_messages.0.message.contents.1.message_content.text": "world",
      "llm.output_messages.0.message.role": "assistant",
      "llm.output_messages.0.message.contents.0.message_content.text": "Hi there",
    },
  });

  translator.onEnd(span);

  assert.equal(span.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(span.attributes["gen_ai.prompt.0.content"], "Hello\nworld");
  assert.equal(span.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(span.attributes["gen_ai.completion.0.content"], "Hi there");
});

test("translator falls back to JSON when content blocks are not plain text", () => {
  const translator = new OpenInferenceTranslator();
  const span = makeSpan({
    attributes: {
      [OI_SPAN_KIND]: "LLM",
      "llm.input_messages.0.message.contents.0.message_content.type": "image",
    },
  });

  translator.onEnd(span);

  assert.equal(
    span.attributes["gen_ai.prompt.0.content"],
    JSON.stringify([{ type: "image" }]),
  );
});

test("instrumentor hook sanitizes exported OpenInference spans and restores passthrough after deactivate", () => {
  const capturedSpans = [];
  const processor = {
    onEnd(span) {
      capturedSpans.push(span);
    },
  };
  const provider = createFakeTracerProvider(processor);
  resetTracerProvider(provider);

  try {
    const instrumentor = new OpenInferenceInstrumentor(FakeInstrumentor);
    instrumentor.activate();

    const oiSpan = makeSpan({
      name: "ClaudeAgent.query",
      instrumentationScopeName: CLAUDE_AGENT_SDK_SCOPE_NAME,
      resourceAttributes: {
        "service.name": "respan-test",
        "process.pid": 42,
        "host.name": "local-dev",
        "custom.attr": "kept",
      },
      attributes: {
        [OI_SPAN_KIND]: "AGENT",
        "llm.model_name": "claude-sonnet-4-6",
        "llm.token_count.prompt": 3,
        "llm.token_count.completion": 97,
        "llm.token_count.total": 100,
      },
    });

    processor.onEnd(oiSpan);

    assert.equal(capturedSpans.length, 1);
    assert.notStrictEqual(capturedSpans[0], oiSpan);
    assert.deepEqual(capturedSpans[0].resource.attributes, {
      "service.name": "respan-test",
      "custom.attr": "kept",
    });
    assert.equal(capturedSpans[0].instrumentationScope.name, "");
    assert.equal(capturedSpans[0].instrumentationLibrary.name, "");
    assert.equal(oiSpan.resource.attributes["process.pid"], 42);
    assert.equal(
      oiSpan.instrumentationScope.name,
      CLAUDE_AGENT_SDK_SCOPE_NAME,
    );

    instrumentor.deactivate();

    const rawSpan = makeSpan({
      attributes: {
        custom: "value",
      },
      resourceAttributes: {
        "service.name": "respan-test",
        "process.pid": 99,
      },
    });

    processor.onEnd(rawSpan);

    assert.equal(capturedSpans.length, 2);
    assert.strictEqual(capturedSpans[1], rawSpan);
  } finally {
    resetTracerProvider();
  }
});

test("instrumentor hook leaves non-openinference spans unsanitized while active", () => {
  const capturedSpans = [];
  const processor = {
    onEnd(span) {
      capturedSpans.push(span);
    },
  };
  const provider = createFakeTracerProvider(processor);
  resetTracerProvider(provider);

  try {
    const instrumentor = new OpenInferenceInstrumentor(FakeInstrumentor);
    instrumentor.activate();

    const nonOiSpan = makeSpan({
      attributes: {
        custom: "value",
      },
      resourceAttributes: {
        "service.name": "respan-test",
        "process.pid": 77,
      },
    });

    processor.onEnd(nonOiSpan);

    assert.equal(capturedSpans.length, 1);
    assert.strictEqual(capturedSpans[0], nonOiSpan);

    instrumentor.deactivate();
  } finally {
    resetTracerProvider();
  }
});
