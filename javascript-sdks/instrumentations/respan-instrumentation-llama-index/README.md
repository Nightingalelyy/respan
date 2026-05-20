# @respan/instrumentation-llama-index

Respan instrumentation plugin for LlamaIndex TypeScript.

The instrumentor attaches to LlamaIndex's callback manager and emits Respan
OTEL spans for LLM calls, query engines, retrievers, synthesizers, agents, and
tool executions.

## Install

```bash
npm install @respan/respan @respan/instrumentation-llama-index llamaindex @llamaindex/openai
```

## Usage

```typescript
import { Respan } from "@respan/respan";
import { LlamaIndexInstrumentor } from "@respan/instrumentation-llama-index";
import { OpenAI } from "@llamaindex/openai";
import * as LlamaIndex from "llamaindex";

const workflowName = "llama_index_ts_basic_llm";

const respan = new Respan({
  appName: workflowName,
  traceContent: true,
  instrumentations: [
    new LlamaIndexInstrumentor({
      workflowName,
      llamaIndexModule: LlamaIndex,
    }),
  ],
});
await respan.initialize();

try {
  await respan.propagateAttributes(
    {
      trace_group_identifier: workflowName,
      metadata: { workflow_name: workflowName },
    },
    () =>
      respan.withWorkflow({ name: workflowName }, async () => {
        const llm = new OpenAI({ model: "gpt-4.1-nano" });
        const response = await llm.complete({
          prompt: "Say hello in three languages.",
        });
        console.log(response.text);
      }),
  );
} finally {
  await respan.shutdown();
}
```

Use `workflowName` when you want auto-emitted LlamaIndex spans to carry the
same workflow label that appears in Respan traces. Use the same value as the
`trace_group_identifier` when examples or application requests should be found
by workflow name.
