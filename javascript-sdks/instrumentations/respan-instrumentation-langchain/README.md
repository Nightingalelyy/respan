# @respan/instrumentation-langchain

Respan callback instrumentation for LangChain JS, LangGraph JS, and Langflow-style custom component flows.

This package emits Respan-compatible OpenTelemetry readable spans through `@respan/tracing`. It does not patch LangChain globally; attach the exported callback handler to runnable configs, chains, tools, or graph invocations.

```ts
import { Respan } from "@respan/respan";
import {
  LangChainInstrumentor,
  addRespanCallback,
} from "@respan/instrumentation-langchain";

const langchain = new LangChainInstrumentor();
const respan = new Respan({
  instrumentations: [langchain],
});
await respan.initialize();

const config = addRespanCallback({
  tags: ["langgraph"],
  metadata: { framework: "langgraph", langgraph_node: "router" },
});

await runnable.invoke({ question: "hello" }, config);
```

For Langflow-style component logic, reuse one handler for the component invocation so independent root callback runs are grouped into the same trace:

```ts
import { addRespanCallback, getCallbackHandler } from "@respan/instrumentation-langchain";

const handler = getCallbackHandler();
const config = addRespanCallback({
  tags: ["langflow", "custom-component"],
  metadata: { framework: "langflow", langflow_component: "RoutingComponent" },
}, handler);
```
