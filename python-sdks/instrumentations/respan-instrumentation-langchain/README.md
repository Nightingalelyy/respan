# respan-instrumentation-langchain

Python instrumentation for LangChain-compatible callback flows. It emits
Respan/Traceloop spans through `respan-tracing`, so LangChain chains, chat
models, tools, retrievers, LangGraph runs, and Langflow components that execute
LangChain/LangGraph callbacks are exported into the normal Respan trace pipeline.

## Usage

```python
from respan import Respan
from respan_instrumentation_langchain import LangChainInstrumentor

respan = Respan(
    api_key="...",
    instrumentations=[LangChainInstrumentor()],
)

chain.invoke({"question": "hello"})
respan.flush()
```

For explicit per-run callbacks:

```python
from respan_instrumentation_langchain import add_respan_callback

chain.invoke(
    {"question": "hello"},
    config=add_respan_callback({"tags": ["demo"]}),
)
```

The same config helper works with LangGraph:

```python
graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    config=add_respan_callback({"configurable": {"thread_id": "thread-1"}}),
)
```

Langflow custom components can use `get_callback_handler()` or
`add_respan_callback()` when invoking LangChain or LangGraph objects.
