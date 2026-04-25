import builtins
import sys
from types import ModuleType, SimpleNamespace
from uuid import UUID, uuid4

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_langchain import (
    LangChainInstrumentor,
    RespanCallbackHandler,
    add_respan_callback,
    get_callback_handler,
)
from respan_instrumentation_langchain import _callback as callback_module
from respan_instrumentation_langchain._constants import LANGCHAIN_FRAMEWORK_ATTR
from respan_sdk.constants.span_attributes import (
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)
from respan_sdk.constants.otlp_constants import ERROR_MESSAGE_ATTR


def _capture_spans(monkeypatch):
    captured = []

    def _fake_build_readable_span(name, **kwargs):
        span = SimpleNamespace(name=name, attributes=kwargs.get("attributes", {}), kwargs=kwargs)
        captured.append(span)
        return span

    monkeypatch.setattr(callback_module, "build_readable_span", _fake_build_readable_span)
    monkeypatch.setattr(callback_module, "inject_span", lambda span: True)
    return captured


def test_get_callback_handler_returns_new_handler():
    first = get_callback_handler()
    second = get_callback_handler(include_content=False)

    assert isinstance(first, RespanCallbackHandler)
    assert isinstance(second, RespanCallbackHandler)
    assert first is not second
    assert second.include_content is False
    assert first.group_langflow_root_runs is True


def test_add_respan_callback_adds_handler_without_mutating_input_list():
    existing = object()
    handler = RespanCallbackHandler()
    config = {"callbacks": [existing], "tags": ["demo"]}

    new_config = add_respan_callback(config, handler)

    assert config["callbacks"] == [existing]
    assert new_config["callbacks"] == [existing, handler]
    assert new_config["tags"] == ["demo"]


def test_add_respan_callback_does_not_duplicate_respan_handlers():
    handler = RespanCallbackHandler()

    new_config = add_respan_callback({"callbacks": [handler]}, handler)

    assert new_config["callbacks"] == [handler]


def test_add_respan_callback_adds_to_callback_manager_like_object():
    handler = RespanCallbackHandler()
    manager = SimpleNamespace(handlers=[])
    manager.add_handler = lambda callback, inherit=True: manager.handlers.append(callback)

    new_config = add_respan_callback({"callbacks": manager}, handler)

    assert new_config["callbacks"] is manager
    assert manager.handlers == [handler]


def test_chain_root_and_child_emit_workflow_and_task_spans(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    root_run_id = uuid4()
    child_run_id = uuid4()

    handler.on_chain_start(
        {"name": "root_chain"},
        {"question": "hi"},
        run_id=root_run_id,
        tags=["langgraph"],
        metadata={"langgraph_node": "root"},
    )
    handler.on_chain_start(
        {"name": "child_chain"},
        {"input": "hi"},
        run_id=child_run_id,
        parent_run_id=root_run_id,
    )
    handler.on_chain_end({"answer": "hello"}, run_id=child_run_id)
    handler.on_chain_end({"done": True}, run_id=root_run_id)

    child_span, root_span = captured
    assert child_span.attributes[RESPAN_LOG_TYPE] == "task"
    assert root_span.attributes[RESPAN_LOG_TYPE] == "workflow"
    assert child_span.kwargs["trace_id"] == root_span.kwargs["trace_id"]
    assert child_span.kwargs["parent_id"] == root_span.kwargs["span_id"]
    assert root_span.attributes[LANGCHAIN_FRAMEWORK_ATTR] == "langgraph"


def test_root_run_uses_active_otel_span_as_parent(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    active_trace_id = "1234567890abcdef1234567890abcdef"
    active_span_id = "abcdef1234567890"

    class _FakeSpan:
        def get_span_context(self):
            return SimpleNamespace(
                trace_id=int(active_trace_id, 16),
                span_id=int(active_span_id, 16),
            )

    monkeypatch.setattr(callback_module.trace, "get_current_span", lambda: _FakeSpan())

    handler.on_chain_start({"name": "root_chain"}, {}, run_id=run_id)
    handler.on_chain_end({"ok": True}, run_id=run_id)

    assert captured[0].kwargs["trace_id"] == active_trace_id
    assert captured[0].kwargs["parent_id"] == active_span_id


def test_explicit_langflow_handler_groups_root_runs_without_active_parent(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = get_callback_handler()
    first_run_id = uuid4()
    second_run_id = uuid4()
    metadata = {
        "framework": "langflow",
        "langflow_component": "DemoComponent",
    }

    handler.on_tool_start(
        {"name": "route_to_workspace"},
        "security",
        run_id=first_run_id,
        tags=["langflow"],
        metadata=metadata,
    )
    handler.on_tool_end("secops-critical", run_id=first_run_id)
    handler.on_chain_start(
        {"name": "component_chain"},
        {"question": "hi"},
        run_id=second_run_id,
        tags=["langflow"],
        metadata=metadata,
    )
    handler.on_chain_end({"answer": "ok"}, run_id=second_run_id)

    assert captured[0].kwargs["trace_id"] == captured[1].kwargs["trace_id"]
    assert captured[0].kwargs["parent_id"] is None
    assert captured[1].kwargs["parent_id"] is None


def test_chat_model_start_end_maps_messages_usage_model_and_tool_calls(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    user_message = SimpleNamespace(type="human", content="What is 2+2?")
    ai_message = SimpleNamespace(
        type="ai",
        content="4",
        tool_calls=[
            {
                "id": "call_1",
                "name": "calculator",
                "args": {"expression": "2+2"},
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 3},
    )
    generation = SimpleNamespace(message=ai_message)
    response = SimpleNamespace(generations=[[generation]], llm_output={"model_name": "gpt-4o-mini"})

    handler.on_chat_model_start(
        {"name": "ChatOpenAI", "kwargs": {"model": "gpt-4o-mini"}},
        [[user_message]],
        run_id=run_id,
    )
    handler.on_llm_end(response, run_id=run_id)

    attrs = captured[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "chat"
    assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == LLMRequestTypeValues.CHAT.value
    assert attrs[LLM_REQUEST_TYPE] == "chat"
    assert attrs[LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[LLM_USAGE_PROMPT_TOKENS] == 10
    assert attrs[LLM_USAGE_COMPLETION_TOKENS] == 3
    assert attrs["gen_ai.prompt.0.role"] == "user"
    assert attrs["gen_ai.completion.0.role"] == "assistant"
    assert "calculator" in attrs["respan.span.tool_calls"]


def test_llm_json_code_fence_output_is_unwrapped_for_logged_content(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    fenced_json = '```json\n{"owner": "Security Operations Team"}\n```'
    ai_message = SimpleNamespace(type="ai", content=fenced_json)
    generation = SimpleNamespace(message=ai_message)
    response = SimpleNamespace(generations=[[generation]], llm_output={"model_name": "gpt-4o-mini"})

    handler.on_chat_model_start(
        {"name": "ChatOpenAI", "kwargs": {"model": "gpt-4o-mini"}},
        [[SimpleNamespace(type="human", content="Route this case")]],
        run_id=run_id,
    )
    handler.on_llm_end(response, run_id=run_id)

    attrs = captured[0].attributes
    assert attrs["gen_ai.completion.0.content"] == '{"owner": "Security Operations Team"}'
    assert "```" not in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert "```" not in attrs["output"]


def test_chain_json_code_fence_output_is_unwrapped_for_logged_output(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    fenced_json = '```json\n{"next_step": "Investigate login history"}\n```'

    handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=run_id)
    handler.on_chain_end({"answer": fenced_json}, run_id=run_id)

    attrs = captured[0].attributes
    assert "```" not in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert "```" not in attrs["output"]
    assert "Investigate login history" in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]


def test_llm_start_new_tokens_and_end_emit_completion_span(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    generation = SimpleNamespace(text="")
    response = SimpleNamespace(
        generations=[[generation]],
        llm_output={"token_usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    )

    handler.on_llm_start(
        {"name": "OpenAI", "kwargs": {"model_name": "text-davinci"}},
        ["Write a haiku"],
        run_id=run_id,
    )
    handler.on_llm_new_token("old ", run_id=run_id)
    handler.on_llm_new_token("pond", run_id=run_id)
    handler.on_llm_end(response, run_id=run_id)

    attrs = captured[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "completion"
    assert attrs[LLM_REQUEST_TYPE] == "completion"
    assert attrs[LLM_REQUEST_MODEL] == "text-davinci"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == '[[{"role": "assistant", "content": ""}]]'


def test_tool_start_end_maps_tool_fields(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()

    handler.on_tool_start(
        {"name": "calculator"},
        "2+2",
        run_id=run_id,
        inputs={"expression": "2+2"},
    )
    handler.on_tool_end({"answer": 4}, run_id=run_id)

    attrs = captured[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "tool"
    assert attrs[GEN_AI_TOOL_NAME] == "calculator"
    assert attrs[GEN_AI_TOOL_CALL_ARGUMENTS] == '{"expression": "2+2"}'
    assert attrs[GEN_AI_TOOL_CALL_RESULT] == '{"answer": 4}'


def test_retriever_start_end_serializes_documents(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    document = SimpleNamespace(page_content="doc text", metadata={"source": "unit"})

    handler.on_retriever_start({"name": "vectorstore"}, "query", run_id=run_id)
    handler.on_retriever_end([document], run_id=run_id)

    attrs = captured[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert "doc text" in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]


def test_error_callbacks_mark_span_as_error(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    error = RuntimeError("failed")

    handler.on_chain_start({"name": "chain"}, {}, run_id=run_id)
    handler.on_chain_error(error, run_id=run_id)

    assert captured[0].kwargs["status_code"] == 500
    assert captured[0].kwargs["error_message"] == "failed"
    assert captured[0].attributes[ERROR_MESSAGE_ATTR] == "failed"
    assert captured[0].attributes["status_code"] == 500


def test_llm_tool_and_retriever_error_callbacks_mark_spans_as_error(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    llm_run_id = uuid4()
    tool_run_id = uuid4()
    retriever_run_id = uuid4()

    handler.on_llm_start({"name": "llm"}, ["prompt"], run_id=llm_run_id)
    handler.on_llm_error(RuntimeError("llm failed"), run_id=llm_run_id)
    handler.on_tool_start({"name": "tool"}, "input", run_id=tool_run_id)
    handler.on_tool_error(RuntimeError("tool failed"), run_id=tool_run_id)
    handler.on_retriever_start({"name": "retriever"}, "query", run_id=retriever_run_id)
    handler.on_retriever_error(RuntimeError("retriever failed"), run_id=retriever_run_id)

    assert [span.kwargs["status_code"] for span in captured] == [500, 500, 500]
    assert [span.kwargs["error_message"] for span in captured] == [
        "llm failed",
        "tool failed",
        "retriever failed",
    ]
    assert [span.attributes[ERROR_MESSAGE_ATTR] for span in captured] == [
        "llm failed",
        "tool failed",
        "retriever failed",
    ]
    assert [span.attributes["status_code"] for span in captured] == [500, 500, 500]


def test_on_text_uses_streamed_text_when_run_has_no_output(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()

    handler.on_chain_start({"name": "streaming_chain"}, {}, run_id=run_id)
    handler.on_text("hello ", run_id=run_id)
    handler.on_text("world", run_id=run_id)
    handler.on_chain_end(None, run_id=run_id)

    assert captured[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "hello world"


def test_agent_action_and_finish_emit_event_spans(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()
    action = SimpleNamespace(tool="search", tool_input={"q": "respan"}, log="searching")
    finish = SimpleNamespace(return_values={"output": "done"})

    handler.on_chain_start({"name": "agent"}, {}, run_id=run_id)
    handler.on_agent_action(action, run_id=run_id)
    handler.on_agent_finish(finish, run_id=run_id)

    assert captured[0].attributes[RESPAN_LOG_TYPE] == "tool"
    assert captured[0].attributes[GEN_AI_TOOL_NAME] == "search"
    assert captured[1].attributes[RESPAN_LOG_TYPE] == "agent"


def test_custom_event_and_graph_lifecycle_events_emit_spans(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()

    handler.on_custom_event("custom_step", {"value": 1}, run_id=run_id)
    handler.on_interrupt(SimpleNamespace(run_id=run_id, status="interrupted"))
    handler.on_resume(SimpleNamespace(run_id=run_id, checkpoint_id="c1"))

    assert [span.name for span in captured] == [
        "custom_step",
        "langgraph.interrupt",
        "langgraph.resume",
    ]
    assert captured[1].attributes[LANGCHAIN_FRAMEWORK_ATTR] == "langgraph"


def test_on_retry_records_retry_metadata_on_final_span(monkeypatch):
    captured = _capture_spans(monkeypatch)
    handler = RespanCallbackHandler()
    run_id = uuid4()

    handler.on_chain_start({"name": "chain"}, {}, run_id=run_id)
    handler.on_retry(SimpleNamespace(attempt_number=2), run_id=run_id)
    handler.on_chain_end({"ok": True}, run_id=run_id)

    attrs = captured[0].attributes
    assert attrs["langchain.retry_count"] == 1
    assert "attempt_number" in attrs["langchain.retry_state"]


def _install_fake_langchain_modules(monkeypatch):
    class FakeCallbackManager:
        @classmethod
        def configure(
            cls,
            inheritable_callbacks=None,
            local_callbacks=None,
            *args,
            **kwargs,
        ):
            return SimpleNamespace(
                manager=cls,
                inheritable_callbacks=inheritable_callbacks,
                local_callbacks=local_callbacks,
            )

    class FakeAsyncCallbackManager(FakeCallbackManager):
        pass

    langchain_core = ModuleType("langchain_core")
    callbacks = ModuleType("langchain_core.callbacks")
    manager = ModuleType("langchain_core.callbacks.manager")
    manager.CallbackManager = FakeCallbackManager
    manager.AsyncCallbackManager = FakeAsyncCallbackManager
    callbacks.BaseCallbackHandler = object
    callbacks.manager = manager
    langchain_core.callbacks = callbacks

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks.manager", manager)

    return FakeCallbackManager, FakeAsyncCallbackManager


def test_instrumentor_patches_and_restores_callback_managers(monkeypatch):
    fake_manager, fake_async_manager = _install_fake_langchain_modules(monkeypatch)
    original_configure = fake_manager.__dict__["configure"]
    instrumentor = LangChainInstrumentor()

    instrumentor.activate()
    configured = fake_manager.configure()
    async_configured = fake_async_manager.configure()

    assert configured.inheritable_callbacks == [instrumentor.callback_handler]
    assert async_configured.inheritable_callbacks == [instrumentor.callback_handler]

    instrumentor.deactivate()

    assert fake_manager.__dict__["configure"] is original_configure


def test_instrumentor_patches_langgraph_config_helpers(monkeypatch):
    _install_fake_langchain_modules(monkeypatch)
    langgraph = ModuleType("langgraph")
    callbacks = ModuleType("langgraph.callbacks")
    callbacks.get_sync_graph_callback_manager_for_config = lambda config, **kwargs: config
    callbacks.get_async_graph_callback_manager_for_config = lambda config, **kwargs: config
    langgraph.callbacks = callbacks
    monkeypatch.setitem(sys.modules, "langgraph", langgraph)
    monkeypatch.setitem(sys.modules, "langgraph.callbacks", callbacks)
    instrumentor = LangChainInstrumentor()

    instrumentor.activate()
    config = callbacks.get_sync_graph_callback_manager_for_config({"callbacks": []})

    assert config["callbacks"] == [instrumentor.callback_handler]

    instrumentor.deactivate()


def test_instrumentor_logs_warning_when_langchain_missing(monkeypatch, caplog):
    original_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "langchain_core.callbacks.manager":
            raise ImportError("missing langchain")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    instrumentor = LangChainInstrumentor()

    with caplog.at_level("WARNING"):
        instrumentor.activate()

    assert "Failed to activate LangChain instrumentation" in caplog.text
    assert instrumentor._is_instrumented is False


def test_run_id_to_hex_accepts_uuid_strings_and_plain_strings():
    run_id = UUID("12345678-1234-5678-1234-567812345678")

    assert callback_module._run_id_to_hex(run_id) == "12345678123456781234567812345678"
    assert callback_module._run_id_to_hex(str(run_id)) == "12345678123456781234567812345678"
    assert len(callback_module._run_id_to_hex("plain-id")) == 32
