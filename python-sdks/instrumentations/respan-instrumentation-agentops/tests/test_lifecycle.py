from types import SimpleNamespace

from respan_instrumentation_agentops import AgentOpsInstrumentor


def test_lifecycle_reference_counts_processor_and_core(monkeypatch) -> None:
    import respan_instrumentation_agentops._instrumentation as lifecycle

    class Active:
        _span_processors: tuple[object, ...] = ()

    class Provider:
        _active_span_processor = Active()

        def force_flush(self):
            return True

    provider = Provider()
    core = SimpleNamespace(
        _initialized=False,
        initialized=False,
        provider=None,
        _meter_provider=None,
    )
    real_import = lifecycle.importlib.import_module

    def import_module(name: str):
        if name == "agentops":
            return SimpleNamespace()
        if name == "agentops.sdk.core":
            return SimpleNamespace(tracer=core)
        return real_import(name)

    monkeypatch.setattr(lifecycle.importlib, "import_module", import_module)
    monkeypatch.setattr(lifecycle.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(lifecycle, "_REFCOUNT", 0)
    monkeypatch.setattr(lifecycle, "_PROCESSOR", None)
    monkeypatch.setattr(lifecycle, "_PROVIDER", None)
    monkeypatch.setattr(lifecycle, "_AGENTOPS_CORE", None)
    monkeypatch.setattr(lifecycle, "_AGENTOPS_PROVIDER_PROXY", None)
    monkeypatch.setattr(lifecycle, "_OWNED_CORE_INITIALIZATION", False)
    monkeypatch.setattr(lifecycle, "_PREVIOUS_CORE_STATE", None)

    first = AgentOpsInstrumentor()
    second = AgentOpsInstrumentor(capture_content=False)
    first.activate()
    second.activate()

    assert lifecycle._REFCOUNT == 2
    assert core._initialized is True
    assert len(provider._active_span_processor._span_processors) == 1

    first.deactivate()
    assert lifecycle._REFCOUNT == 1
    second.deactivate()

    assert lifecycle._REFCOUNT == 0
    assert core._initialized is False
    assert provider._active_span_processor._span_processors == ()
