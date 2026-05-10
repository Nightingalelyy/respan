import os

import pytest

pytestmark = pytest.mark.integration

if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
    pytest.skip("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run.", allow_module_level=True)

respan_api_key = os.getenv("RESPAN_API_KEY")
if not respan_api_key:
    pytest.skip("Set RESPAN_API_KEY to run.", allow_module_level=True)

respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

from crewai import Agent, Crew, Task
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter

from respan_instrumentation_crewai import CrewAIInstrumentor


def test_real_crewai_gateway_pipeline_exports_spans():
    RespanTracer.reset_instance()
    span_exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="crewai-integration-test",
        api_key=respan_api_key,
        base_url=respan_base_url,
        is_batching_enabled=False,
        is_auto_instrument=False,
    )
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter)
    )

    instrumentor = CrewAIInstrumentor()
    instrumentor.activate()
    try:
        agent = Agent(
            role="Poet",
            goal="Write a short haiku about recursion in programming",
            backstory="You are a programmer who writes concise haikus.",
            llm="gpt-4o-mini",
            verbose=False,
        )
        task = Task(
            description="Write a haiku about recursion in programming.",
            expected_output="A single haiku.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        telemetry.flush()
        spans = span_exporter.get_finished_spans()

        assert result.raw
        assert spans
        assert any((span.attributes or {}).get(RESPAN_LOG_TYPE) for span in spans)
    finally:
        instrumentor.deactivate()
        RespanTracer.reset_instance()
