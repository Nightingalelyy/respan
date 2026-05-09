"""
Basic Haystack pipeline example with Respan tracing and gateway.

Traces pipeline runs, component executions, and LLM calls automatically.
Routes LLM calls through the Respan gateway (no separate OpenAI key needed).

Prerequisites:
    pip install respan-instrumentation-haystack

Environment variables:
    RESPAN_API_KEY   - Your Respan API key (used for both tracing and gateway)
    RESPAN_BASE_URL  - Respan API endpoint (default: https://api.respan.ai)
"""

import os

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor


def run_basic_pipeline() -> None:
    respan_api_key = os.environ["RESPAN_API_KEY"]
    respan_base_url = os.getenv(
        "RESPAN_BASE_URL",
        "https://api.respan.ai",
    ).rstrip("/")

    os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    os.environ["OPENAI_API_KEY"] = respan_api_key
    os.environ["OPENAI_BASE_URL"] = f"{respan_base_url}/api/openai"

    respan = Respan(
        api_key=respan_api_key,
        base_url=respan_base_url,
        instrumentations=[HaystackInstrumentor()],
    )

    template = """Answer the following question concisely: {{question}}"""

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component(
        "generator",
        OpenAIGenerator(model="gpt-4o-mini"),
    )
    pipeline.connect("prompt_builder", "generator")

    result = pipeline.run(
        {"prompt_builder": {"question": "What is the capital of France?"}}
    )
    print(result["generator"]["replies"][0])

    respan.flush()


if __name__ == "__main__":
    run_basic_pipeline()
