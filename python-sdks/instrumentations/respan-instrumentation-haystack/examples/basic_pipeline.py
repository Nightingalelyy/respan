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
from dotenv import load_dotenv

load_dotenv()

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

# Point OpenAI traffic through Respan gateway (same base URL)
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor

# Initialize Respan BEFORE importing Haystack components
respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[HaystackInstrumentor()],
)

from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

template = """Answer the following question concisely: {{question}}"""

pipe = Pipeline()
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component(
    "llm",
    OpenAIGenerator(model="gpt-4o-mini"),
)
pipe.connect("prompt_builder", "llm")

result = pipe.run({"prompt_builder": {"question": "What is the capital of France?"}})
print(result["llm"]["replies"][0])

respan.flush()
