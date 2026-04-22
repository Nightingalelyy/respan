"""
Haystack RAG pipeline example with Respan tracing and gateway.

Demonstrates a simple retrieval-augmented generation pipeline using
an in-memory document store. All pipeline components are traced.
Routes LLM calls through the Respan gateway.

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

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Build a small document store
doc_store = InMemoryDocumentStore()
doc_store.write_documents(
    [
        Document(content="Python was created by Guido van Rossum and first released in 1991."),
        Document(content="Rust is a systems programming language focused on safety and performance."),
        Document(content="TypeScript is a typed superset of JavaScript developed by Microsoft."),
    ]
)

template = """
Given the following documents, answer the question.

Documents:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=doc_store, top_k=2))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component(
    "llm",
    OpenAIGenerator(model="gpt-4o-mini"),
)
pipe.connect("retriever.documents", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

result = pipe.run(
    {
        "retriever": {"query": "Who created Python?"},
        "prompt_builder": {"question": "Who created Python?"},
    }
)
print(result["llm"]["replies"][0])

respan.flush()
