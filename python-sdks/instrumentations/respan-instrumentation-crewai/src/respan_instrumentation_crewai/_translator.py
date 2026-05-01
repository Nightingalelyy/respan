"""CrewAI span translation helpers.

CrewAI instrumentation emits OpenInference spans, so the CrewAI package uses
the shared OpenInference translator as its single source of truth for mapping
into Respan's span shape.
"""

from respan_instrumentation_openinference import OpenInferenceTranslator


class CrewAITranslator(OpenInferenceTranslator):
    """Translate CrewAI OpenInference spans into Respan OTLP attributes."""

