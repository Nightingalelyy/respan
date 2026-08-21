"""Respan Instrumentation for Langfuse.

This package provides OTEL-compliant automatic instrumentation for Langfuse 
to send traces to Respan.

Usage:
    # IMPORTANT: Instrument BEFORE importing Langfuse
    from respan_instrumentation_langfuse import LangfuseInstrumentor
    
    LangfuseInstrumentor().instrument(api_key="your-api-key")
    
    # Now use Langfuse normally
    from langfuse import Langfuse, observe
    
    @observe()
    def my_function():
        return "Traced to Respan!"

Auto-instrumentation:
    Set RESPAN_API_KEY environment variable, then:
    
        opentelemetry-instrument python your_app.py
"""

from .instrumentor import LangfuseInstrumentor

__all__ = ["LangfuseInstrumentor"]
