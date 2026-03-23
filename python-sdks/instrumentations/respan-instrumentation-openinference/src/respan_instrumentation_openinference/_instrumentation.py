import importlib
import logging

from opentelemetry import trace

logger = logging.getLogger(__name__)


class _OpenInferenceWrapper:
    """Base for OpenInference-backed instrumentors.

    Subclasses set ``_oi_import_path``, ``_oi_class_name``, and ``name``.
    ``activate()`` / ``deactivate()`` match the Respan Instrumentation protocol.
    """

    _oi_import_path: str   # e.g. "openinference.instrumentation.google_adk"
    _oi_class_name: str    # e.g. "GoogleADKInstrumentor"
    name: str              # e.g. "google-adk"

    def __init__(self) -> None:
        self._is_instrumented = False
        self._instrumentor = None

    def activate(self) -> None:
        try:
            mod = importlib.import_module(self._oi_import_path)
            cls = getattr(mod, self._oi_class_name)
            self._instrumentor = cls()
            tp = trace.get_tracer_provider()
            self._instrumentor.instrument(tracer_provider=tp)
            self._is_instrumented = True
            logger.info("%s instrumentation activated (via OpenInference)", self.name)
        except ImportError as exc:
            logger.warning(
                "Failed to activate %s instrumentation — missing dependency: %s",
                self.name, exc,
            )

    def deactivate(self) -> None:
        if self._is_instrumented and self._instrumentor is not None:
            try:
                self._instrumentor.uninstrument()
            except Exception:
                pass
            self._is_instrumented = False
        logger.info("%s instrumentation deactivated", self.name)


class _OpenInferenceSpanProcessorWrapper:
    """Base for OpenInference packages that expose a SpanProcessor instead of
    an Instrumentor (e.g. pydantic-ai, strands-agents).

    These packages transform already-emitted OTEL spans into OpenInference
    format, so we register them as a span processor on the tracer provider
    rather than calling ``.instrument()``.
    """

    _oi_import_path: str
    _oi_class_name: str
    name: str

    def __init__(self) -> None:
        self._is_instrumented = False
        self._processor = None

    def activate(self) -> None:
        try:
            mod = importlib.import_module(self._oi_import_path)
            cls = getattr(mod, self._oi_class_name)
            self._processor = cls()
            tp = trace.get_tracer_provider()
            if hasattr(tp, "add_span_processor"):
                tp.add_span_processor(self._processor)
            self._is_instrumented = True
            logger.info("%s span processor activated (via OpenInference)", self.name)
        except ImportError as exc:
            logger.warning(
                "Failed to activate %s span processor — missing dependency: %s",
                self.name, exc,
            )

    def deactivate(self) -> None:
        if self._is_instrumented and self._processor is not None:
            try:
                self._processor.shutdown()
            except Exception:
                pass
            self._is_instrumented = False
        logger.info("%s span processor deactivated", self.name)


# ---------------------------------------------------------------------------
# Standard instrumentors (each = 5 lines)
# ---------------------------------------------------------------------------

class AgentSpecInstrumentor(_OpenInferenceWrapper):
    name = "agentspec"
    _oi_import_path = "openinference.instrumentation.agentspec"
    _oi_class_name = "AgentSpecInstrumentor"

class AgnoInstrumentor(_OpenInferenceWrapper):
    name = "agno"
    _oi_import_path = "openinference.instrumentation.agno"
    _oi_class_name = "AgnoInstrumentor"

class AnthropicInstrumentor(_OpenInferenceWrapper):
    name = "anthropic"
    _oi_import_path = "openinference.instrumentation.anthropic"
    _oi_class_name = "AnthropicInstrumentor"

class AutogenAgentChatInstrumentor(_OpenInferenceWrapper):
    name = "autogen-agentchat"
    _oi_import_path = "openinference.instrumentation.autogen_agentchat"
    _oi_class_name = "AutogenAgentChatInstrumentor"

class BeeAIInstrumentor(_OpenInferenceWrapper):
    name = "beeai"
    _oi_import_path = "openinference.instrumentation.beeai"
    _oi_class_name = "BeeAIInstrumentor"

class BedrockInstrumentor(_OpenInferenceWrapper):
    name = "bedrock"
    _oi_import_path = "openinference.instrumentation.bedrock"
    _oi_class_name = "BedrockInstrumentor"

class ClaudeAgentSDKInstrumentor(_OpenInferenceWrapper):
    name = "claude-agent-sdk"
    _oi_import_path = "openinference.instrumentation.claude_agent_sdk"
    _oi_class_name = "ClaudeAgentSDKInstrumentor"

class CrewAIInstrumentor(_OpenInferenceWrapper):
    name = "crewai"
    _oi_import_path = "openinference.instrumentation.crewai"
    _oi_class_name = "CrewAIInstrumentor"

class DSPyInstrumentor(_OpenInferenceWrapper):
    name = "dspy"
    _oi_import_path = "openinference.instrumentation.dspy"
    _oi_class_name = "DSPyInstrumentor"

class GoogleADKInstrumentor(_OpenInferenceWrapper):
    name = "google-adk"
    _oi_import_path = "openinference.instrumentation.google_adk"
    _oi_class_name = "GoogleADKInstrumentor"

class GoogleGenAIInstrumentor(_OpenInferenceWrapper):
    name = "google-genai"
    _oi_import_path = "openinference.instrumentation.google_genai"
    _oi_class_name = "GoogleGenAIInstrumentor"

class GroqInstrumentor(_OpenInferenceWrapper):
    name = "groq"
    _oi_import_path = "openinference.instrumentation.groq"
    _oi_class_name = "GroqInstrumentor"

class GuardrailsInstrumentor(_OpenInferenceWrapper):
    name = "guardrails"
    _oi_import_path = "openinference.instrumentation.guardrails"
    _oi_class_name = "GuardrailsInstrumentor"

class HaystackInstrumentor(_OpenInferenceWrapper):
    name = "haystack"
    _oi_import_path = "openinference.instrumentation.haystack"
    _oi_class_name = "HaystackInstrumentor"

class InstructorInstrumentor(_OpenInferenceWrapper):
    name = "instructor"
    _oi_import_path = "openinference.instrumentation.instructor"
    _oi_class_name = "InstructorInstrumentor"

class LangChainInstrumentor(_OpenInferenceWrapper):
    name = "langchain"
    _oi_import_path = "openinference.instrumentation.langchain"
    _oi_class_name = "LangChainInstrumentor"

class LiteLLMInstrumentor(_OpenInferenceWrapper):
    name = "litellm"
    _oi_import_path = "openinference.instrumentation.litellm"
    _oi_class_name = "LiteLLMInstrumentor"

class LlamaIndexInstrumentor(_OpenInferenceWrapper):
    name = "llama-index"
    _oi_import_path = "openinference.instrumentation.llama_index"
    _oi_class_name = "LlamaIndexInstrumentor"

class MCPInstrumentor(_OpenInferenceWrapper):
    name = "mcp"
    _oi_import_path = "openinference.instrumentation.mcp"
    _oi_class_name = "MCPInstrumentor"

class MistralAIInstrumentor(_OpenInferenceWrapper):
    name = "mistralai"
    _oi_import_path = "openinference.instrumentation.mistralai"
    _oi_class_name = "MistralAIInstrumentor"

class OpenAIInstrumentor(_OpenInferenceWrapper):
    name = "openai-oi"
    _oi_import_path = "openinference.instrumentation.openai"
    _oi_class_name = "OpenAIInstrumentor"

class OpenAIAgentsInstrumentor(_OpenInferenceWrapper):
    name = "openai-agents-oi"
    _oi_import_path = "openinference.instrumentation.openai_agents"
    _oi_class_name = "OpenAIAgentsInstrumentor"

class PipecatInstrumentor(_OpenInferenceWrapper):
    name = "pipecat"
    _oi_import_path = "openinference.instrumentation.pipecat"
    _oi_class_name = "PipecatInstrumentor"

class PortkeyInstrumentor(_OpenInferenceWrapper):
    name = "portkey"
    _oi_import_path = "openinference.instrumentation.portkey"
    _oi_class_name = "PortkeyInstrumentor"

class SmolagentsInstrumentor(_OpenInferenceWrapper):
    name = "smolagents"
    _oi_import_path = "openinference.instrumentation.smolagents"
    _oi_class_name = "SmolagentsInstrumentor"

class VertexAIInstrumentor(_OpenInferenceWrapper):
    name = "vertexai"
    _oi_import_path = "openinference.instrumentation.vertexai"
    _oi_class_name = "VertexAIInstrumentor"

# ---------------------------------------------------------------------------
# SpanProcessor-based integrations (non-standard OI pattern)
# These wrap SDKs that already emit native OTEL spans; the OI package only
# transforms them into OpenInference format via a SpanProcessor.
# ---------------------------------------------------------------------------

class PydanticAIInstrumentor(_OpenInferenceSpanProcessorWrapper):
    name = "pydantic-ai"
    _oi_import_path = "openinference.instrumentation.pydantic_ai"
    _oi_class_name = "OpenInferenceSpanProcessor"

class StrandsAgentsInstrumentor(_OpenInferenceSpanProcessorWrapper):
    name = "strands-agents"
    _oi_import_path = "openinference.instrumentation.strands_agents"
    _oi_class_name = "StrandsAgentsToOpenInferenceProcessor"
