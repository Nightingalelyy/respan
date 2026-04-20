import { Instrumentation } from "@opentelemetry/instrumentation";
import { INSTRUMENTATION_INFO } from "../types/index.js";

/**
 * Dynamic instrumentation loading — silently loads available instrumentations.
 */
export const loadInstrumentation = async (
  name: string
): Promise<Instrumentation | null> => {
  const info = INSTRUMENTATION_INFO[name];
  if (!info) {
    return null;
  }

  try {
    switch (name) {
      case "openAI": {
        const { OpenAIInstrumentation } = await import(
          "@traceloop/instrumentation-openai"
        );
        return new OpenAIInstrumentation({});
      }

      case "anthropic": {
        const { AnthropicInstrumentation } = await import(
          "@traceloop/instrumentation-anthropic"
        );
        return new AnthropicInstrumentation({});
      }

      case "azureOpenAI": {
        const { AzureOpenAIInstrumentation } = await import(
          "@traceloop/instrumentation-azure"
        );
        return new AzureOpenAIInstrumentation({});
      }

      case "bedrock": {
        const { BedrockInstrumentation } = await import(
          "@traceloop/instrumentation-bedrock"
        );
        return new BedrockInstrumentation({});
      }

      case "cohere": {
        const { CohereInstrumentation } = await import(
          "@traceloop/instrumentation-cohere"
        );
        return new CohereInstrumentation({});
      }

      case "langChain": {
        const { LangChainInstrumentation } = await import(
          "@traceloop/instrumentation-langchain"
        );
        return new LangChainInstrumentation({});
      }

      case "llamaIndex": {
        const { LlamaIndexInstrumentation } = await import(
          "@traceloop/instrumentation-llamaindex"
        );
        return new LlamaIndexInstrumentation({});
      }

      case "pinecone": {
        const { PineconeInstrumentation } = await import(
          "@traceloop/instrumentation-pinecone"
        );
        return new PineconeInstrumentation({});
      }

      case "chromaDB": {
        const { ChromaDBInstrumentation } = await import(
          "@traceloop/instrumentation-chromadb"
        );
        return new ChromaDBInstrumentation({});
      }

      case "qdrant": {
        const { QdrantInstrumentation } = await import(
          "@traceloop/instrumentation-qdrant"
        );
        return new QdrantInstrumentation({});
      }

      case "together": {
        const { TogetherInstrumentation } = await import(
          "@traceloop/instrumentation-together"
        );
        return new TogetherInstrumentation({});
      }

      case "googleVertexAI": {
        const { VertexAIInstrumentation } = await import(
          "@traceloop/instrumentation-vertexai"
        );
        return new VertexAIInstrumentation({});
      }

      case "googleAIPlatform": {
        const { AIPlatformInstrumentation } = await import(
          "@traceloop/instrumentation-vertexai"
        );
        return new AIPlatformInstrumentation({});
      }

      default:
        return null;
    }
  } catch {
    return null;
  }
};
