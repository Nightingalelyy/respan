import { Instrumentation } from "@opentelemetry/instrumentation";
import { InstrumentationName, RespanOptions } from "../types/clientTypes.js";
import { loadInstrumentation } from "./loader.js";
import { 
  InstrumentationLoadResult, 
  InstrumentationConfig, 
  ManualInstrumentationConfig,
  INSTRUMENTATION_INFO 
} from "../types/index.js";

// Array to hold all instrumentations that will be loaded dynamically
const instrumentations: Instrumentation[] = [];

// Store instrumentation instances for configuration
let openAIInstrumentation: any | undefined;
let anthropicInstrumentation: any | undefined;
let azureOpenAIInstrumentation: any | undefined;
let cohereInstrumentation: any | undefined;
let vertexaiInstrumentation: any | undefined;
let bedrockInstrumentation: any | undefined;
let langchainInstrumentation: any | undefined;
let llamaIndexInstrumentation: any | undefined;
let pineconeInstrumentation: any | undefined;
let chromadbInstrumentation: any | undefined;
let qdrantInstrumentation: any | undefined;
let togetherInstrumentation: any | undefined;

/**
 * Get all loaded instrumentations
 */
export const getInstrumentations = (): Instrumentation[] => {
  return [...instrumentations];
};

/**
 * Get a specific instrumentation instance by name
 */
export const getInstrumentationInstance = (name: InstrumentationName): any => {
  switch (name) {
    case "openAI": return openAIInstrumentation;
    case "anthropic": return anthropicInstrumentation;
    case "azureOpenAI": return azureOpenAIInstrumentation;
    case "cohere": return cohereInstrumentation;
    case "googleVertexAI": return vertexaiInstrumentation;
    case "bedrock": return bedrockInstrumentation;
    case "langChain": return langchainInstrumentation;
    case "llamaIndex": return llamaIndexInstrumentation;
    case "pinecone": return pineconeInstrumentation;
    case "chromaDB": return chromadbInstrumentation;
    case "qdrant": return qdrantInstrumentation;
    case "together": return togetherInstrumentation;
    default: return undefined;
  }
};

/**
 * Configure trace content for instrumentations
 */
export const configureTraceContent = (enabled: boolean): void => {
  const traceContent = enabled;
  
  openAIInstrumentation?.setConfig?.({ traceContent });
  anthropicInstrumentation?.setConfig?.({ traceContent });
  azureOpenAIInstrumentation?.setConfig?.({ traceContent });
  llamaIndexInstrumentation?.setConfig?.({ traceContent });
  vertexaiInstrumentation?.setConfig?.({ traceContent });
  bedrockInstrumentation?.setConfig?.({ traceContent });
  cohereInstrumentation?.setConfig?.({ traceContent });
  chromadbInstrumentation?.setConfig?.({ traceContent });
  togetherInstrumentation?.setConfig?.({ traceContent });
};

/**
 * Initialize all available instrumentations automatically.
 * This is used when no specific instrumentModules are provided.
 */
export const initInstrumentations = async (
  disabledInstrumentations: InstrumentationName[] = [],
): Promise<void> => {
  const exceptionLogger = (e: Error) =>
    console.error("Instrumentation error:", e);

  // Clear the instrumentations array
  instrumentations.length = 0;

  // Define all instrumentations to attempt loading
  const instrumentationsToLoad: InstrumentationConfig[] = [
    {
      name: "openAI",
      description: "OpenAI API instrumentation",
      loadFunction: async () => {
        const { OpenAIInstrumentation } = await import(
          "@traceloop/instrumentation-openai"
        );
        openAIInstrumentation = new OpenAIInstrumentation({
          exceptionLogger: (e: Error) =>
            console.error("OpenAI instrumentation error:", e),
        });
        return openAIInstrumentation;
      },
    },
    {
      name: "anthropic",
      description: "Anthropic API instrumentation",
      loadFunction: async () => {
        const { AnthropicInstrumentation } = await import(
          "@traceloop/instrumentation-anthropic"
        );
        anthropicInstrumentation = new AnthropicInstrumentation({
          exceptionLogger,
        });
        return anthropicInstrumentation;
      },
    },
    {
      name: "azureOpenAI",
      description: "Azure OpenAI instrumentation",
      loadFunction: async () => {
        const { AzureOpenAIInstrumentation } = await import(
          "@traceloop/instrumentation-azure"
        );
        azureOpenAIInstrumentation = new AzureOpenAIInstrumentation({
          exceptionLogger,
        });
        return azureOpenAIInstrumentation;
      },
    },
    {
      name: "cohere",
      description: "Cohere API instrumentation",
      loadFunction: async () => {
        const { CohereInstrumentation } = await import(
          "@traceloop/instrumentation-cohere"
        );
        cohereInstrumentation = new CohereInstrumentation({ exceptionLogger });
        return cohereInstrumentation;
      },
    },
    {
      name: "googleVertexAI",
      description: "Google Vertex AI instrumentation",
      loadFunction: async () => {
        const { VertexAIInstrumentation } = await import(
          "@traceloop/instrumentation-vertexai"
        );
        vertexaiInstrumentation = new VertexAIInstrumentation({
          exceptionLogger,
        });
        return vertexaiInstrumentation;
      },
    },
    {
      name: "bedrock",
      description: "AWS Bedrock instrumentation",
      loadFunction: async () => {
        const { BedrockInstrumentation } = await import(
          "@traceloop/instrumentation-bedrock"
        );
        bedrockInstrumentation = new BedrockInstrumentation({
          exceptionLogger,
        });
        return bedrockInstrumentation;
      },
    },
    {
      name: "langChain",
      description: "LangChain framework instrumentation",
      loadFunction: async () => {
        const { LangChainInstrumentation } = await import(
          "@traceloop/instrumentation-langchain"
        );
        langchainInstrumentation = new LangChainInstrumentation({
          exceptionLogger,
        });
        return langchainInstrumentation;
      },
    },
    {
      name: "llamaIndex",
      description: "LlamaIndex framework instrumentation",
      loadFunction: async () => {
        const { LlamaIndexInstrumentation } = await import(
          "@traceloop/instrumentation-llamaindex"
        );
        llamaIndexInstrumentation = new LlamaIndexInstrumentation({
          exceptionLogger,
        });
        return llamaIndexInstrumentation;
      },
    },
    {
      name: "pinecone",
      description: "Pinecone vector database instrumentation",
      loadFunction: async () => {
        const { PineconeInstrumentation } = await import(
          "@traceloop/instrumentation-pinecone"
        );
        pineconeInstrumentation = new PineconeInstrumentation({
          exceptionLogger,
        });
        return pineconeInstrumentation;
      },
    },
    {
      name: "chromaDB",
      description: "ChromaDB vector database instrumentation",
      loadFunction: async () => {
        const { ChromaDBInstrumentation } = await import(
          "@traceloop/instrumentation-chromadb"
        );
        chromadbInstrumentation = new ChromaDBInstrumentation({
          exceptionLogger,
        });
        return chromadbInstrumentation;
      },
    },
    {
      name: "qdrant",
      description: "Qdrant vector database instrumentation",
      loadFunction: async () => {
        const { QdrantInstrumentation } = await import(
          "@traceloop/instrumentation-qdrant"
        );
        qdrantInstrumentation = new QdrantInstrumentation({ exceptionLogger });
        return qdrantInstrumentation;
      },
    },
    {
      name: "together",
      description: "Together AI instrumentation",
      loadFunction: async () => {
        const { TogetherInstrumentation } = await import(
          "@traceloop/instrumentation-together"
        );
        togetherInstrumentation = new TogetherInstrumentation({
          exceptionLogger,
        });
        return togetherInstrumentation;
      },
    },
  ];

  // Load each instrumentation
  for (const { name, description, loadFunction } of instrumentationsToLoad) {
    if (disabledInstrumentations.includes(name)) {
      continue;
    }

    try {
      const instrumentation = await loadFunction();
      instrumentations.push(instrumentation);
    } catch {
      // Package not installed — skip silently
    }
  }
};

/**
 * Manually initialize instrumentations with provided modules.
 * This is similar to Traceloop's approach for environments like Next.js
 * where dynamic imports might not work properly.
 */
export const manuallyInitInstrumentations = async (
  instrumentModules: NonNullable<RespanOptions["instrumentModules"]>,
  disabledInstrumentations: InstrumentationName[] = []
): Promise<void> => {
  const exceptionLogger = (e: Error) =>
    console.error("Instrumentation error:", e);


  // Track instrumentation loading results (using string for name to allow custom modules)
  // Clear the instrumentations array
  instrumentations.length = 0;

  // Define all possible manual instrumentations
  const manualInstrumentationConfigs: ManualInstrumentationConfig[] = [
    {
      name: "openAI",
      description: "OpenAI API instrumentation",
      moduleKey: "openAI",
      initFunction: async (module: any) => {
        const { OpenAIInstrumentation } = await import(
          "@traceloop/instrumentation-openai"
        );
        openAIInstrumentation = new OpenAIInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(openAIInstrumentation);
        openAIInstrumentation.manuallyInstrument(module);
        return openAIInstrumentation;
      },
    },
    {
      name: "anthropic",
      description: "Anthropic API instrumentation",
      moduleKey: "anthropic",
      initFunction: async (module: any) => {
        const { AnthropicInstrumentation } = await import(
          "@traceloop/instrumentation-anthropic"
        );
        anthropicInstrumentation = new AnthropicInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(anthropicInstrumentation);
        anthropicInstrumentation.manuallyInstrument(module);
        return anthropicInstrumentation;
      },
    },
    {
      name: "azureOpenAI",
      description: "Azure OpenAI instrumentation",
      moduleKey: "azureOpenAI",
      initFunction: async (module: any) => {
        const { AzureOpenAIInstrumentation } = await import(
          "@traceloop/instrumentation-azure"
        );
        azureOpenAIInstrumentation = new AzureOpenAIInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(azureOpenAIInstrumentation);
        azureOpenAIInstrumentation.manuallyInstrument(module);
        return azureOpenAIInstrumentation;
      },
    },
    {
      name: "cohere",
      description: "Cohere API instrumentation",
      moduleKey: "cohere",
      initFunction: async (module: any) => {
        const { CohereInstrumentation } = await import(
          "@traceloop/instrumentation-cohere"
        );
        cohereInstrumentation = new CohereInstrumentation({ exceptionLogger });
        instrumentations.push(cohereInstrumentation);
        cohereInstrumentation.manuallyInstrument(module);
        return cohereInstrumentation;
      },
    },
    {
      name: "googleVertexAI",
      description: "Google Vertex AI instrumentation",
      moduleKey: "googleVertexAI",
      initFunction: async (module: any) => {
        const { VertexAIInstrumentation } = await import(
          "@traceloop/instrumentation-vertexai"
        );
        vertexaiInstrumentation = new VertexAIInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(vertexaiInstrumentation);
        vertexaiInstrumentation.manuallyInstrument(module);
        return vertexaiInstrumentation;
      },
    },
    {
      name: "googleAIPlatform",
      description: "Google AI Platform instrumentation",
      moduleKey: "googleAIPlatform",
      initFunction: async (module: any) => {
        const { AIPlatformInstrumentation } = await import(
          "@traceloop/instrumentation-vertexai"
        );
        const aiplatformInstrumentation = new AIPlatformInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(aiplatformInstrumentation);
        aiplatformInstrumentation.manuallyInstrument(module);
        return aiplatformInstrumentation;
      },
    },
    {
      name: "bedrock",
      description: "AWS Bedrock instrumentation",
      moduleKey: "bedrock",
      initFunction: async (module: any) => {
        const { BedrockInstrumentation } = await import(
          "@traceloop/instrumentation-bedrock"
        );
        bedrockInstrumentation = new BedrockInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(bedrockInstrumentation);
        bedrockInstrumentation.manuallyInstrument(module);
        return bedrockInstrumentation;
      },
    },
    {
      name: "pinecone",
      description: "Pinecone vector database instrumentation",
      moduleKey: "pinecone",
      initFunction: async (module: any) => {
        const { PineconeInstrumentation } = await import(
          "@traceloop/instrumentation-pinecone"
        );
        pineconeInstrumentation = new PineconeInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(pineconeInstrumentation);
        pineconeInstrumentation.manuallyInstrument(module);
        return pineconeInstrumentation;
      },
    },
    {
      name: "langChain",
      description: "LangChain framework instrumentation",
      moduleKey: "langChain",
      initFunction: async (module: any) => {
        const { LangChainInstrumentation } = await import(
          "@traceloop/instrumentation-langchain"
        );
        langchainInstrumentation = new LangChainInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(langchainInstrumentation);
        langchainInstrumentation.manuallyInstrument(module);
        return langchainInstrumentation;
      },
    },
    {
      name: "llamaIndex",
      description: "LlamaIndex framework instrumentation",
      moduleKey: "llamaIndex",
      initFunction: async (module: any) => {
        const { LlamaIndexInstrumentation } = await import(
          "@traceloop/instrumentation-llamaindex"
        );
        llamaIndexInstrumentation = new LlamaIndexInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(llamaIndexInstrumentation);
        llamaIndexInstrumentation.manuallyInstrument(module);
        return llamaIndexInstrumentation;
      },
    },
    {
      name: "chromaDB",
      description: "ChromaDB vector database instrumentation",
      moduleKey: "chromaDB",
      initFunction: async (module: any) => {
        const { ChromaDBInstrumentation } = await import(
          "@traceloop/instrumentation-chromadb"
        );
        chromadbInstrumentation = new ChromaDBInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(chromadbInstrumentation);
        chromadbInstrumentation.manuallyInstrument(module);
        return chromadbInstrumentation;
      },
    },
    {
      name: "qdrant",
      description: "Qdrant vector database instrumentation",
      moduleKey: "qdrant",
      initFunction: async (module: any) => {
        const { QdrantInstrumentation } = await import(
          "@traceloop/instrumentation-qdrant"
        );
        qdrantInstrumentation = new QdrantInstrumentation({ exceptionLogger });
        instrumentations.push(qdrantInstrumentation);
        qdrantInstrumentation.manuallyInstrument(module);
        return qdrantInstrumentation;
      },
    },
    {
      name: "together",
      description: "Together AI instrumentation",
      moduleKey: "together",
      initFunction: async (module: any) => {
        const { TogetherInstrumentation } = await import(
          "@traceloop/instrumentation-together"
        );
        togetherInstrumentation = new TogetherInstrumentation({
          exceptionLogger,
        });
        instrumentations.push(togetherInstrumentation);
        togetherInstrumentation.manuallyInstrument(module);
        return togetherInstrumentation;
      },
    },
  ];

  // Keep track of processed module keys
  const processedModuleKeys = new Set<string>();

  // Process each pre-defined instrumentation
  for (const {
    name,
    description,
    moduleKey,
    initFunction,
  } of manualInstrumentationConfigs) {
    const module = instrumentModules[moduleKey as keyof typeof instrumentModules];
    processedModuleKeys.add(moduleKey);

    if (disabledInstrumentations.includes(name)) {
      continue;
    }

    if (!module) {
      continue;
    }

    try {
      await initFunction(module);
    } catch {
      // failed to init — skip silently
    }
  }

  // Process any additional modules not in our pre-defined list
  for (const [moduleKey, module] of Object.entries(instrumentModules)) {
    if (processedModuleKeys.has(moduleKey) || !module) {
      continue; // Skip already processed or null modules
    }

    const customName = moduleKey;
    const customDescription = `Custom ${moduleKey} instrumentation`;

    try {
      if (typeof module.manuallyInstrument === "function") {
        module.manuallyInstrument(module);
      } else if (
        typeof module.setTracerProvider === "function" &&
        typeof module.getConfig === "function"
      ) {
        instrumentations.push(module);
      }
    } catch {
      // failed to init custom module — skip silently
    }
  }

};

/**
 * Add an instrumentation to the collection
 */
export const enableInstrumentation = async (name: string): Promise<void> => {
  const instrumentation = await loadInstrumentation(name);
  if (instrumentation) {
    instrumentations.push(instrumentation);
  }
}; 