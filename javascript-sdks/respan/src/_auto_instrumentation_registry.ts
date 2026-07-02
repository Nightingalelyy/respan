export type AutoInstrumentationCategory =
  | "direct-llm"
  | "agent-framework"
  | "app-framework"
  | "protocol-or-tooling"
  | "vector-db";

export type InstrumentationStatus =
  | "enabled"
  | "disabled"
  | "missing"
  | "failed";

export interface AutoInstrumentationEntry {
  id: string;
  category: AutoInstrumentationCategory;
  provider?: string;
  sdkPackage: string;
  instrumentationPackage: string;
  instrumentorClass: string;
  enabledByDefault: boolean;
  priority: number;
  aliases?: string[];
  conflictsWith?: string[];
  genericTracingNames?: string[];
  docsUrl?: string;
}

export interface InstrumentationStatusEntry {
  id: string;
  category: AutoInstrumentationCategory;
  provider?: string;
  sdkPackage: string;
  instrumentationPackage: string;
  instrumentorClass: string;
  status: InstrumentationStatus;
  reason?: string;
}

export const DIRECT_LLM_AUTO_INSTRUMENTATIONS: AutoInstrumentationEntry[] = [
  {
    id: "openai",
    category: "direct-llm",
    provider: "openai",
    sdkPackage: "openai",
    instrumentationPackage: "@respan/instrumentation-openai",
    instrumentorClass: "OpenAIInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["openAI", "OpenAIInstrumentor"],
    conflictsWith: ["openai-agents"],
    genericTracingNames: ["openAI"],
    docsUrl: "https://respan.ai/docs/integrations/openai-sdk",
  },
  {
    id: "anthropic",
    category: "direct-llm",
    provider: "anthropic",
    sdkPackage: "@anthropic-ai/sdk",
    instrumentationPackage: "@respan/instrumentation-anthropic",
    instrumentorClass: "AnthropicInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["AnthropicInstrumentor"],
    conflictsWith: ["claude-agent-sdk"],
    genericTracingNames: ["anthropic"],
    docsUrl: "https://respan.ai/docs/integrations/anthropic",
  },
  {
    id: "azure-openai",
    category: "direct-llm",
    provider: "azure-openai",
    sdkPackage: "openai",
    instrumentationPackage: "@respan/instrumentation-azure-openai",
    instrumentorClass: "AzureOpenAIInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["azureOpenAI", "AzureOpenAIInstrumentor"],
    genericTracingNames: ["azureOpenAI"],
    docsUrl: "https://respan.ai/docs/integrations/providers/azure",
  },
  {
    id: "vertexai",
    category: "direct-llm",
    provider: "google",
    sdkPackage: "@google-cloud/vertexai",
    instrumentationPackage: "@respan/instrumentation-vertexai",
    instrumentorClass: "VertexAIInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["googleVertexAI", "vertex-ai", "VertexAIInstrumentor"],
    genericTracingNames: ["googleVertexAI"],
    docsUrl: "https://respan.ai/docs/integrations/vertex-ai",
  },
  {
    id: "openrouter",
    category: "direct-llm",
    provider: "openrouter",
    sdkPackage: "@openrouter/sdk",
    instrumentationPackage: "@respan/instrumentation-openrouter",
    instrumentorClass: "OpenRouterInstrumentor",
    enabledByDefault: true,
    priority: 100,
    aliases: ["OpenRouterInstrumentor"],
    docsUrl: "https://respan.ai/docs/integrations/openrouter",
  },
];

export function directLlmGenericTracingNames(): string[] {
  return DIRECT_LLM_AUTO_INSTRUMENTATIONS.flatMap(
    (entry) => entry.genericTracingNames ?? [],
  );
}

export function matchesAutoInstrumentationSelector(
  entry: AutoInstrumentationEntry,
  selector: string,
): boolean {
  const normalized = selector.toLowerCase();
  return [
    entry.id,
    entry.provider,
    entry.sdkPackage,
    entry.instrumentationPackage,
    entry.instrumentorClass,
    ...(entry.aliases ?? []),
  ]
    .filter((value): value is string => Boolean(value))
    .some((value) => value.toLowerCase() === normalized);
}

export function statusFromEntry(
  entry: AutoInstrumentationEntry,
  status: InstrumentationStatus,
  reason?: string,
): InstrumentationStatusEntry {
  return {
    id: entry.id,
    category: entry.category,
    provider: entry.provider,
    sdkPackage: entry.sdkPackage,
    instrumentationPackage: entry.instrumentationPackage,
    instrumentorClass: entry.instrumentorClass,
    status,
    reason,
  };
}
