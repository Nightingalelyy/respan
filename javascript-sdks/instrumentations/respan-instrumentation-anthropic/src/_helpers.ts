import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import {
  FunctionToolSchema,
  MessageSchema,
  ToolCallSchema,
} from "@respan/respan-sdk";

export const PACKAGE_VERSION = "1.1.1";
export const INSTRUMENTATION_LIBRARY_NAME = "@respan/instrumentation-anthropic";
export const ANTHROPIC_CHAT_ENTITY_NAME = "anthropic.chat";
export const STREAM_INSTRUMENTED = Symbol("respan.anthropic.stream.instrumented");
export const TOOL_USE_JSON_BUFFER_KEY = Symbol("respan.anthropic.tool.use.json.buffer");

export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return String(value);
  }
}

export function toSerializableValue(value: any): any {
  if (value === null) return null;
  if (value === undefined) return undefined;
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (value instanceof Date) {
    return value.toISOString();
  }

  try {
    return JSON.parse(
      JSON.stringify(value, (_key, innerValue) =>
        typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
      ),
    );
  } catch {
    // Fall back to recursive normalization below.
  }

  if (Array.isArray(value)) {
    return value.map((item) => toSerializableValue(item));
  }
  if (typeof value === "object") {
    if (typeof value.toJSON === "function") {
      try {
        return toSerializableValue(value.toJSON());
      } catch {
        // Ignore and continue to the structural copy below.
      }
    }

    const normalized: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, itemValue]) => {
      normalized[key] = toSerializableValue(itemValue);
    });
    return normalized;
  }

  return String(value);
}

export function stringifyStructured(value: unknown): string {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  return safeJson(serialized);
}

export interface ToolExecution {
  id: string;
  name: string;
  input: unknown;
  output: unknown;
  isError: boolean;
}

function normalizeMessage(message: Record<string, any>): Record<string, any> {
  const parsed = MessageSchema.safeParse(message);
  return parsed.success ? parsed.data : message;
}

function normalizeToolCall(toolCall: Record<string, any>): Record<string, any> {
  const parsed = ToolCallSchema.safeParse(toolCall);
  return parsed.success ? parsed.data : toolCall;
}

function normalizeFunctionTool(tool: Record<string, any>): Record<string, any> {
  const parsed = FunctionToolSchema.safeParse(tool);
  return parsed.success ? parsed.data : tool;
}

export function normalizeContentBlock(block: any): string {
  if (typeof block === "string") return block;
  if (block && typeof block === "object") {
    if (typeof block.text === "string") return block.text;
    if (block.type === "image") return "[image]";
  }
  return "";
}

export function normalizeToolCallBlock(block: any): Record<string, any> | null {
  if (!block || typeof block !== "object" || block.type !== "tool_use") {
    return null;
  }

  return normalizeToolCall({
    id: block.id ?? "",
    type: "function",
    function: {
      name: block.name ?? "",
      arguments: safeJson(block.input ?? {}),
    },
  });
}

export function normalizeToolResultBlock(block: any): Record<string, any> | null {
  if (!block || typeof block !== "object" || block.type !== "tool_result") {
    return null;
  }

  const normalized = normalizeMessage({
    role: "tool",
    content: stringifyStructured(block.content ?? ""),
  });
  if (block.tool_use_id) normalized.tool_call_id = block.tool_use_id;
  if (block.is_error === true) normalized.is_error = true;
  return normalized;
}

export function formatInputMessages(messages: any[], system?: any): any[] {
  const result: any[] = [];

  if (system != null) {
    if (typeof system === "string") {
      result.push(normalizeMessage({ role: "system", content: system }));
    } else if (Array.isArray(system)) {
      const parts = system
        .map((block: any) => {
          if (typeof block === "string") return block;
          if (block && typeof block.text === "string") return block.text;
          return String(block);
        })
        .filter(Boolean);
      result.push(normalizeMessage({ role: "system", content: parts.join("\n") }));
    }
  }

  for (const message of messages) {
    const role = message?.role ?? "user";
    const content = message?.content ?? "";

    if (!Array.isArray(content)) {
      result.push({ role, content: toSerializableValue(content) });
      continue;
    }

    const textParts: string[] = [];
    const toolCalls: Record<string, any>[] = [];
    const toolResults: Record<string, any>[] = [];

    for (const block of content) {
      const text = normalizeContentBlock(block);
      if (text) textParts.push(text);

      const toolCall = normalizeToolCallBlock(block);
      if (toolCall) {
        toolCalls.push(toolCall);
        continue;
      }

      const toolResult = normalizeToolResultBlock(block);
      if (toolResult) {
        toolResults.push(toolResult);
      }
    }

    if (textParts.length > 0 || toolCalls.length > 0) {
      const normalizedMessage = normalizeMessage({
        role,
        content: textParts.join("\n"),
      });
      if (toolCalls.length > 0) {
        normalizedMessage.tool_calls = toolCalls;
      }
      result.push(normalizedMessage);
    }

    result.push(...toolResults);
  }

  return result;
}

export function formatOutput(message: any): string {
  const content = message?.content;
  if (!content) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  const parts: string[] = [];
  for (const block of content) {
    const text = normalizeContentBlock(block);
    if (text) parts.push(text);
  }
  return parts.join("\n");
}

export function extractToolCalls(message: any): any[] | null {
  const content = message?.content;
  if (!Array.isArray(content)) return null;

  const toolCalls: any[] = [];
  for (const block of content) {
    const toolCall = normalizeToolCallBlock(block);
    if (toolCall) toolCalls.push(toolCall);
  }

  return toolCalls.length ? toolCalls : null;
}

export function formatOutputMessage(message: any): Record<string, any> {
  const outputMessage = normalizeMessage({
    role: "assistant",
    content: formatOutput(message),
  });

  const toolCalls = extractToolCalls(message);
  if (toolCalls) {
    outputMessage.tool_calls = toolCalls;
  }

  return outputMessage;
}

export function extractToolCallsFromInputMessages(messages: any[] | undefined): any[] | null {
  if (!Array.isArray(messages)) return null;

  const toolCalls: any[] = [];
  for (const message of messages) {
    if (Array.isArray(message?.tool_calls)) {
      for (const toolCall of message.tool_calls) {
        toolCalls.push(normalizeToolCall(toolCall));
      }
    }

    if (!Array.isArray(message?.content)) continue;
    for (const block of message.content) {
      const toolCall = normalizeToolCallBlock(block);
      if (toolCall) toolCalls.push(toolCall);
    }
  }

  return toolCalls.length ? toolCalls : null;
}

export function mergeToolCalls(...groups: Array<any[] | null | undefined>): any[] | null {
  const merged: any[] = [];
  const seen = new Set<string>();

  for (const group of groups) {
    for (const toolCall of group ?? []) {
      const key = safeJson(toolCall);
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(toolCall);
    }
  }

  return merged.length ? merged : null;
}

export function extractToolExecutions(messages: any[] | undefined): ToolExecution[] {
  if (!Array.isArray(messages)) return [];

  const toolUses = new Map<string, { name: string; input: unknown }>();
  for (const message of messages) {
    if (!Array.isArray(message?.content)) continue;
    for (const block of message.content) {
      const toolCall = normalizeToolCallBlock(block);
      if (!toolCall) continue;
      toolUses.set(toolCall.id, {
        name: toolCall.function?.name ?? "tool",
        input: block.input ?? {},
      });
    }
  }

  const executions: ToolExecution[] = [];
  for (const message of messages) {
    if (!Array.isArray(message?.content)) continue;
    for (const block of message.content) {
      if (!block || typeof block !== "object" || block.type !== "tool_result") continue;
      const toolUseId = block.tool_use_id ?? "";
      const toolUse = toolUses.get(toolUseId);
      executions.push({
        id: toolUseId,
        name: toolUse?.name ?? "tool",
        input: toolUse?.input ?? {},
        output: block.content ?? "",
        isError: block.is_error === true,
      });
    }
  }

  return executions;
}

export function formatTools(tools: any[] | undefined): any[] | null {
  if (!tools || !tools.length) return null;

  const result: any[] = [];
  for (const tool of tools) {
    const entry = normalizeFunctionTool({
      type: "function",
      name: tool.name ?? "",
      ...(tool.description ? { description: tool.description } : {}),
      ...(tool.input_schema ? { parameters: tool.input_schema } : {}),
    });
    result.push(entry);
  }

  return result.length ? toSerializableValue(result) : null;
}

function findPackageDirectory(resolvedEntry: string): string | null {
  let currentDir = dirname(resolvedEntry);

  while (true) {
    if (existsSync(join(currentDir, "package.json"))) {
      return currentDir;
    }

    const parentDir = dirname(currentDir);
    if (parentDir === currentDir) {
      return null;
    }
    currentDir = parentDir;
  }
}

function addAnthropicModuleCandidates(urls: Set<string>, resolverBase: string | URL): void {
  try {
    const require = createRequire(resolverBase);
    const resolvedEntry = require.resolve("@anthropic-ai/sdk");
    const packageDir = findPackageDirectory(resolvedEntry);

    if (!packageDir) return;

    for (const entryFile of ["index.mjs", "index.js"]) {
      const entryPath = join(packageDir, entryFile);
      if (existsSync(entryPath)) {
        urls.add(pathToFileURL(entryPath).href);
      }
    }
  } catch {
    // Ignore resolution failures for this candidate.
  }
}

export async function loadAnthropicConstructors(): Promise<any[]> {
  const candidateUrls = new Set<string>();
  const runtimeResolutionBases = [
    join(process.cwd(), "__respan_runtime__.js"),
    process.env.INIT_CWD ? join(process.env.INIT_CWD, "__respan_init__.js") : null,
    process.argv[1] ?? null,
    import.meta.url,
  ].filter(Boolean) as Array<string | URL>;

  for (const resolutionBase of runtimeResolutionBases) {
    addAnthropicModuleCandidates(candidateUrls, resolutionBase);
  }

  const constructors: any[] = [];
  for (const moduleUrl of candidateUrls) {
    try {
      const importedModule = await import(moduleUrl);
      const Anthropic = importedModule?.default ?? importedModule;
      if (typeof Anthropic === "function" && !constructors.includes(Anthropic)) {
        constructors.push(Anthropic);
      }
    } catch {
      // Ignore candidate import failures so we can keep trying others.
    }
  }

  return constructors;
}
