import { types } from "node:util";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return String(value);
  }
}

export function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function toSerializableValue(
  value: any,
  seen = new WeakSet<object>(),
): any {
  if (
    value === null ||
    value === undefined ||
    typeof value === "string" ||
    typeof value === "boolean" ||
    typeof value === "number"
  )
    return value;
  if (typeof value === "bigint") return value.toString();
  if (typeof value !== "object" || types.isProxy(value)) return undefined;
  if (typeof Response !== "undefined" && value instanceof Response)
    return summarizeResponse(value);
  if (typeof Blob !== "undefined" && value instanceof Blob)
    return { size: value.size, type: value.type };
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  const result: any = Array.isArray(value) ? [] : {};
  for (const key of Object.keys(value)) {
    const d = Object.getOwnPropertyDescriptor(value, key);
    if (d && "value" in d && typeof d.value !== "function")
      result[key] = toSerializableValue(d.value, seen);
  }
  seen.delete(value);
  return result;
}

export function stringifyStructured(value: unknown): string {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) return "";
  if (typeof serialized === "string") return serialized;
  return safeJson(serialized);
}

export function summarizeResponse(response: Response): Record<string, unknown> {
  return {
    status: response.status,
    statusText: response.statusText,
    contentType: response.headers.get("content-type") ?? undefined,
    contentLength: response.headers.get("content-length") ?? undefined,
  };
}

export function summarizeRequestBody(
  body: Record<string, any> | undefined,
): Record<string, any> {
  if (!body || typeof body !== "object") return {};

  const result: Record<string, any> = {};
  for (const [key, value] of Object.entries(body)) {
    if (key === "file" && typeof value !== "string") {
      result[key] = "[uploadable]";
      continue;
    }
    result[key] = toSerializableValue(value);
  }
  return result;
}

function normalizeMessage(message: Record<string, any>): Record<string, any> {
  return message;
}
function normalizeToolCall(toolCall: Record<string, any>): Record<string, any> {
  return toolCall;
}
function normalizeFunctionTool(tool: Record<string, any>): Record<string, any> {
  return tool;
}

export function normalizeContent(content: unknown): any {
  return content === undefined ? "" : toSerializableValue(content);
}

export function normalizeChatToolCall(
  toolCall: any,
): Record<string, any> | null {
  if (!isRecord(toolCall)) return null;

  const normalized = normalizeToolCall({
    ...(typeof toolCall.id === "string" ? { id: toolCall.id } : {}),
    type: toolCall.type ?? "function",
    function: {
      name: toolCall.function?.name ?? toolCall.name ?? "",
      arguments:
        typeof toolCall.function?.arguments === "string"
          ? toolCall.function.arguments
          : safeJson(toolCall.function?.arguments ?? toolCall.arguments ?? {}),
    },
  });

  if (typeof toolCall.index === "number") normalized.index = toolCall.index;
  return normalized;
}

export function formatInputMessages(
  messages: any[] | undefined,
): Record<string, any>[] {
  if (!Array.isArray(messages)) return [];

  const result: Record<string, any>[] = [];
  for (const message of messages) {
    if (!isRecord(message)) continue;

    const normalizedMessage = normalizeMessage({
      role: message.role ?? "user",
      content: normalizeContent(message.content),
    });

    const toolCalls = extractToolCallsFromMessage(message);
    if (toolCalls.length > 0) {
      normalizedMessage.tool_calls = toolCalls;
    }
    if (typeof message.tool_call_id === "string") {
      normalizedMessage.tool_call_id = message.tool_call_id;
    }
    if (message.name) {
      normalizedMessage.name = String(message.name);
    }

    result.push(normalizedMessage);
  }
  return result;
}

export function formatTextPrompt(prompt: unknown): Record<string, any>[] {
  return [
    normalizeMessage({ role: "user", content: normalizeContent(prompt) }),
  ];
}

export function extractToolCallsFromMessage(
  message: Record<string, any> | undefined,
): Record<string, any>[] {
  if (!message) return [];
  const toolCalls: Record<string, any>[] = [];

  if (Array.isArray(message.tool_calls)) {
    for (const toolCall of message.tool_calls) {
      const normalized = normalizeChatToolCall(toolCall);
      if (normalized) toolCalls.push(normalized);
    }
  }

  if (isRecord(message.function_call)) {
    const normalized = normalizeChatToolCall({
      id: "",
      type: "function",
      function: message.function_call,
    });
    if (normalized) toolCalls.push(normalized);
  }

  return toolCalls;
}

export function formatChatOutputMessage(response: any): Record<string, any> {
  const choice = Array.isArray(response?.choices)
    ? response.choices[0]
    : undefined;
  const message = choice?.message ?? {};
  const outputMessage = normalizeMessage({
    role: message.role ?? "assistant",
    content: normalizeContent(message.content ?? choice?.text ?? ""),
  });
  const toolCalls = extractToolCallsFromMessage(message);
  if (toolCalls.length > 0) outputMessage.tool_calls = toolCalls;
  return outputMessage;
}

export function formatTextCompletion(response: any): string {
  if (!Array.isArray(response?.choices)) return "";
  return response.choices
    .map((choice: any) => choice?.text ?? choice?.message?.content ?? "")
    .filter(Boolean)
    .join("\n");
}

export function formatTools(tools: any[] | undefined): Record<string, any>[] {
  if (!Array.isArray(tools)) return [];
  return tools.filter(isRecord).map((tool) => {
    const fn = isRecord(tool.function) ? tool.function : tool;
    return { type: "function", function: toSerializableValue(fn) };
  });
}

export function applyTokenUsage(
  attrs: Record<string, any>,
  usage: any,
  keys: {
    inputTokens: string;
    outputTokens: string;
    promptTokens: string;
    completionTokens: string;
    totalTokens: string;
  },
): void {
  if (!usage || typeof usage !== "object") return;

  const promptTokens = numericValue(usage.prompt_tokens ?? usage.input_tokens);
  const completionTokens = numericValue(
    usage.completion_tokens ?? usage.output_tokens,
  );
  const totalTokens = numericValue(usage.total_tokens);

  if (promptTokens !== undefined) {
    attrs[keys.inputTokens] = promptTokens;
    attrs[keys.promptTokens] = promptTokens;
  }
  if (completionTokens !== undefined) {
    attrs[keys.outputTokens] = completionTokens;
    attrs[keys.completionTokens] = completionTokens;
  }
  if (totalTokens !== undefined) {
    attrs[keys.totalTokens] = totalTokens;
  }
}

function numericValue(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

export function resolveModel(request: any, response?: any): string | undefined {
  return stringValue(response?.model ?? request?.model);
}

export function resolveErrorMessage(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

export function resolveStatusCode(err: unknown): number | undefined {
  if (!isRecord(err)) return undefined;
  const status = err.status ?? err.statusCode ?? err.code;
  if (typeof status === "number" && Number.isFinite(status)) return status;
  if (typeof status === "string") {
    const parsed = Number(status);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function stringValue(value: unknown): string | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  return String(value);
}

function findPackageDirectory(resolvedEntry: string): string | null {
  let currentDir = dirname(resolvedEntry);

  while (true) {
    if (existsSync(join(currentDir, "package.json"))) return currentDir;
    const parentDir = dirname(currentDir);
    if (parentDir === currentDir) return null;
    currentDir = parentDir;
  }
}

function addTogetherModuleCandidates(
  urls: Set<string>,
  resolverBase: string | URL,
): void {
  try {
    const require = createRequire(resolverBase);
    const resolvedEntry = require.resolve("together-ai");
    const packageDir = findPackageDirectory(resolvedEntry);
    if (!packageDir) return;

    for (const entryFile of ["index.mjs", "index.js"]) {
      const entryPath = join(packageDir, entryFile);
      if (existsSync(entryPath)) urls.add(pathToFileURL(entryPath).href);
    }
  } catch {
    // Ignore this candidate and keep trying other resolution bases.
  }
}

export async function loadTogetherConstructors(): Promise<any[]> {
  const candidateUrls = new Set<string>();
  const runtimeResolutionBases = [
    join(process.cwd(), "__respan_runtime__.js"),
    process.env.INIT_CWD
      ? join(process.env.INIT_CWD, "__respan_init__.js")
      : null,
    process.argv[1] ?? null,
    import.meta.url,
  ].filter(Boolean) as Array<string | URL>;

  for (const resolutionBase of runtimeResolutionBases) {
    addTogetherModuleCandidates(candidateUrls, resolutionBase);
  }

  const constructors: any[] = [];
  for (const moduleUrl of candidateUrls) {
    try {
      const importedModule = await import(moduleUrl);
      const Together =
        importedModule?.default ?? importedModule?.Together ?? importedModule;
      if (typeof Together === "function" && !constructors.includes(Together)) {
        constructors.push(Together);
      }
    } catch {
      // Ignore candidate import failures so one bad resolution path does not block activation.
    }
  }

  return constructors;
}
