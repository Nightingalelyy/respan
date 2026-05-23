import type { McpClientMethod, OperationConfig } from "./_constants.js";

const MAX_ATTRIBUTE_CHARS = 16000;
const MAX_SERIALIZATION_DEPTH = 6;

export function toSerializableValue(value: unknown, depth = 0): unknown {
  if (depth > MAX_SERIALIZATION_DEPTH) {
    return String(value);
  }

  if (value === undefined || value === null) {
    return value;
  }

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

  if (typeof value === "function" || typeof value === "symbol") {
    return undefined;
  }

  if (value instanceof Date) {
    return value.toISOString();
  }

  if (value instanceof Error) {
    return {
      error: value.name,
      message: value.message,
    };
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => toSerializableValue(item, depth + 1))
      .filter((item) => item !== undefined);
  }

  if (typeof value === "object") {
    const maybeToJSON = (value as { toJSON?: () => unknown }).toJSON;
    if (typeof maybeToJSON === "function") {
      try {
        return toSerializableValue(maybeToJSON.call(value), depth + 1);
      } catch {
        // Continue with structural serialization below.
      }
    }

    const normalized: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (key.startsWith("_")) {
        continue;
      }
      const serialized = toSerializableValue(item, depth + 1);
      if (serialized !== undefined) {
        normalized[key] = serialized;
      }
    }
    return normalized;
  }

  return String(value);
}

export function safeJson(value: unknown): string {
  let serialized: string | undefined;
  try {
    serialized = JSON.stringify(toSerializableValue(value), (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    serialized = String(value);
  }

  if (typeof serialized !== "string") {
    serialized = String(serialized);
  }

  if (serialized.length <= MAX_ATTRIBUTE_CHARS) {
    return serialized;
  }
  return `${serialized.slice(0, MAX_ATTRIBUTE_CHARS)}...`;
}

function firstArg(args: unknown[]): unknown {
  return args.length > 0 ? args[0] : undefined;
}

function firstObjectArg(args: unknown[]): Record<string, unknown> {
  const arg = firstArg(args);
  if (!arg || typeof arg !== "object" || Array.isArray(arg)) {
    return {};
  }
  return arg as Record<string, unknown>;
}

function transportName(transport: unknown): string | undefined {
  if (!transport || typeof transport !== "object") {
    return undefined;
  }
  return (transport as { constructor?: { name?: string } }).constructor?.name;
}

export function resolveClientEntityName(
  methodName: McpClientMethod,
  args: unknown[],
  config: OperationConfig,
): string {
  const params = firstObjectArg(args);

  if (methodName === "callTool" || methodName === "getPrompt") {
    const name = params.name;
    return typeof name === "string" && name ? name : config.entityName;
  }

  if (methodName === "readResource") {
    const uri = params.uri;
    return typeof uri === "string" && uri ? uri : config.entityName;
  }

  return config.entityName;
}

export function resolveClientSpanName(
  methodName: McpClientMethod,
  entityName: string,
  config: OperationConfig,
): string {
  if (methodName === "callTool") {
    return `mcp.tool.${entityName}`;
  }
  if (methodName === "getPrompt") {
    return `mcp.prompt.${entityName}`;
  }
  return config.spanName;
}

export function clientMethodInput(
  methodName: McpClientMethod,
  args: unknown[],
): Record<string, unknown> {
  if (methodName === "connect") {
    return {
      method: methodName,
      transport: transportName(firstArg(args)),
    };
  }

  const params = firstObjectArg(args);

  if (methodName === "callTool") {
    return {
      name: params.name,
      arguments: params.arguments,
    };
  }

  if (methodName === "readResource") {
    return {
      uri: params.uri,
    };
  }

  if (methodName === "getPrompt") {
    return {
      name: params.name,
      arguments: params.arguments,
    };
  }

  return {
    method: methodName,
    params,
  };
}

export function errorOutput(error: unknown): Record<string, string> {
  if (error instanceof Error) {
    return {
      error: error.name,
      message: error.message,
    };
  }
  return {
    error: "Error",
    message: String(error),
  };
}

export function findLastFunctionIndex(args: unknown[]): number {
  for (let index = args.length - 1; index >= 0; index -= 1) {
    if (typeof args[index] === "function") {
      return index;
    }
  }
  return -1;
}
