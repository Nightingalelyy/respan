/**
 * Respan instrumentation plugin for the Model Context Protocol TypeScript SDK.
 *
 * Patches common @modelcontextprotocol/sdk client methods and server tool
 * callbacks to emit canonical Respan OTEL spans.
 */

import { SpanStatusCode, trace } from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  CLIENT_OPERATION_CONFIG,
  MCP_INSTRUMENTATION_NAME,
  MCP_INSTRUMENTATION_PACKAGE,
  MCP_LOG_METHOD_TS_TRACING,
  MCP_METHOD_ATTRIBUTE,
  MCP_OPERATION_ATTRIBUTE,
  MCP_TRANSPORT_ATTRIBUTE,
  SERVER_TOOL_OPERATION_CONFIG,
  type McpClientMethod,
  type OperationConfig,
} from "./_constants.js";
import {
  clientMethodInput,
  errorOutput,
  findLastFunctionIndex,
  resolveClientEntityName,
  resolveClientSpanName,
  safeJson,
  toSerializableValue,
} from "./_utils.js";

const PACKAGE_VERSION = "1.0.0";
const WRAPPED_HANDLER = Symbol.for("respan.instrumentation.mcp.wrappedHandler");

type AnyFunction = (...args: unknown[]) => unknown;
type PatchablePrototype = Record<string, unknown>;

export interface MCPClientModule {
  Client?: { prototype?: object };
}

export interface MCPServerModule {
  McpServer?: { prototype?: object };
}

export interface MCPInstrumentorOptions {
  clientModule?: MCPClientModule;
  serverModule?: MCPServerModule;
  captureClientOperations?: boolean;
  captureServerTools?: boolean;
}

interface TraceOperationOptions {
  config: OperationConfig;
  entityName: string;
  input: unknown;
  methodName: string;
  operation: string;
  run: () => unknown;
  spanName: string;
  transportName?: string;
}

function traceOperation<T>(opts: TraceOperationOptions): T {
  const tracer = trace.getTracer(MCP_INSTRUMENTATION_PACKAGE, PACKAGE_VERSION);
  const attributes: Record<string, string> = {
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: MCP_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: opts.config.logType,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: opts.entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: opts.entityName,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJson(opts.input),
    [MCP_METHOD_ATTRIBUTE]: opts.methodName,
    [MCP_OPERATION_ATTRIBUTE]: opts.operation,
  };

  if (opts.transportName) {
    attributes[MCP_TRANSPORT_ATTRIBUTE] = opts.transportName;
  }

  return tracer.startActiveSpan(opts.spanName, { attributes }, (span) => {
    const finishSuccess = (result: unknown): unknown => {
      span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        safeJson(result ?? null),
      );
      span.setStatus({ code: SpanStatusCode.OK });
      span.end();
      return result;
    };

    const finishError = (error: unknown): never => {
      if (error instanceof Error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      } else {
        span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      }
      span.setAttribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        safeJson(errorOutput(error)),
      );
      span.end();
      throw error;
    };

    try {
      const result = opts.run();
      if (result && typeof (result as Promise<unknown>).then === "function") {
        return (result as Promise<unknown>).then(finishSuccess, finishError) as T;
      }
      return finishSuccess(result) as T;
    } catch (error) {
      return finishError(error);
    }
  }) as T;
}

function getConstructorName(value: unknown): string | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  return (value as { constructor?: { name?: string } }).constructor?.name;
}

function traceClientMethod(
  methodName: McpClientMethod,
  original: AnyFunction,
  instance: unknown,
  args: unknown[],
): unknown {
  const config = CLIENT_OPERATION_CONFIG[methodName];
  const entityName = resolveClientEntityName(methodName, args, config);

  return traceOperation({
    config,
    entityName,
    input: clientMethodInput(methodName, args),
    methodName,
    operation: config.operation,
    run: () => original.apply(instance, args),
    spanName: resolveClientSpanName(methodName, entityName, config),
    transportName: methodName === "connect" ? getConstructorName(args[0]) : undefined,
  });
}

function wrapToolHandler(toolName: string, handler: AnyFunction): AnyFunction {
  if ((handler as unknown as Record<symbol, unknown>)[WRAPPED_HANDLER]) {
    return handler;
  }

  const wrapped = function wrappedMcpToolHandler(
    this: unknown,
    ...args: unknown[]
  ): unknown {
    return traceOperation({
      config: SERVER_TOOL_OPERATION_CONFIG,
      entityName: toolName || SERVER_TOOL_OPERATION_CONFIG.entityName,
      input: {
        name: toolName,
        arguments: args[0],
      },
      methodName: "tool",
      operation: SERVER_TOOL_OPERATION_CONFIG.operation,
      run: () => handler.apply(this, args),
      spanName: `mcp.tool.${toolName || "handler"}`,
    });
  };

  Object.defineProperty(wrapped, WRAPPED_HANDLER, {
    enumerable: false,
    value: true,
  });
  return wrapped;
}

export class MCPInstrumentor {
  public readonly name = MCP_INSTRUMENTATION_NAME;

  private static _clientOriginals: Map<string, AnyFunction> = new Map();
  private static _serverOriginals: Map<string, AnyFunction> = new Map();
  private static _clientPatchCount = 0;
  private static _serverPatchCount = 0;

  private readonly _clientModule?: MCPClientModule;
  private readonly _serverModule?: MCPServerModule;
  private readonly _captureClientOperations: boolean;
  private readonly _captureServerTools: boolean;
  private _clientPrototype?: PatchablePrototype;
  private _serverPrototype?: PatchablePrototype;
  private _clientPatched = false;
  private _serverPatched = false;
  private _isInstrumented = false;

  constructor(options: MCPInstrumentorOptions = {}) {
    this._clientModule = options.clientModule;
    this._serverModule = options.serverModule;
    this._captureClientOperations = options.captureClientOperations ?? true;
    this._captureServerTools = options.captureServerTools ?? true;
  }

  async activate(): Promise<void> {
    if (this._isInstrumented) {
      return;
    }

    const [clientModule, serverModule] = await Promise.all([
      this._resolveClientModule(),
      this._resolveServerModule(),
    ]);

    if (this._captureClientOperations && clientModule?.Client?.prototype) {
      this._patchClientPrototype(clientModule.Client.prototype);
    }

    if (this._captureServerTools && serverModule?.McpServer?.prototype) {
      this._patchServerPrototype(serverModule.McpServer.prototype);
    }

    this._isInstrumented = this._clientPatched || this._serverPatched;
  }

  deactivate(): void {
    if (this._clientPatched) {
      this._restoreClientPrototype();
    }
    if (this._serverPatched) {
      this._restoreServerPrototype();
    }
    this._isInstrumented = false;
  }

  isActive(): boolean {
    return this._isInstrumented;
  }

  private async _resolveClientModule(): Promise<MCPClientModule | undefined> {
    if (this._clientModule) {
      return this._clientModule;
    }

    try {
      return (await import("@modelcontextprotocol/sdk/client/index.js")) as unknown as MCPClientModule;
    } catch {
      return undefined;
    }
  }

  private async _resolveServerModule(): Promise<MCPServerModule | undefined> {
    if (this._serverModule) {
      return this._serverModule;
    }

    try {
      return (await import("@modelcontextprotocol/sdk/server/mcp.js")) as unknown as MCPServerModule;
    } catch {
      return undefined;
    }
  }

  private _patchClientPrototype(prototype: object): void {
    const patchablePrototype = prototype as PatchablePrototype;
    this._clientPrototype = patchablePrototype;
    if (MCPInstrumentor._clientPatchCount === 0) {
      for (const methodName of Object.keys(CLIENT_OPERATION_CONFIG) as McpClientMethod[]) {
        const original = patchablePrototype[methodName];
        if (typeof original !== "function") {
          continue;
        }

        MCPInstrumentor._clientOriginals.set(methodName, original as AnyFunction);
        patchablePrototype[methodName] = function instrumentedMcpClientMethod(
          this: unknown,
          ...args: unknown[]
        ): unknown {
          return traceClientMethod(
            methodName,
            original as AnyFunction,
            this,
            args,
          );
        };
      }
    }

    if (MCPInstrumentor._clientOriginals.size > 0) {
      MCPInstrumentor._clientPatchCount += 1;
      this._clientPatched = true;
    }
  }

  private _patchServerPrototype(prototype: object): void {
    const patchablePrototype = prototype as PatchablePrototype;
    this._serverPrototype = patchablePrototype;
    if (MCPInstrumentor._serverPatchCount === 0) {
      const registerTool = patchablePrototype.registerTool;
      if (typeof registerTool === "function") {
        MCPInstrumentor._serverOriginals.set("registerTool", registerTool as AnyFunction);
        patchablePrototype.registerTool = function instrumentedRegisterTool(
          this: unknown,
          name: unknown,
          config: unknown,
          cb: unknown,
        ): unknown {
          const handler =
            typeof cb === "function" ? wrapToolHandler(String(name), cb as AnyFunction) : cb;
          return (registerTool as AnyFunction).call(this, name, config, handler);
        };
      }

      const tool = patchablePrototype.tool;
      if (typeof tool === "function") {
        MCPInstrumentor._serverOriginals.set("tool", tool as AnyFunction);
        patchablePrototype.tool = function instrumentedTool(
          this: unknown,
          name: unknown,
          ...args: unknown[]
        ): unknown {
          const callbackIndex = findLastFunctionIndex(args);
          const patchedArgs = [...args];
          if (callbackIndex >= 0) {
            patchedArgs[callbackIndex] = wrapToolHandler(
              String(name),
              patchedArgs[callbackIndex] as AnyFunction,
            );
          }
          return (tool as AnyFunction).call(this, name, ...patchedArgs);
        };
      }
    }

    if (MCPInstrumentor._serverOriginals.size > 0) {
      MCPInstrumentor._serverPatchCount += 1;
      this._serverPatched = true;
    }
  }

  private _restoreClientPrototype(): void {
    MCPInstrumentor._clientPatchCount = Math.max(
      0,
      MCPInstrumentor._clientPatchCount - 1,
    );

    if (MCPInstrumentor._clientPatchCount === 0 && this._clientPrototype) {
      for (const [methodName, original] of MCPInstrumentor._clientOriginals) {
        this._clientPrototype[methodName] = original;
      }
      MCPInstrumentor._clientOriginals.clear();
    }

    this._clientPatched = false;
  }

  private _restoreServerPrototype(): void {
    MCPInstrumentor._serverPatchCount = Math.max(
      0,
      MCPInstrumentor._serverPatchCount - 1,
    );

    if (MCPInstrumentor._serverPatchCount === 0 && this._serverPrototype) {
      for (const [methodName, original] of MCPInstrumentor._serverOriginals) {
        this._serverPrototype[methodName] = original;
      }
      MCPInstrumentor._serverOriginals.clear();
    }

    this._serverPatched = false;
  }
}

export {
  clientMethodInput,
  resolveClientEntityName,
  resolveClientSpanName,
  safeJson,
  toSerializableValue,
};
