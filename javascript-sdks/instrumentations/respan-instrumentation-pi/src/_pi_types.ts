/**
 * Minimal structural types for the pi coding agent
 * (`@earendil-works/pi-coding-agent`).
 *
 * pi is never imported at runtime, and it is not imported in these types
 * either: the pi runtime hands its API objects to the extension factory and
 * the SDK caller hands us an `AgentSession`, so all we need is a loose,
 * structural description of the fields we read. Everything is optional or
 * `unknown` on purpose — the emitter guards every field at runtime so an
 * unexpected payload from a newer/older pi version degrades to "less
 * metadata" instead of a crash inside the pi process.
 */

export type PiRecord = Record<string, unknown>;

export interface PiTextContent {
  type: "text";
  text: string;
  [key: string]: unknown;
}

export interface PiThinkingContent {
  type: "thinking";
  thinking: string;
  [key: string]: unknown;
}

export interface PiImageContent {
  type: "image";
  data?: string;
  mimeType?: string;
  [key: string]: unknown;
}

export interface PiToolCallContent {
  type: "toolCall";
  id: string;
  name: string;
  arguments?: Record<string, unknown>;
  [key: string]: unknown;
}

export type PiContentBlock =
  | PiTextContent
  | PiThinkingContent
  | PiImageContent
  | PiToolCallContent
  | PiRecord;

export interface PiUsageCost {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  total?: number;
}

/**
 * pi usage semantics (Anthropic style): `input` is the NON-cached input token
 * count, so total prompt tokens = input + cacheRead + cacheWrite.
 * `cacheWrite1h` is a subset of `cacheWrite` and `reasoning` is a subset of
 * `output`; neither is added again.
 */
export interface PiUsage {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  cacheWrite1h?: number;
  reasoning?: number;
  totalTokens?: number;
  cost?: PiUsageCost;
}

export interface PiUserMessage {
  role: "user";
  content: string | PiContentBlock[];
  timestamp?: number;
}

export interface PiAssistantMessage {
  role: "assistant";
  content: PiContentBlock[];
  api?: string;
  provider?: string;
  model?: string;
  responseModel?: string;
  responseId?: string;
  usage?: PiUsage;
  stopReason?: string;
  errorMessage?: string;
  timestamp?: number;
}

export interface PiToolResultMessage {
  role: "toolResult";
  toolCallId?: string;
  toolName?: string;
  content?: PiContentBlock[];
  details?: unknown;
  isError?: boolean;
  timestamp?: number;
}

/** Custom app messages (`bashExecution`, extension messages, ...). */
export interface PiCustomMessage {
  role: string;
  content?: string | PiContentBlock[];
  customType?: string;
  [key: string]: unknown;
}

export type PiAgentMessage =
  | PiUserMessage
  | PiAssistantMessage
  | PiToolResultMessage
  | PiCustomMessage;

export interface PiModelLike {
  id?: string;
  provider?: string;
  name?: string;
  api?: string;
}

export interface PiToolDefinitionLike {
  name: string;
  description?: string;
  parameters?: unknown;
}

export interface PiAgentToolResultLike {
  content?: PiContentBlock[];
  details?: unknown;
  [key: string]: unknown;
}

export interface PiAssistantMessageEventLike {
  type: string;
  [key: string]: unknown;
}

export interface PiExtensionUILike {
  setStatus?(key: string, text: string | undefined): void;
  notify?(message: string, type?: string): void;
}

export interface PiSessionManagerLike {
  getSessionId?(): string;
  getSessionFile?(): string | undefined;
  getCwd?(): string;
}

/** Subset of pi's `ExtensionContext` that the instrumentation reads. */
export interface PiExtensionContextLike {
  cwd?: string;
  hasUI?: boolean;
  mode?: string;
  ui?: PiExtensionUILike;
  model?: PiModelLike;
  thinkingLevel?: string;
  sessionManager?: PiSessionManagerLike;
}

export type PiExtensionHandler = (
  event: any,
  ctx: PiExtensionContextLike,
) => unknown;

/** Subset of pi's `ExtensionAPI` that the instrumentation uses. */
export interface PiExtensionAPI {
  on(event: string, handler: PiExtensionHandler): void;
  getAllTools?(): PiToolDefinitionLike[];
  getActiveTools?(): string[];
}

/** A pi extension factory: `export default (pi) => { pi.on(...) }`. */
export type PiExtensionFactory = (pi: PiExtensionAPI) => void;

/** Subset of pi's `AgentSession` that `PiInstrumentor.attach()` reads. */
export interface PiAgentSessionLike {
  subscribe(listener: (event: any) => void): () => void;
  readonly messages?: PiAgentMessage[] | unknown[];
  readonly model?: PiModelLike;
  readonly sessionId?: string;
  readonly sessionFile?: string;
  readonly thinkingLevel?: string;
  readonly sessionManager?: PiSessionManagerLike;
  readonly agent?: unknown;
  getAllTools?(): PiToolDefinitionLike[];
  getActiveToolNames?(): string[];
}
