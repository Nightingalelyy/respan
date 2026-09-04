/**
 * Configuration for the pi package entry (`dist/extension.js`, CLI mode).
 *
 * Precedence: defaults → `~/.pi/agent/respan.json` → `<cwd>/.pi/respan.json`
 * → environment. Secrets never live in the JSON files: the API key comes from
 * `RESPAN_API_KEY` or from `~/.respan/credentials.json` (written by
 * `respan auth login`).
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { isDebugEnabled, parseBooleanFlag } from "./_debug.js";
import type { PiTraceScope } from "./_otel_emitter.js";

export const DEFAULT_RESPAN_BASE_URL = "https://api.respan.ai/api";
export const PI_CONFIG_FILE_NAME = "respan.json";

export interface PiRespanConfig {
  enabled: boolean;
  apiKey?: string;
  baseURL?: string;
  workflowName?: string;
  agentName?: string;
  /** `"session"` (default: one multi-root trace per pi session) or `"run"` (one trace per agent run). */
  traceScope?: PiTraceScope;
  customerIdentifier?: string;
  metadata?: Record<string, string>;
  projectId?: string;
  debug: boolean;
  /** Which sources contributed (for diagnostics). */
  sources: string[];
}

export interface ResolvePiRespanConfigOptions {
  cwd?: string;
  env?: Record<string, string | undefined>;
  homeDir?: string;
  /** Returns file contents, or `undefined` when the file does not exist. */
  readFile?: (filePath: string) => string | undefined;
}

type RecordValue = Record<string, unknown>;

export function piGlobalConfigPath(homeDir: string = os.homedir()): string {
  return path.join(homeDir, ".pi", "agent", PI_CONFIG_FILE_NAME);
}

export function piProjectConfigPath(cwd: string = process.cwd()): string {
  return path.join(cwd, ".pi", PI_CONFIG_FILE_NAME);
}

export function normalizeBaseUrl(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  let url = value.trim().replace(/\/+$/, "");
  if (!url) {
    return undefined;
  }
  if (!url.endsWith("/api")) {
    url = `${url}/api`;
  }
  return url;
}

export function resolvePiRespanConfig(
  options: ResolvePiRespanConfigOptions = {},
): PiRespanConfig {
  const cwd = options.cwd ?? process.cwd();
  const env = options.env ?? process.env;
  const homeDir = options.homeDir ?? os.homedir();
  const readFile = options.readFile ?? defaultReadFile;

  const config: PiRespanConfig = { enabled: true, debug: false, sources: [] };
  let explicitEnabled: boolean | undefined;

  const applyFile = (filePath: string, label: string): void => {
    const record = readJsonRecord(readFile, filePath);
    if (!record) {
      return;
    }
    config.sources.push(label);
    const enabled = parseBooleanFlag(record.enabled);
    if (enabled !== undefined) {
      explicitEnabled = enabled;
    }
    const baseURL = normalizeBaseUrl(record.base_url);
    if (baseURL) {
      config.baseURL = baseURL;
    }
    const workflowName = nonEmptyString(record.workflow_name) ?? nonEmptyString(record.span_name);
    if (workflowName) {
      config.workflowName = workflowName;
    }
    const agentName = nonEmptyString(record.agent_name);
    if (agentName) {
      config.agentName = agentName;
    }
    const traceScope = parseTraceScope(record.trace_scope);
    if (traceScope) {
      config.traceScope = traceScope;
    }
    const customerId = nonEmptyString(record.customer_id);
    if (customerId) {
      config.customerIdentifier = customerId;
    }
    const projectId = nonEmptyString(record.project_id);
    if (projectId) {
      config.projectId = projectId;
    }
    if (isRecord(record.metadata)) {
      const metadata: Record<string, string> = { ...(config.metadata ?? {}) };
      for (const [key, value] of Object.entries(record.metadata)) {
        if (!key || value === undefined || value === null) {
          continue;
        }
        metadata[key] =
          typeof value === "string" ? value : typeof value === "object" ? safeJson(value) : String(value);
      }
      config.metadata = metadata;
    }
  };

  applyFile(piGlobalConfigPath(homeDir), "~/.pi/agent/respan.json");
  applyFile(piProjectConfigPath(cwd), ".pi/respan.json");

  let envUsed = false;
  const tracing = parseBooleanFlag(env.RESPAN_PI_TRACING);
  if (tracing !== undefined) {
    explicitEnabled = tracing;
    envUsed = true;
  }
  const envApiKey = nonEmptyString(env.RESPAN_API_KEY);
  if (envApiKey) {
    config.apiKey = envApiKey.trim();
    envUsed = true;
  }
  const envBaseUrl = normalizeBaseUrl(env.RESPAN_BASE_URL);
  if (envBaseUrl) {
    config.baseURL = envBaseUrl;
    envUsed = true;
  }
  const envCustomerId = nonEmptyString(env.RESPAN_CUSTOMER_ID);
  if (envCustomerId) {
    config.customerIdentifier = envCustomerId;
    envUsed = true;
  }
  const envProjectId = nonEmptyString(env.RESPAN_PROJECT_ID);
  if (envProjectId) {
    config.projectId = envProjectId;
    envUsed = true;
  }
  const envTraceScope = parseTraceScope(env.RESPAN_PI_TRACE_SCOPE);
  if (envTraceScope) {
    config.traceScope = envTraceScope;
    envUsed = true;
  }
  if (env.RESPAN_PI_DEBUG !== undefined) {
    config.debug = isDebugEnabled(env);
    envUsed = true;
  }
  if (envUsed) {
    config.sources.push("env");
  }

  if (!config.apiKey) {
    const stored = readStoredCredentials(readFile, homeDir);
    if (stored?.apiKey) {
      config.apiKey = stored.apiKey;
      config.sources.push("~/.respan/credentials.json");
      if (!config.baseURL && stored.baseUrl) {
        config.baseURL = stored.baseUrl;
      }
    }
  }

  if (config.projectId) {
    config.metadata = { ...(config.metadata ?? {}), project_id: config.projectId };
  }

  config.enabled = (explicitEnabled ?? true) && Boolean(config.apiKey);
  return config;
}

/** Accepts only `run` / `session` (case-insensitive); anything else is ignored. */
export function parseTraceScope(value: unknown): PiTraceScope | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const normalized = value.trim().toLowerCase();
  return normalized === "run" || normalized === "session" ? normalized : undefined;
}

function readStoredCredentials(
  readFile: (filePath: string) => string | undefined,
  homeDir: string,
): { apiKey?: string; baseUrl?: string } | undefined {
  const credentials = readJsonRecord(readFile, path.join(homeDir, ".respan", "credentials.json"));
  if (!credentials) {
    return undefined;
  }
  const settings = readJsonRecord(readFile, path.join(homeDir, ".respan", "config.json"));
  const profile = nonEmptyString(settings?.activeProfile) ?? "default";
  const entry = isRecord(credentials[profile]) ? credentials[profile] : undefined;
  if (!entry) {
    return undefined;
  }
  return {
    apiKey: nonEmptyString(entry.apiKey) ?? nonEmptyString(entry.accessToken),
    baseUrl: normalizeBaseUrl(entry.baseUrl),
  };
}

function readJsonRecord(
  readFile: (filePath: string) => string | undefined,
  filePath: string,
): RecordValue | undefined {
  let raw: string | undefined;
  try {
    raw = readFile(filePath);
  } catch {
    return undefined;
  }
  if (raw === undefined) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function defaultReadFile(filePath: string): string | undefined {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch (error) {
    if (isRecord(error) && error.code === "ENOENT") {
      return undefined;
    }
    return undefined;
  }
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return String(value);
  }
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function isRecord(value: unknown): value is RecordValue {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
