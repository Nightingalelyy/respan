import * as os from 'node:os';
import * as path from 'node:path';
import { DEFAULT_BASE_URL, Scope } from './integrate.js';
import { collectorBaseUrl } from './collector.js';

// Pure helpers for `respan integrate pi`. Kept free of oclif / process state so
// they can be unit-tested directly (tests/integrate-pi.test.mjs).

/** npm package that ships the Respan pi extension. */
export const PI_PACKAGE_NAME = '@respan/instrumentation-pi';

/** Package spec understood by `pi install` / `pi remove`. */
export const PI_PACKAGE_SPEC = `npm:${PI_PACKAGE_NAME}`;

/**
 * Arguments for the `pi` binary that install the extension package.
 *
 * - global (default): `pi install npm:@respan/instrumentation-pi`
 * - local:            `pi install -l npm:@respan/instrumentation-pi`
 */
export function piInstallArgs(scope: Scope): string[] {
  return scope === 'local'
    ? ['install', '-l', PI_PACKAGE_SPEC]
    : ['install', PI_PACKAGE_SPEC];
}

/**
 * Location of the non-secret config file read by the extension:
 * `~/.pi/agent/respan.json` (global) or `<projectRoot>/.pi/respan.json` (local).
 * The extension applies defaults → global → project → env, so both may coexist.
 */
export function piConfigPath(
  scope: Scope,
  projectRoot: string,
  home: string = os.homedir(),
): string {
  return scope === 'local'
    ? path.join(projectRoot, '.pi', 'respan.json')
    : path.join(home, '.pi', 'agent', 'respan.json');
}

/** `session` = one multi-root trace per pi session (default); `run` = one trace per agent run. */
export type PiTraceScope = 'run' | 'session';

export interface PiRespanConfigOptions {
  enabled: boolean;
  customerId?: string;
  workflowName?: string;
  projectId?: string;
  /** Written as `trace_scope`. */
  traceScope?: PiTraceScope;
  /** Only written when it differs from the SaaS default. */
  baseUrl?: string;
  /**
   * Port of a local `respan collector`. When set, `base_url` points at it
   * (http://127.0.0.1:<port>) instead of `baseUrl`, so traces are delivered
   * through the collector's persistent queue.
   */
  collectorPort?: number;
  /** Custom attributes, stored under `metadata` and merged with existing ones. */
  attrs?: Record<string, string>;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Build the respan.json contents for the @respan/instrumentation-pi extension.
 *
 * Keys are snake_case and match the extension's config loader:
 * `enabled`, `customer_id`, `workflow_name`, `project_id`, `trace_scope`,
 * `base_url`, `metadata`.
 * Nothing secret goes here — the API key is resolved at runtime from
 * RESPAN_API_KEY or ~/.respan/credentials.json. Existing keys are preserved so
 * re-running the command only layers the flags that were passed.
 */
export function buildPiRespanConfig(
  existing: Record<string, unknown>,
  opts: PiRespanConfigOptions,
): Record<string, unknown> {
  const config: Record<string, unknown> = { ...existing, enabled: opts.enabled };

  if (opts.customerId) {
    config.customer_id = opts.customerId;
  }
  if (opts.workflowName) {
    config.workflow_name = opts.workflowName;
  }
  if (opts.projectId) {
    config.project_id = opts.projectId;
  }
  if (opts.traceScope) {
    config.trace_scope = opts.traceScope;
  }
  if (opts.collectorPort !== undefined) {
    config.base_url = collectorBaseUrl(opts.collectorPort);
  } else if (opts.baseUrl) {
    const normalized = opts.baseUrl.replace(/\/+$/, '');
    if (normalized && normalized !== DEFAULT_BASE_URL) {
      config.base_url = normalized;
    }
  }

  const attrs = opts.attrs ?? {};
  if (Object.keys(attrs).length > 0) {
    const existingMetadata = isPlainObject(existing.metadata) ? existing.metadata : {};
    config.metadata = { ...existingMetadata, ...attrs };
  }

  return config;
}
