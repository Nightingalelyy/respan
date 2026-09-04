import * as fs from 'node:fs';
import * as path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import type { ReadableStream as WebReadableStream } from 'node:stream/web';
import { Flags } from '@oclif/core';
import { DEFAULT_BASE_URL, resolveAuth } from './auth.js';
import { ensureDir } from './integrate.js';
import {
  COLLECTOR_VERSION,
  CONTAINER_NAME,
  DEFAULT_COLLECTOR_PORT,
  DEFAULT_HEALTH_PORT,
  DEFAULT_METRICS_PORT,
  DEFAULT_QUEUE_SIZE,
  VOLUME_NAME,
  binaryDownloadUrl,
  checksumUrl,
  exportUrlFromBaseUrl,
  parsePrometheusText,
  parseSha256,
  summarizeCollectorMetrics,
  type CollectorMetricsSummary,
  type CollectorPaths,
} from './collector.js';

// Side-effectful helpers for `respan collector ...` (docker, processes, HTTP,
// downloads). The pure counterparts live in lib/collector.ts.

export type CollectorMode = 'docker' | 'binary';

/** Smallest --max-size-mib otelcol-contrib 0.160.0 accepts with the config's compaction settings. */
export const MIN_MAX_SIZE_MIB = 100;

// ── Shared flags ──────────────────────────────────────────────────────────

export const collectorSizingFlags = {
  'export-url': Flags.string({
    description: 'Respan ingest URL (default: <base URL>/api/v2/traces from your auth config)',
    env: 'RESPAN_COLLECTOR_EXPORT_URL',
  }),
  'data-dir': Flags.string({
    description: 'Persistent queue directory, binary mode only (default: ~/.respan/collector/data)',
  }),
  'max-size-mib': Flags.integer({
    // file_storage rejects a max_size below its compaction rebound threshold (100 MiB by default).
    description: 'Disk cap for the persistent queue in MiB (minimum 100)',
    default: 512,
    min: MIN_MAX_SIZE_MIB,
  }),
  'queue-size': Flags.integer({
    description: 'Batches kept while Respan is unreachable',
    default: DEFAULT_QUEUE_SIZE,
    min: 1,
  }),
  port: Flags.integer({
    description: 'Local OTLP/HTTP port the SDKs send to',
    default: DEFAULT_COLLECTOR_PORT,
    min: 1,
    max: 65535,
  }),
};

export const collectorModeFlags = {
  docker: Flags.boolean({
    description: 'Use the Docker container (default when Docker is available)',
    default: false,
    exclusive: ['binary'],
  }),
  binary: Flags.boolean({
    description: 'Use the otelcol-contrib binary in ~/.respan/collector/bin',
    default: false,
    exclusive: ['docker'],
  }),
};

/**
 * Ingest URL for the collector: --export-url / RESPAN_COLLECTOR_EXPORT_URL,
 * else derived from the resolved auth base URL. Falls back to the configured
 * base URL (flag / RESPAN_API_BASE_URL / SaaS default) when not logged in, so
 * `collector config --print` works without credentials.
 */
export function resolveExportUrl(flags: {
  'export-url'?: string;
  'api-key'?: string;
  'base-url'?: string;
  profile?: string;
}): string {
  if (flags['export-url']) return flags['export-url'];
  let baseUrl: string;
  try {
    baseUrl = resolveAuth(flags).baseUrl;
  } catch {
    baseUrl = flags['base-url'] || process.env.RESPAN_API_BASE_URL || DEFAULT_BASE_URL;
  }
  return exportUrlFromBaseUrl(baseUrl);
}

// ── Docker ────────────────────────────────────────────────────────────────

export interface CommandResult {
  status: number | null;
  stdout: string;
  stderr: string;
}

/** Run `docker <args>` to completion. Never rejects; a missing binary is a failed result. */
export function runDocker(args: string[], env: NodeJS.ProcessEnv = process.env): Promise<CommandResult> {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';
    const child = spawn('docker', args, { env, stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', (err) => resolve({ status: null, stdout, stderr: stderr || err.message }));
    child.on('close', (status) => resolve({ status, stdout, stderr }));
  });
}

function dockerSync(args: string[]): { ok: boolean; stdout: string } {
  const result = spawnSync('docker', args, {
    encoding: 'utf-8',
    stdio: ['ignore', 'pipe', 'pipe'],
    timeout: 15000,
  });
  return { ok: !result.error && result.status === 0, stdout: result.stdout ?? '' };
}

/** True when the Docker CLI is installed and can reach a daemon. */
export function dockerAvailable(): boolean {
  return dockerSync(['info']).ok;
}

export interface ContainerState {
  exists: boolean;
  running: boolean;
  /** created | running | paused | restarting | removing | exited | dead */
  status?: string;
  startedAt?: string;
  restartCount?: number;
  image?: string;
}

export function dockerContainerState(name: string = CONTAINER_NAME): ContainerState {
  const result = dockerSync([
    'inspect', '--format', '{{.State.Status}}|{{.State.StartedAt}}|{{.RestartCount}}|{{.Config.Image}}', name,
  ]);
  if (!result.ok) return { exists: false, running: false };
  const [status, startedAt, restartCount, image] = result.stdout.trim().split('|');
  return {
    exists: true,
    running: status === 'running',
    status,
    startedAt,
    restartCount: Number.parseInt(restartCount ?? '', 10) || 0,
    image,
  };
}

export function dockerVolumeExists(name: string = VOLUME_NAME): boolean {
  return dockerSync(['volume', 'inspect', name]).ok;
}

// ── Processes ─────────────────────────────────────────────────────────────

export function readPid(pidPath: string): number | undefined {
  try {
    const pid = Number.parseInt(fs.readFileSync(pidPath, 'utf-8').trim(), 10);
    return Number.isInteger(pid) && pid > 0 ? pid : undefined;
  } catch {
    return undefined;
  }
}

export function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM';
  }
}

export interface RunningCollector {
  mode: CollectorMode;
  pid?: number;
  container?: ContainerState;
}

/**
 * What this CLI started, if anything: the container (in any state — an exited
 * container still owns the name) or a live pid from the pid file. A stale pid
 * file is removed. `mode` restricts the check to one runtime.
 */
export function detectRunning(paths: CollectorPaths, mode?: CollectorMode): RunningCollector | undefined {
  if (mode !== 'binary') {
    const container = dockerContainerState();
    if (container.exists) return { mode: 'docker', container };
  }
  if (mode !== 'docker') {
    const pid = readPid(paths.pidPath);
    if (pid !== undefined) {
      if (isPidAlive(pid)) return { mode: 'binary', pid };
      fs.rmSync(paths.pidPath, { force: true });
    }
  }
  return undefined;
}

export function spawnCollectorBinary(paths: CollectorPaths, env: NodeJS.ProcessEnv): number {
  ensureDir(paths.root);
  const logFd = fs.openSync(paths.logPath, 'a');
  try {
    const child = spawn(paths.binaryPath, ['--config', paths.configPath], {
      detached: true,
      stdio: ['ignore', logFd, logFd],
      env,
    });
    // Spawn failures surface asynchronously; the caller notices through the
    // pid liveness check instead of an unhandled 'error' event.
    child.on('error', () => {});
    if (child.pid === undefined) {
      throw new Error(`Failed to start ${paths.binaryPath}`);
    }
    child.unref();
    fs.writeFileSync(paths.pidPath, `${child.pid}\n`);
    return child.pid;
  } finally {
    fs.closeSync(logFd);
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** SIGTERM, wait for exit, SIGKILL as a last resort. */
export async function stopPid(pid: number, timeoutMs = 10000): Promise<'stopped' | 'killed' | 'not-running'> {
  if (!isPidAlive(pid)) return 'not-running';
  try {
    process.kill(pid, 'SIGTERM');
  } catch {
    return 'not-running';
  }
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isPidAlive(pid)) return 'stopped';
    await sleep(250);
  }
  try {
    process.kill(pid, 'SIGKILL');
  } catch {
    // already gone
  }
  return 'killed';
}

export function tailFile(filePath: string, lines: number): string {
  try {
    return fs.readFileSync(filePath, 'utf-8').trimEnd().split('\n').slice(-lines).join('\n');
  } catch {
    return '';
  }
}

export function directorySize(dir: string): number {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return 0;
  }
  let total = 0;
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      total += directorySize(full);
    } else if (entry.isFile()) {
      try {
        total += fs.statSync(full).size;
      } catch {
        // removed while walking
      }
    }
  }
  return total;
}

// ── Health & metrics ──────────────────────────────────────────────────────

export function healthUrl(port: number = DEFAULT_HEALTH_PORT): string {
  return `http://127.0.0.1:${port}/`;
}

export function metricsUrl(port: number = DEFAULT_METRICS_PORT): string {
  return `http://127.0.0.1:${port}/metrics`;
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export interface HealthStatus {
  healthy: boolean;
  /** health_check body status, e.g. "Server available", or "HTTP 503". */
  status?: string;
  uptime?: string;
  upSince?: string;
  error?: string;
}

/** GET the health_check extension (200 once the pipeline is ready). */
export async function checkHealth(port: number = DEFAULT_HEALTH_PORT): Promise<HealthStatus> {
  try {
    const response = await fetchWithTimeout(healthUrl(port), 2000);
    const text = await response.text();
    let body: Record<string, unknown> = {};
    try {
      body = JSON.parse(text) as Record<string, unknown>;
    } catch {
      // not JSON
    }
    return {
      healthy: response.ok,
      status: typeof body.status === 'string' ? body.status : `HTTP ${response.status}`,
      uptime: typeof body.uptime === 'string' ? body.uptime : undefined,
      upSince: typeof body.upSince === 'string' ? body.upSince : undefined,
    };
  } catch (err) {
    const cause = (err as { cause?: { code?: string } }).cause;
    return { healthy: false, error: cause?.code ?? (err instanceof Error ? err.message : String(err)) };
  }
}

/** Poll health until it passes, the timeout elapses, or `isAlive` reports the process gone. */
export async function waitForHealth(
  port: number,
  timeoutMs: number,
  isAlive?: () => boolean,
): Promise<HealthStatus> {
  const deadline = Date.now() + timeoutMs;
  let last: HealthStatus = { healthy: false };
  while (Date.now() < deadline) {
    last = await checkHealth(port);
    if (last.healthy) return last;
    if (isAlive && !isAlive()) return { ...last, error: 'process exited' };
    await sleep(500);
  }
  return last;
}

export async function fetchMetrics(port: number = DEFAULT_METRICS_PORT): Promise<CollectorMetricsSummary | undefined> {
  try {
    const response = await fetchWithTimeout(metricsUrl(port), 2000);
    if (!response.ok) return undefined;
    return summarizeCollectorMetrics(parsePrometheusText(await response.text()));
  } catch {
    return undefined;
  }
}

// ── Binary download ───────────────────────────────────────────────────────

export async function downloadFile(url: string, dest: string): Promise<void> {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(`Download failed (${response.status} ${response.statusText}): ${url}`);
  }
  ensureDir(path.dirname(dest));
  await pipeline(
    Readable.fromWeb(response.body as unknown as WebReadableStream),
    fs.createWriteStream(dest),
  );
}

export async function sha256File(filePath: string): Promise<string> {
  const hash = createHash('sha256');
  await pipeline(fs.createReadStream(filePath), hash);
  return hash.digest('hex');
}

export interface CollectorLogger {
  log(message: string): void;
  warn(message: string): void;
}

/**
 * Make sure ~/.respan/collector/bin/otelcol-contrib exists: download the
 * release tarball, verify its SHA-256 against the published `.sha256` asset
 * (warn when that cannot be fetched), extract the binary and drop the tarball.
 */
export async function ensureCollectorBinary(paths: CollectorPaths, logger: CollectorLogger): Promise<string> {
  if (fs.existsSync(paths.binaryPath)) return paths.binaryPath;

  const url = binaryDownloadUrl(COLLECTOR_VERSION);
  const tarball = path.join(paths.binDir, `otelcol-contrib_${COLLECTOR_VERSION}.tar.gz`);
  ensureDir(paths.binDir);
  logger.log(`Downloading otelcol-contrib ${COLLECTOR_VERSION} (about 95 MB)...`);
  logger.log(`  ${url}`);
  await downloadFile(url, tarball);

  let expected: string | undefined;
  try {
    const response = await fetchWithTimeout(checksumUrl(COLLECTOR_VERSION), 15000);
    if (response.ok) expected = parseSha256(await response.text());
  } catch {
    // reported below
  }
  if (expected) {
    const actual = await sha256File(tarball);
    if (actual !== expected) {
      fs.rmSync(tarball, { force: true });
      throw new Error(
        `SHA-256 mismatch for ${url}: expected ${expected}, got ${actual}. The download was discarded.`,
      );
    }
    logger.log('Verified SHA-256 checksum.');
  } else {
    logger.warn('Could not fetch the release checksum; the download was NOT verified.');
  }

  const extract = spawnSync('tar', ['-xzf', tarball, '-C', paths.binDir, 'otelcol-contrib'], {
    encoding: 'utf-8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (extract.error || extract.status !== 0) {
    throw new Error(`Failed to extract ${tarball}: ${(extract.stderr || extract.error?.message || '').trim()}`);
  }
  fs.chmodSync(paths.binaryPath, 0o755);
  fs.rmSync(tarball, { force: true });
  logger.log(`Installed ${paths.binaryPath}`);
  return paths.binaryPath;
}
