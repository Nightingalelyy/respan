import * as os from 'node:os';
import * as path from 'node:path';

// Pure helpers for `respan collector ...`. Kept free of process / network side
// effects so they can be unit-tested directly (tests/collector.test.mjs); the
// commands and lib/collector-runtime.ts do the spawning and polling.

/** otelcol-contrib release the config is validated against (collector/config.yaml). */
export const COLLECTOR_VERSION = '0.160.0';
/** Upstream contrib image used by `respan collector start --docker`. */
export const COLLECTOR_IMAGE = `otel/opentelemetry-collector-contrib:${COLLECTOR_VERSION}`;
/** OTLP/HTTP receiver port — what the SDK's RESPAN_BASE_URL points at. */
export const DEFAULT_COLLECTOR_PORT = 4318;
/** health_check extension port (fixed in the config unless overridden by env). */
export const DEFAULT_HEALTH_PORT = 13133;
/** Prometheus metrics port (hard-coded in the config). */
export const DEFAULT_METRICS_PORT = 8888;
/** file_storage max_size default: 512 MiB. */
export const DEFAULT_MAX_SIZE_BYTES = 536870912;
/** sending_queue.queue_size default (batches). */
export const DEFAULT_QUEUE_SIZE = 10000;
export const CONTAINER_NAME = 'respan-collector';
export const VOLUME_NAME = 'respan-collector-queue';
/** Queue directory inside the container (matches collector/Dockerfile). */
export const CONTAINER_DATA_DIR = '/var/lib/otelcol/respan';
/** Config mount point inside the container (the upstream image's default path). */
export const CONTAINER_CONFIG_PATH = '/etc/otelcol-contrib/config.yaml';
/** Uid the upstream contrib image runs as; the queue volume must be writable by it. */
export const CONTAINER_UID = '10001';
/** Helper image used once to chown a fresh queue volume (the contrib image has no shell). */
export const VOLUME_INIT_IMAGE = 'alpine:3.20';

/**
 * The collector configuration — an exact copy of collector/config.yaml (the
 * single source of truth; tests/collector.test.mjs asserts they never drift).
 * Every tunable is a `${env:NAME:-default}` placeholder; the API key is only
 * ever read from the RESPAN_API_KEY environment variable.
 */
export const COLLECTOR_CONFIG_TEMPLATE = `# Respan Collector — an OpenTelemetry Collector configuration.
#
# Receives OTLP/HTTP from the Respan SDKs on localhost and forwards it to the
# Respan ingest endpoint through a bounded, persistent (write-ahead-log) queue,
# so spans survive Respan outages and collector restarts without unbounded
# local growth. Every value can be overridden with an environment variable.
#
# Validated against otelcol-contrib 0.160.0.

extensions:
  file_storage:
    # Persistent queue. Entries are deleted as soon as Respan acknowledges them;
    # in steady state the directory is nearly empty.
    directory: \${env:RESPAN_COLLECTOR_DATA_DIR:-/var/lib/otelcol/respan}
    create_directory: true
    # Hard cap on disk (bytes). When reached, new batches are dropped and counted
    # in otelcol_exporter_enqueue_failed_spans instead of filling the disk.
    max_size: \${env:RESPAN_COLLECTOR_MAX_SIZE_BYTES:-536870912}
    timeout: 1s
    compaction:
      on_start: true
      on_rebound: true
      directory: \${env:RESPAN_COLLECTOR_DATA_DIR:-/var/lib/otelcol/respan}
  health_check:
    endpoint: \${env:RESPAN_COLLECTOR_HEALTH_ENDPOINT:-127.0.0.1:13133}

receivers:
  otlp:
    protocols:
      http:
        endpoint: \${env:RESPAN_COLLECTOR_LISTEN:-127.0.0.1:4318}
        # The Respan SDKs post to <RESPAN_BASE_URL>/api/v2/traces.
        traces_url_path: /api/v2/traces

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: \${env:RESPAN_COLLECTOR_MEMORY_LIMIT_MIB:-512}
    spike_limit_mib: 128
  batch:
    send_batch_size: 512
    send_batch_max_size: 1024
    timeout: 2s

exporters:
  otlphttp/respan:
    traces_endpoint: \${env:RESPAN_COLLECTOR_EXPORT_URL:-https://api.respan.ai/api/v2/traces}
    encoding: json
    # Respan's ingest does not decompress request bodies; keep this \`none\` unless
    # your endpoint (e.g. a proxy in front of a self-hosted deployment) accepts gzip.
    compression: \${env:RESPAN_COLLECTOR_COMPRESSION:-none}
    headers:
      Authorization: "Bearer \${env:RESPAN_API_KEY}"
    timeout: 30s
    sending_queue:
      enabled: true
      storage: file_storage
      # Number of batches kept while Respan is unreachable (also bounded by max_size).
      queue_size: \${env:RESPAN_COLLECTOR_QUEUE_SIZE:-10000}
      num_consumers: 4
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 60s
      # Never give up on a batch; the queue bounds are the only limit.
      max_elapsed_time: 0

service:
  extensions: [file_storage, health_check]
  telemetry:
    metrics:
      level: normal
      readers:
        - pull:
            exporter:
              prometheus:
                host: \${env:RESPAN_COLLECTOR_METRICS_HOST:-127.0.0.1}
                port: 8888
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlphttp/respan]
`;

export interface CollectorConfigOptions {
  /** Full ingest URL, e.g. https://api.respan.ai/api/v2/traces (RESPAN_COLLECTOR_EXPORT_URL). */
  exportUrl?: string;
  /** Persistent queue directory (RESPAN_COLLECTOR_DATA_DIR). */
  dataDir?: string;
  /** Disk cap for the queue in bytes (RESPAN_COLLECTOR_MAX_SIZE_BYTES). */
  maxSizeBytes?: number;
  /** Batches kept while Respan is unreachable (RESPAN_COLLECTOR_QUEUE_SIZE). */
  queueSize?: number;
  /** OTLP/HTTP listen address, e.g. 127.0.0.1:4318 (RESPAN_COLLECTOR_LISTEN). */
  listen?: string;
  /** health_check listen address (RESPAN_COLLECTOR_HEALTH_ENDPOINT). */
  healthEndpoint?: string;
  /** Prometheus metrics bind host (RESPAN_COLLECTOR_METRICS_HOST). */
  metricsHost?: string;
  /** memory_limiter limit (RESPAN_COLLECTOR_MEMORY_LIMIT_MIB). */
  memoryLimitMib?: number;
}

const CONFIG_ENV_VARS: Record<keyof CollectorConfigOptions, string> = {
  exportUrl: 'RESPAN_COLLECTOR_EXPORT_URL',
  dataDir: 'RESPAN_COLLECTOR_DATA_DIR',
  maxSizeBytes: 'RESPAN_COLLECTOR_MAX_SIZE_BYTES',
  queueSize: 'RESPAN_COLLECTOR_QUEUE_SIZE',
  listen: 'RESPAN_COLLECTOR_LISTEN',
  healthEndpoint: 'RESPAN_COLLECTOR_HEALTH_ENDPOINT',
  metricsHost: 'RESPAN_COLLECTOR_METRICS_HOST',
  memoryLimitMib: 'RESPAN_COLLECTOR_MEMORY_LIMIT_MIB',
};

/**
 * Render the collector config. Each `${env:NAME:-default}` placeholder whose
 * option is given is replaced by that value; the others keep the placeholder
 * (and therefore stay overridable through the environment). RESPAN_API_KEY is
 * never substituted — the file must not contain the secret.
 */
export function buildCollectorConfig(opts: CollectorConfigOptions = {}): string {
  const values = new Map<string, string>();
  for (const [key, envVar] of Object.entries(CONFIG_ENV_VARS) as [keyof CollectorConfigOptions, string][]) {
    const value = opts[key];
    if (value !== undefined && value !== null && String(value) !== '') {
      values.set(envVar, String(value));
    }
  }
  return COLLECTOR_CONFIG_TEMPLATE.replace(
    /\$\{env:([A-Z0-9_]+):-[^}]*\}/g,
    (placeholder, name: string) => values.get(name) ?? placeholder,
  );
}

export interface CollectorPaths {
  /** ~/.respan/collector */
  root: string;
  configPath: string;
  dataDir: string;
  binDir: string;
  binaryPath: string;
  pidPath: string;
  logPath: string;
}

/** Where the CLI keeps the collector config, queue, binary, pid and log. */
export function collectorPaths(home: string = os.homedir()): CollectorPaths {
  const root = path.join(home, '.respan', 'collector');
  return {
    root,
    configPath: path.join(root, 'config.yaml'),
    dataDir: path.join(root, 'data'),
    binDir: path.join(root, 'bin'),
    binaryPath: path.join(root, 'bin', 'otelcol-contrib'),
    pidPath: path.join(root, 'collector.pid'),
    logPath: path.join(root, 'collector.log'),
  };
}

/**
 * Ingest URL for a Respan API base URL. Mirrors the SDKs, which accept both
 * `https://api.respan.ai` and `https://api.respan.ai/api` and post to
 * `<origin>/api/v2/traces`.
 */
export function exportUrlFromBaseUrl(baseUrl: string): string {
  const origin = baseUrl.trim().replace(/\/+$/, '').replace(/\/api$/, '').replace(/\/+$/, '');
  return `${origin}/api/v2/traces`;
}

/** Base URL the SDKs / pi extension should use to reach a local collector. */
export function collectorBaseUrl(port: number = DEFAULT_COLLECTOR_PORT): string {
  return `http://127.0.0.1:${port}`;
}

export interface DockerRunOptions {
  /** Host path of the rendered config (mounted read-only). */
  configPath: string;
  /** Host port published on 127.0.0.1 for the OTLP/HTTP receiver. */
  port: number;
  exportUrl: string;
  maxSizeBytes: number;
  queueSize: number;
  image?: string;
}

/**
 * Arguments for `docker <args>` that start the collector container. All ports
 * are published on 127.0.0.1 only. RESPAN_API_KEY is passed through from the
 * caller's environment (`-e RESPAN_API_KEY` without a value) so the secret
 * never appears on the command line.
 */
export function dockerRunArgs(opts: DockerRunOptions): string[] {
  return [
    'run',
    '-d',
    '--name', CONTAINER_NAME,
    '--restart', 'unless-stopped',
    '-p', `127.0.0.1:${opts.port}:${DEFAULT_COLLECTOR_PORT}`,
    '-p', `127.0.0.1:${DEFAULT_HEALTH_PORT}:${DEFAULT_HEALTH_PORT}`,
    '-p', `127.0.0.1:${DEFAULT_METRICS_PORT}:${DEFAULT_METRICS_PORT}`,
    '-e', 'RESPAN_API_KEY',
    '-e', `RESPAN_COLLECTOR_EXPORT_URL=${opts.exportUrl}`,
    '-e', `RESPAN_COLLECTOR_MAX_SIZE_BYTES=${opts.maxSizeBytes}`,
    '-e', `RESPAN_COLLECTOR_QUEUE_SIZE=${opts.queueSize}`,
    '-e', `RESPAN_COLLECTOR_LISTEN=0.0.0.0:${DEFAULT_COLLECTOR_PORT}`,
    '-e', `RESPAN_COLLECTOR_HEALTH_ENDPOINT=0.0.0.0:${DEFAULT_HEALTH_PORT}`,
    '-e', 'RESPAN_COLLECTOR_METRICS_HOST=0.0.0.0',
    '-e', `RESPAN_COLLECTOR_DATA_DIR=${CONTAINER_DATA_DIR}`,
    '-v', `${opts.configPath}:${CONTAINER_CONFIG_PATH}:ro`,
    '-v', `${VOLUME_NAME}:${CONTAINER_DATA_DIR}`,
    opts.image ?? COLLECTOR_IMAGE,
    '--config', CONTAINER_CONFIG_PATH,
  ];
}

/**
 * One-off `docker <args>` that makes a fresh queue volume writable by the
 * collector. Docker creates a new named volume owned by root, and the upstream
 * image runs as uid 10001 without a shell, so a helper image does the chown
 * (the same trick collector/Dockerfile uses at build time).
 */
export function dockerVolumeInitArgs(volume: string = VOLUME_NAME): string[] {
  return [
    'run', '--rm',
    '-v', `${volume}:/queue`,
    VOLUME_INIT_IMAGE,
    'chown', `${CONTAINER_UID}:${CONTAINER_UID}`, '/queue',
  ];
}

const RELEASES_BASE = 'https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download';

const RELEASE_OS: Record<string, string> = { darwin: 'darwin', linux: 'linux' };
const RELEASE_ARCH: Record<string, string> = { arm64: 'arm64', x64: 'amd64' };

/** Release tarball for otelcol-contrib (contains `otelcol-contrib` and README.md). */
export function binaryDownloadUrl(
  version: string = COLLECTOR_VERSION,
  platform: string = process.platform,
  arch: string = process.arch,
): string {
  const os = RELEASE_OS[platform];
  const cpu = RELEASE_ARCH[arch];
  if (!os || !cpu) {
    throw new Error(
      `No otelcol-contrib binary for ${platform}/${arch} (supported: darwin/linux on arm64/x64). ` +
        'Use `respan collector start --docker` or run the collector another way.',
    );
  }
  return `${RELEASES_BASE}/v${version}/otelcol-contrib_${version}_${os}_${cpu}.tar.gz`;
}

/**
 * SHA-256 of the tarball. The release publishes one `<asset>.sha256` file per
 * asset (there is no aggregate checksums file); its body is the bare hex digest.
 */
export function checksumUrl(
  version: string = COLLECTOR_VERSION,
  platform: string = process.platform,
  arch: string = process.arch,
): string {
  return `${binaryDownloadUrl(version, platform, arch)}.sha256`;
}

/** Extract the hex digest from a `.sha256` body (`<hex>` or `<hex>  <file>`). */
export function parseSha256(text: string): string | undefined {
  return text.match(/\b[0-9a-fA-F]{64}\b/)?.[0].toLowerCase();
}

/**
 * Parse Prometheus text exposition into metric name → value. Labels are
 * stripped; series that share a name are summed.
 */
export function parsePrometheusText(text: string): Map<string, number> {
  const metrics = new Map<string, number>();
  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const match = line.match(/^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{.*\})?\s+(\S+)/);
    if (!match) continue;
    const value = Number(match[2]);
    if (!Number.isFinite(value)) continue;
    metrics.set(match[1], (metrics.get(match[1]) ?? 0) + value);
  }
  return metrics;
}

export interface CollectorMetricsSummary {
  /** Batches currently waiting in the persistent queue. */
  queueSize: number;
  /** sending_queue.queue_size as reported by the exporter. */
  queueCapacity: number;
  /** Spans dropped because the queue (or its disk cap) was full. */
  enqueueFailedSpans: number;
  /** Spans given up on after export failures (permanent errors such as 4xx). */
  sendFailedSpans: number;
  /** Spans acknowledged by Respan. */
  sentSpans: number;
}

/**
 * Pick the exporter metrics out of a scrape. otelcol-contrib 0.160.0 exposes
 * them without a `_total` suffix (e.g. `otelcol_exporter_sent_spans`); the
 * suffixed spelling is accepted too in case a build re-enables it. Counters
 * only appear once they have been incremented, so a missing one reads as 0.
 */
export function summarizeCollectorMetrics(metrics: Map<string, number>): CollectorMetricsSummary {
  const get = (name: string): number => metrics.get(name) ?? metrics.get(`${name}_total`) ?? 0;
  return {
    queueSize: get('otelcol_exporter_queue_size'),
    queueCapacity: get('otelcol_exporter_queue_capacity'),
    enqueueFailedSpans: get('otelcol_exporter_enqueue_failed_spans'),
    sendFailedSpans: get('otelcol_exporter_send_failed_spans'),
    sentSpans: get('otelcol_exporter_sent_spans'),
  };
}

/** Human-readable byte count (1024-based). */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '-';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}
