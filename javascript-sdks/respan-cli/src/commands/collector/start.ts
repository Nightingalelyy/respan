import * as fs from 'node:fs';
import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { ensureDir, expandHome, writeTextFile } from '../../lib/integrate.js';
import {
  COLLECTOR_IMAGE,
  CONTAINER_NAME,
  DEFAULT_HEALTH_PORT,
  VOLUME_NAME,
  binaryDownloadUrl,
  buildCollectorConfig,
  collectorBaseUrl,
  collectorPaths,
  dockerRunArgs,
  dockerVolumeInitArgs,
} from '../../lib/collector.js';
import {
  checkHealth,
  collectorModeFlags,
  collectorSizingFlags,
  detectRunning,
  dockerAvailable,
  dockerVolumeExists,
  ensureCollectorBinary,
  healthUrl,
  isPidAlive,
  metricsUrl,
  resolveExportUrl,
  runDocker,
  spawnCollectorBinary,
  tailFile,
  waitForHealth,
  type CollectorMode,
  type HealthStatus,
} from '../../lib/collector-runtime.js';

const HEALTH_TIMEOUT_MS = 20000;

export default class CollectorStart extends BaseCommand {
  static description = `Start a local Respan collector.

Runs the OpenTelemetry Collector (contrib) with the Respan config: the SDKs
send to http://127.0.0.1:4318, and the collector forwards to Respan through
a bounded, persistent queue so traces survive outages and restarts.

Modes:
  --docker   docker run otel/opentelemetry-collector-contrib with the queue in
             the ${VOLUME_NAME} volume (default when Docker is available)
  --binary   download otelcol-contrib to ~/.respan/collector/bin and run it
             detached, logging to ~/.respan/collector/collector.log

The API key is passed to the collector through its environment only; the
config file never contains it.`;

  static examples = [
    'respan collector start',
    'respan collector start --binary',
    'respan collector start --docker --max-size-mib 1024 --queue-size 20000',
    'respan collector start --export-url https://respan.internal/api/v2/traces',
    'respan collector start --dry-run',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...collectorModeFlags,
    ...collectorSizingFlags,
    'dry-run': Flags.boolean({
      description: 'Show what would be done without starting anything',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(CollectorStart);
    this.globalFlags = flags;

    try {
      const dryRun = flags['dry-run'];
      const apiKey = this.resolveApiKey();
      const paths = collectorPaths();

      let mode: CollectorMode;
      if (flags.docker) {
        if (!dockerAvailable()) {
          this.error('Docker is not available (`docker info` failed). Start Docker or use --binary.');
        }
        mode = 'docker';
      } else if (flags.binary) {
        mode = 'binary';
      } else {
        mode = dockerAvailable() ? 'docker' : 'binary';
      }

      // ── Refuse to double-start ────────────────────────────────────
      const running = detectRunning(paths);
      if (running?.mode === 'docker') {
        const status = running.container?.status ?? 'unknown';
        this.error(
          `Container ${CONTAINER_NAME} already exists (${status}). ` +
            'Run `respan collector stop` first' +
            (status === 'running' ? '.' : `, or resume it with \`docker start ${CONTAINER_NAME}\`.`),
        );
      }
      if (running?.mode === 'binary') {
        this.error(
          `The collector is already running (pid ${running.pid}). ` +
            'See `respan collector status` or stop it with `respan collector stop`.',
        );
      }
      if (!dryRun && (await checkHealth()).healthy) {
        this.error(
          `Something already answers on ${healthUrl()} (another collector?) but was not started by this CLI. ` +
            'Stop it before starting a new one.',
        );
      }

      const exportUrl = resolveExportUrl(flags);
      const port = flags.port;
      const maxSizeBytes = flags['max-size-mib'] * 1024 * 1024;
      const queueSize = flags['queue-size'];
      const dataDir = expandHome(flags['data-dir'] ?? paths.dataDir);
      const env = { ...process.env, RESPAN_API_KEY: apiKey };

      if (mode === 'docker') {
        // Everything is passed as container env; the mounted file is the pure template.
        const yaml = buildCollectorConfig();
        const runArgs = dockerRunArgs({ configPath: paths.configPath, port, exportUrl, maxSizeBytes, queueSize });
        const needsVolume = !dockerVolumeExists();
        if (flags['data-dir']) {
          this.warn(`--data-dir is ignored in docker mode; the queue lives in the ${VOLUME_NAME} volume.`);
        }

        if (dryRun) {
          this.log('[dry-run] Mode: docker');
          this.log(`[dry-run] Would write: ${paths.configPath}`);
          if (needsVolume) {
            this.log(`[dry-run] Would run: docker ${dockerVolumeInitArgs().join(' ')}`);
          }
          this.log(`[dry-run] Would run: RESPAN_API_KEY=*** docker ${runArgs.join(' ')}`);
          this.log(`[dry-run] Would wait for ${healthUrl()}`);
          return;
        }

        writeTextFile(paths.configPath, yaml);
        this.log(`Wrote collector config: ${paths.configPath}`);
        if (needsVolume) {
          const init = await this.spin(`Creating volume ${VOLUME_NAME}`, () => runDocker(dockerVolumeInitArgs()));
          if (init.status !== 0) {
            this.error(`Failed to prepare volume ${VOLUME_NAME}: ${init.stderr.trim()}`);
          }
        }
        const started = await this.spin(`Starting ${CONTAINER_NAME} (${COLLECTOR_IMAGE})`, () => runDocker(runArgs, env));
        if (started.status !== 0) {
          this.error(`docker run failed: ${started.stderr.trim()}`);
        }
        const health = await this.awaitHealth();
        if (!health.healthy) {
          this.warn(
            `Collector is not healthy yet (${health.error ?? health.status}). ` +
              `Check \`docker logs ${CONTAINER_NAME}\` or \`respan collector status\`.`,
          );
        }
        this.printStarted(mode, health, port, exportUrl);
        return;
      }

      // ── Binary mode ───────────────────────────────────────────────
      const yaml = buildCollectorConfig({ exportUrl, dataDir, maxSizeBytes, queueSize, listen: `127.0.0.1:${port}` });

      if (dryRun) {
        this.log('[dry-run] Mode: binary');
        if (!fs.existsSync(paths.binaryPath)) {
          this.log(`[dry-run] Would download ${binaryDownloadUrl()} to ${paths.binDir}`);
        }
        this.log(`[dry-run] Would write: ${paths.configPath}  (preview: respan collector config --print)`);
        this.log(`[dry-run] Would run: RESPAN_API_KEY=*** ${paths.binaryPath} --config ${paths.configPath}`);
        this.log(`[dry-run]   detached; log: ${paths.logPath}; pid file: ${paths.pidPath}`);
        this.log(`[dry-run] Would wait for ${healthUrl()}`);
        return;
      }

      await ensureCollectorBinary(paths, this);
      writeTextFile(paths.configPath, yaml);
      this.log(`Wrote collector config: ${paths.configPath}`);
      ensureDir(dataDir);
      const pid = spawnCollectorBinary(paths, env);
      this.log(`Started otelcol-contrib (pid ${pid}); log: ${paths.logPath}`);

      const health = await this.awaitHealth(() => isPidAlive(pid));
      if (!health.healthy) {
        if (!isPidAlive(pid)) {
          fs.rmSync(paths.pidPath, { force: true });
          this.error(`The collector exited during startup. Last lines of ${paths.logPath}:\n${tailFile(paths.logPath, 15)}`);
        }
        this.warn(
          `Collector is not healthy yet (${health.error ?? health.status}). ` +
            `Check ${paths.logPath} or \`respan collector status\`.`,
        );
      }
      this.printStarted(mode, health, port, exportUrl);
    } catch (error) {
      this.handleError(error);
    }
  }

  /** Poll the health endpoint behind a spinner that fails visibly on timeout / exit. */
  private async awaitHealth(isAlive?: () => boolean): Promise<HealthStatus> {
    let unhealthy: HealthStatus | undefined;
    try {
      return await this.spin('Waiting for the collector to become healthy', async () => {
        const health = await waitForHealth(DEFAULT_HEALTH_PORT, HEALTH_TIMEOUT_MS, isAlive);
        if (!health.healthy) {
          unhealthy = health;
          throw new Error(health.error ?? health.status ?? 'unhealthy');
        }
        return health;
      });
    } catch (error) {
      if (unhealthy) return unhealthy;
      throw error;
    }
  }

  private printStarted(mode: CollectorMode, health: HealthStatus, port: number, exportUrl: string): void {
    this.log('');
    this.log(`Collector ${health.healthy ? 'is healthy' : 'started'} (${mode}).`);
    this.log(`  Receiver: ${collectorBaseUrl(port)}/api/v2/traces`);
    this.log(`  Health:   ${healthUrl()}`);
    this.log(`  Metrics:  ${metricsUrl()}`);
    this.log(`  Export:   ${exportUrl}`);
    this.log('');
    this.log('Point the SDK at the collector:');
    this.log(`  export RESPAN_BASE_URL=${collectorBaseUrl(port)}`);
    this.log('');
    this.log('Check / stop:');
    this.log('  respan collector status');
    this.log('  respan collector stop');
  }

  private resolveApiKey(): string {
    const auth = this.getAuth();
    if (auth.apiKey) return auth.apiKey;
    if (auth.accessToken) {
      this.warn('Using access token (JWT) which may expire. Consider using an API key instead.');
      return auth.accessToken;
    }
    this.error('No API key found. Pass --api-key, set RESPAN_API_KEY, or run: respan auth login');
  }
}
