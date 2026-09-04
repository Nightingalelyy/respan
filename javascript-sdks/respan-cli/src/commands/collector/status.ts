import * as fs from 'node:fs';
import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { GREEN, RED, YELLOW, DIM, RESET } from '../../lib/colors.js';
import {
  CONTAINER_NAME,
  DEFAULT_COLLECTOR_PORT,
  VOLUME_NAME,
  collectorBaseUrl,
  collectorPaths,
  formatBytes,
} from '../../lib/collector.js';
import {
  checkHealth,
  collectorModeFlags,
  detectRunning,
  directorySize,
  fetchMetrics,
  healthUrl,
  metricsUrl,
} from '../../lib/collector-runtime.js';

export default class CollectorStatus extends BaseCommand {
  static description = `Show the local Respan collector's health and queue.

Reports how the collector is running (Docker container or binary), the
health_check result, and the exporter metrics that matter for delivery:
queued batches, spans sent, spans dropped because the queue was full
(enqueue-failed) and spans rejected by Respan (send-failed).`;

  static examples = [
    'respan collector status',
    'respan collector status --json',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...collectorModeFlags,
    port: Flags.integer({
      description: 'Local OTLP/HTTP port used in the SDK hint',
      default: DEFAULT_COLLECTOR_PORT,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(CollectorStatus);
    this.globalFlags = flags;

    try {
      const paths = collectorPaths();
      const mode = flags.docker ? 'docker' : flags.binary ? 'binary' : undefined;
      const running = detectRunning(paths, mode);
      const health = await checkHealth();
      const metrics = await fetchMetrics();
      const dataDirBytes = fs.existsSync(paths.dataDir) ? directorySize(paths.dataDir) : undefined;
      const isRunning = running?.mode === 'docker' ? Boolean(running.container?.running) : Boolean(running);

      const result = {
        mode: running?.mode ?? null,
        running: isRunning,
        healthy: health.healthy,
        health,
        container: running?.container ?? null,
        pid: running?.pid ?? null,
        metrics: metrics ?? null,
        dataDir: running?.mode === 'docker' ? null : paths.dataDir,
        dataDirBytes: running?.mode === 'docker' ? null : dataDirBytes ?? null,
        volume: running?.mode === 'docker' ? VOLUME_NAME : null,
        configPath: paths.configPath,
        logPath: running?.mode === 'binary' ? paths.logPath : null,
        baseUrl: collectorBaseUrl(flags.port),
      };

      if (flags.json) {
        this.outputResult(result);
        return;
      }

      const ok = `${GREEN}✓${RESET}`;
      const bad = `${RED}✗${RESET}`;
      const warn = `${YELLOW}!${RESET}`;

      this.log('');
      if (running?.mode === 'docker' && running.container) {
        const c = running.container;
        const mark = c.running ? ok : bad;
        this.log(`  ${mark} Mode:    docker (container ${CONTAINER_NAME}: ${c.status}${c.restartCount ? `, ${c.restartCount} restarts` : ''})`);
        if (c.running && c.startedAt) this.log(`    Started: ${c.startedAt}`);
        if (!c.running) this.log(`    ${DIM}Resume with: docker start ${CONTAINER_NAME}  (or respan collector stop && respan collector start)${RESET}`);
      } else if (running?.mode === 'binary') {
        this.log(`  ${ok} Mode:    binary (pid ${running.pid})`);
        this.log(`    Log:     ${paths.logPath}`);
      } else {
        this.log(`  ${bad} Mode:    not running`);
        this.log(`    ${DIM}Start with: respan collector start${RESET}`);
      }

      if (health.healthy) {
        this.log(`  ${ok} Health:  ${health.status}${health.uptime ? ` (up ${health.uptime})` : ''}  ${DIM}${healthUrl()}${RESET}`);
      } else {
        this.log(`  ${bad} Health:  ${health.error ?? health.status ?? 'unreachable'}  ${DIM}${healthUrl()}${RESET}`);
      }

      if (metrics) {
        const queueMark = metrics.queueSize > 0 ? warn : ok;
        this.log(`  ${queueMark} Queue:   ${metrics.queueSize} / ${metrics.queueCapacity} batches waiting  ${DIM}${metricsUrl()}${RESET}`);
        this.log(`    Spans:   sent ${metrics.sentSpans}, enqueue-failed ${metrics.enqueueFailedSpans}, send-failed ${metrics.sendFailedSpans}`);
        if (metrics.enqueueFailedSpans > 0) {
          this.log(`    ${warn} ${metrics.enqueueFailedSpans} spans were dropped because the queue was full (raise --max-size-mib / --queue-size).`);
        }
        if (metrics.sendFailedSpans > 0) {
          this.log(`    ${warn} ${metrics.sendFailedSpans} spans were rejected by Respan (check the API key and export URL).`);
        }
      } else {
        this.log(`  ${DIM}  Metrics: unavailable (${metricsUrl()})${RESET}`);
      }

      if (running?.mode === 'docker') {
        this.log(`    Queue:   docker volume ${VOLUME_NAME}`);
      } else if (dataDirBytes !== undefined) {
        this.log(`    Queue:   ${paths.dataDir} (${formatBytes(dataDirBytes)})`);
      }
      this.log(`    Config:  ${paths.configPath}${fs.existsSync(paths.configPath) ? '' : ` ${DIM}(not written yet)${RESET}`}`);
      this.log('');
      this.log(`  SDK: export RESPAN_BASE_URL=${collectorBaseUrl(flags.port)}`);
      this.log('');
    } catch (error) {
      this.handleError(error);
    }
  }
}
