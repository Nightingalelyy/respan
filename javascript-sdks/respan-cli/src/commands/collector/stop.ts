import * as fs from 'node:fs';
import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { CONTAINER_NAME, VOLUME_NAME, collectorPaths } from '../../lib/collector.js';
import {
  collectorModeFlags,
  detectRunning,
  dockerVolumeExists,
  runDocker,
  stopPid,
} from '../../lib/collector-runtime.js';

export default class CollectorStop extends BaseCommand {
  static description = `Stop the local Respan collector.

Removes the ${CONTAINER_NAME} container or stops the detached binary. Spans
still queued on disk are kept and delivered the next time the collector
starts; pass --purge --yes to delete them.`;

  static examples = [
    'respan collector stop',
    'respan collector stop --purge --yes',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...collectorModeFlags,
    purge: Flags.boolean({
      description: `Also delete the queued data (the ${VOLUME_NAME} volume / data dir). Requires --yes`,
      default: false,
    }),
    yes: Flags.boolean({
      description: 'Confirm deleting queued spans with --purge',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(CollectorStop);
    this.globalFlags = flags;

    try {
      if (flags.purge && !flags.yes) {
        this.error('--purge deletes any spans still queued for delivery. Re-run with --yes to confirm.');
      }

      const paths = collectorPaths();
      const mode = flags.docker ? 'docker' : flags.binary ? 'binary' : undefined;
      const running = detectRunning(paths, mode);
      let stopped = false;

      if (running?.mode === 'docker') {
        const result = await runDocker(['rm', '-f', CONTAINER_NAME]);
        if (result.status !== 0) {
          this.error(`Failed to remove container ${CONTAINER_NAME}: ${result.stderr.trim()}`);
        }
        this.log(`Removed container ${CONTAINER_NAME}.`);
        stopped = true;
      } else if (running?.mode === 'binary' && running.pid !== undefined) {
        const outcome = await stopPid(running.pid);
        fs.rmSync(paths.pidPath, { force: true });
        if (outcome === 'killed') {
          this.log(`Collector (pid ${running.pid}) did not exit after SIGTERM; killed it.`);
        } else {
          this.log(`Stopped collector (pid ${running.pid}).`);
        }
        stopped = true;
      } else {
        this.log('The collector is not running.');
      }

      if (flags.purge) {
        let purged = false;
        if (mode !== 'binary' && dockerVolumeExists()) {
          const result = await runDocker(['volume', 'rm', VOLUME_NAME]);
          if (result.status === 0) {
            this.log(`Removed volume ${VOLUME_NAME}.`);
            purged = true;
          } else {
            this.warn(`Failed to remove volume ${VOLUME_NAME}: ${result.stderr.trim()}`);
          }
        }
        if (mode !== 'docker' && fs.existsSync(paths.dataDir)) {
          fs.rmSync(paths.dataDir, { recursive: true, force: true });
          this.log(`Removed queue directory ${paths.dataDir}.`);
          purged = true;
        }
        if (!purged) this.log('No queued data to remove.');
      } else if (stopped) {
        this.log('Queued spans are kept on disk and will be delivered when the collector starts again.');
        this.log('To delete them: respan collector stop --purge --yes');
      }
    } catch (error) {
      this.handleError(error);
    }
  }
}
