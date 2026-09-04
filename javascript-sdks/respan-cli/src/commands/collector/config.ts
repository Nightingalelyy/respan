import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { expandHome, writeTextFile } from '../../lib/integrate.js';
import {
  COLLECTOR_VERSION,
  buildCollectorConfig,
  collectorBaseUrl,
  collectorPaths,
} from '../../lib/collector.js';
import { collectorSizingFlags, resolveExportUrl } from '../../lib/collector-runtime.js';

export default class CollectorConfig extends BaseCommand {
  static description = `Write the Respan collector configuration.

The Respan collector is the upstream OpenTelemetry Collector (contrib
${COLLECTOR_VERSION}) with a config that receives OTLP/HTTP from the Respan SDKs
on localhost and forwards it to Respan through a bounded, persistent
(write-ahead-log) queue — spans survive Respan outages and collector
restarts without unbounded local growth.

The written file never contains your API key: it reads RESPAN_API_KEY from
the environment. Use "respan collector start" to run it, or run
otelcol-contrib yourself with --config <file>.`;

  static examples = [
    'respan collector config',
    'respan collector config --print',
    'respan collector config --output ./collector.yaml --max-size-mib 1024',
    'respan collector config --export-url https://respan.internal/api/v2/traces',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...collectorSizingFlags,
    output: Flags.string({
      description: 'Where to write the config (default: ~/.respan/collector/config.yaml)',
    }),
    print: Flags.boolean({
      description: 'Print the config to stdout instead of writing it',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(CollectorConfig);
    this.globalFlags = flags;

    try {
      const paths = collectorPaths();
      const exportUrl = resolveExportUrl(flags);
      const dataDir = expandHome(flags['data-dir'] ?? paths.dataDir);
      const yaml = buildCollectorConfig({
        exportUrl,
        dataDir,
        maxSizeBytes: flags['max-size-mib'] * 1024 * 1024,
        queueSize: flags['queue-size'],
        listen: `127.0.0.1:${flags.port}`,
      });

      if (flags.print) {
        process.stdout.write(yaml);
        return;
      }

      const output = expandHome(flags.output ?? paths.configPath);
      writeTextFile(output, yaml);
      this.log(`Wrote collector config: ${output}`);
      this.log('');
      this.log(`Export URL:   ${exportUrl}`);
      this.log(`Queue dir:    ${dataDir}`);
      this.log(`Queue limits: ${flags['max-size-mib']} MiB on disk, ${flags['queue-size']} batches`);
      this.log('');
      this.log('Run it:');
      this.log('  respan collector start');
      this.log(`  # or: RESPAN_API_KEY=... otelcol-contrib --config ${output}`);
      this.log('');
      this.log('Point the SDK at the collector:');
      this.log(`  export RESPAN_BASE_URL=${collectorBaseUrl(flags.port)}`);
    } catch (error) {
      this.handleError(error);
    }
  }
}
