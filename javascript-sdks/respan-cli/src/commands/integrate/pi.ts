import { execSync } from 'node:child_process';
import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { isBinaryInstalled } from '../../lib/agents.js';
import {
  DEFAULT_BASE_URL,
  integrateFlags,
  readJsonFile,
  writeJsonFile,
  parseAttrs,
  resolveScope,
  findProjectRoot,
} from '../../lib/integrate.js';
import {
  PI_PACKAGE_SPEC,
  buildPiRespanConfig,
  piConfigPath,
  piInstallArgs,
  type PiTraceScope,
} from '../../lib/pi-integrate.js';
import { DEFAULT_COLLECTOR_PORT, collectorBaseUrl } from '../../lib/collector.js';

export default class IntegratePi extends BaseCommand {
  static description = `Integrate Respan with the pi coding agent (https://pi.dev).

Installs the @respan/instrumentation-pi pi package — a pi extension that
traces every agent run as one turn span per prompt with chat and tool spans underneath,
correlating all runs of a pi session by its session id — and writes a
non-secret respan.json config for it.

Scope:
  --global   pi install npm:@respan/instrumentation-pi + ~/.pi/agent/respan.json (default)
  --local    pi install -l npm:@respan/instrumentation-pi + .pi/respan.json in the project root

Trace scope (--trace-scope, "trace_scope" in respan.json):
  session    one multi-root trace per pi session, across processes and pauses (default)
  run        one trace per agent run; runs of a session share a thread id

Durable delivery (--with-collector):
  Sets base_url to a local Respan collector (http://127.0.0.1:4318) so traces
  go through its persistent queue and survive Respan outages. Run
  "respan collector start" once per machine.

Credentials are never written to these files: the extension reads
RESPAN_API_KEY or ~/.respan/credentials.json (from "respan auth login").`;

  static examples = [
    'respan integrate pi',
    'respan integrate pi --disable',
    'respan integrate pi --local',
    'respan integrate pi --customer-id frank --workflow-name email-agent',
    'respan integrate pi --trace-scope run',
    'respan integrate pi --with-collector',
    'respan integrate pi --project-id my-project --attrs \'{"env":"prod"}\'',
    'respan integrate pi --dry-run',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...integrateFlags,
    // The shared descriptions advertise the claude-code default; pi's is "pi".
    'workflow-name': Flags.string({
      description: 'traceloop.workflow.name on pi turn spans (default: pi)',
    }),
    'span-name': Flags.string({
      description: 'Alias of --workflow-name (default: pi)',
    }),
    'trace-scope': Flags.string({
      description:
        'Trace scope: "session" = one multi-root trace per pi session (default), "run" = one trace per agent run',
      options: ['run', 'session'],
    }),
    'with-collector': Flags.boolean({
      description: 'Send traces through a local Respan collector (respan collector start) for durable delivery',
      default: false,
    }),
    'collector-port': Flags.integer({
      description: `Port of the local collector, with --with-collector (default: ${DEFAULT_COLLECTOR_PORT})`,
      min: 1,
      max: 65535,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(IntegratePi);
    this.globalFlags = flags;

    try {
      const dryRun = flags['dry-run'];
      // pi loads global packages from ~/.pi/agent, so global is the default.
      const scope = resolveScope(flags, 'global');
      const projectRoot = scope === 'local' ? findProjectRoot() : process.cwd();
      const configPath = piConfigPath(scope, projectRoot);

      // ── Disable mode ─────────────────────────────────────────────
      if (flags.disable) {
        const disabled = buildPiRespanConfig(readJsonFile(configPath), { enabled: false });
        if (dryRun) {
          this.log(`[dry-run] Would write: ${configPath}`);
          this.log(JSON.stringify(disabled, null, 2));
        } else {
          writeJsonFile(configPath, disabled);
          this.log(`Disabled tracing in: ${configPath}`);
        }
        this.log('');
        this.log('pi tracing disabled (the extension package stays installed). Run "respan integrate pi" to re-enable.');
        this.log(`To uninstall the extension package: pi remove ${PI_PACKAGE_SPEC}`);
        return;
      }

      // ── Enable mode (default) ────────────────────────────────────
      // Auth precheck only — the key is read by the extension at runtime.
      this.resolveApiKey();
      const baseUrl = flags['base-url'];
      const projectId = flags['project-id'];
      const customerId = flags['customer-id'];
      // The extension treats span_name as an alias of workflow_name.
      const workflowName = flags['workflow-name'] ?? flags['span-name'];
      const traceScope = flags['trace-scope'] as PiTraceScope | undefined;
      const attrs = parseAttrs(flags.attrs!);
      const withCollector = flags['with-collector'];
      const collectorPort = flags['collector-port'] ?? DEFAULT_COLLECTOR_PORT;
      if (flags['collector-port'] !== undefined && !withCollector) {
        this.warn('--collector-port has no effect without --with-collector.');
      }
      if (withCollector && baseUrl && baseUrl.replace(/\/+$/, '') !== DEFAULT_BASE_URL) {
        this.warn(
          `--with-collector overrides --base-url: traces go to ${collectorBaseUrl(collectorPort)}. ` +
            'Point the collector at your endpoint with `respan collector start --export-url ...`.',
        );
      }

      // ── 1. Install the pi package ─────────────────────────────────
      const installCommand = ['pi', ...piInstallArgs(scope)].join(' ');
      if (!isBinaryInstalled('pi')) {
        const message =
          'pi is not installed. Install it with `npm install -g @earendil-works/pi-coding-agent` (see https://pi.dev)';
        if (dryRun) {
          this.warn(message);
        } else {
          this.error(message);
        }
      }

      if (dryRun) {
        this.log(`[dry-run] Would run: ${installCommand}`);
      } else {
        this.log(`Installing ${PI_PACKAGE_SPEC}...`);
        try {
          execSync(installCommand, { stdio: 'pipe', cwd: projectRoot });
          this.log(`Installed ${PI_PACKAGE_SPEC} (${scope}).`);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          this.warn(`Failed to install ${PI_PACKAGE_SPEC}: ${msg}`);
          this.warn(`You may need to install it manually: ${installCommand}`);
        }
      }

      // ── 2. Write respan.json (non-secret) ─────────────────────────
      const config = buildPiRespanConfig(readJsonFile(configPath), {
        enabled: true,
        customerId,
        workflowName,
        projectId,
        traceScope,
        baseUrl,
        collectorPort: withCollector ? collectorPort : undefined,
        attrs,
      });

      if (dryRun) {
        this.log(`[dry-run] Would write: ${configPath}`);
        this.log(JSON.stringify(config, null, 2));
      } else {
        writeJsonFile(configPath, config);
        this.log(`Wrote Respan config: ${configPath}`);
      }

      // ── Done ──────────────────────────────────────────────────────
      this.log('');
      this.log(`pi integration complete (${scope}). Restart pi to load the extension.`);
      this.log('');
      this.log('Auth:   RESPAN_API_KEY or ~/.respan/credentials.json  (from `respan auth login`)');
      this.log(`Config: ${configPath}  (shareable, non-secret)`);
      if (withCollector) {
        this.log('');
        this.log(`Traces go through the local collector at ${collectorBaseUrl(collectorPort)} (base_url in respan.json).`);
        this.log('Start it once per machine; it keeps running across pi sessions:');
        this.log(`  respan collector start${collectorPort === DEFAULT_COLLECTOR_PORT ? '' : ` --port ${collectorPort}`}`);
        this.log('Remove base_url from respan.json to send directly to Respan again.');
      }
      this.log('');
      this.log('Set properties via integrate flags or edit respan.json:');
      this.log('  respan integrate pi --customer-id "frank" --workflow-name "my-agent"');
      this.log('  respan integrate pi --attrs \'{"team":"platform","env":"staging"}\'');
      this.log('  respan integrate pi --trace-scope run       # one trace per prompt instead of one per session ("trace_scope")');
      this.log('');
      this.log('Override per-session with env vars:');
      this.log('  export RESPAN_CUSTOMER_ID="your-name"');
      this.log('  export RESPAN_PI_TRACE_SCOPE=run       # or session (default)');
      this.log('  export RESPAN_PI_TRACING=false     # skip tracing for one session');
      this.log('');
      this.log('One-off run without installing (from a project that has the package):');
      this.log('  pi -e ./node_modules/@respan/instrumentation-pi/dist/extension.js');
      this.log('');
      this.log('Debug: RESPAN_PI_DEBUG=true pi   (logs to stderr only; nothing is written to disk)');
    } catch (error) {
      this.handleError(error);
    }
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
