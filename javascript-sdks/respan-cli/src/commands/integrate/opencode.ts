import * as path from 'node:path';
import { execSync } from 'node:child_process';
import { BaseCommand } from '../../lib/base-command.js';
import {
  integrateFlags,
  writeJsonFile,
  expandHome,
  parseAttrs,
  resolveScope,
  findProjectRoot,
} from '../../lib/integrate.js';

export default class IntegrateOpencode extends BaseCommand {
  static description = `Integrate Respan with OpenCode.

OpenCode's built-in OTel does not work reliably, so this uses the
community opencode-otel plugin instead.

Scope:
  --local    Write plugin config to project root (default)
  --global   Write to ~/.config/opencode/plugins/otel.json
  The opencode-otel package is always installed globally.`;

  static examples = [
    'respan integrate opencode',
    'respan integrate opencode --global',
    'respan integrate opencode --project-id my-project --attrs \'{"env":"prod"}\'',
    'respan integrate opencode --dry-run',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...integrateFlags,
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(IntegrateOpencode);
    this.globalFlags = flags;

    try {
      const apiKey = this.resolveApiKey();
      const baseUrl = (flags['base-url']!).replace(/\/+$/, '');
      const projectId = flags['project-id'];
      const attrs = parseAttrs(flags.attrs!);
      const dryRun = flags['dry-run'];
      const scope = resolveScope(flags, 'local');

      // ── 1. Install opencode-otel (always global) ─────────────────
      if (dryRun) {
        this.log('[dry-run] Would run: npm install -g opencode-otel');
      } else {
        this.log('Installing opencode-otel...');
        try {
          execSync('npm install -g opencode-otel', { stdio: 'pipe' });
          this.log('Installed opencode-otel globally.');
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          this.warn(`Failed to install opencode-otel: ${msg}`);
          this.warn('You may need to install it manually: npm install -g opencode-otel');
        }
      }

      // ── 2. Write plugin config ────────────────────────────────────
      const pluginPath = scope === 'global'
        ? expandHome('~/.config/opencode/plugins/otel.json')
        : path.join(findProjectRoot(), '.opencode', 'plugins', 'otel.json');

      const resourceAttrs: Record<string, string> = {
        'service.name': 'opencode',
        ...attrs,
      };
      if (projectId) {
        resourceAttrs['respan.project_id'] = projectId;
      }

      const pluginConfig: Record<string, unknown> = {
        tracesEndpoint: `${baseUrl}/v2/traces`,
        logsEndpoint: `${baseUrl}/v2/logs`,
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
        resourceAttributes: resourceAttrs,
      };

      if (dryRun) {
        this.log(`[dry-run] Would write: ${pluginPath}`);
        this.log(JSON.stringify(pluginConfig, null, 2));
      } else {
        writeJsonFile(pluginPath, pluginConfig);
        this.log(`Wrote plugin config: ${pluginPath}`);
      }

      // ── Done ──────────────────────────────────────────────────────
      this.log('');
      this.log(`OpenCode integration complete (${scope}).`);
      this.log('');
      this.log('Set dynamic attributes before a session:');
      this.log('  export OTEL_RESOURCE_ATTRIBUTES="env=prod,task_id=T-123"');
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
