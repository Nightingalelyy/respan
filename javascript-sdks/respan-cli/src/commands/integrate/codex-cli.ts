import * as fs from 'node:fs';
import { BaseCommand } from '../../lib/base-command.js';
import {
  integrateFlags,
  readTextFile,
  writeTextFile,
  writeJsonFile,
  readJsonFile,
  expandHome,
  parseAttrs,
  getJsHookScript,
  resolveScope,
  findProjectRoot,
} from '../../lib/integrate.js';

export default class IntegrateCodexCli extends BaseCommand {
  static description = `Integrate Respan with Codex CLI.

Installs a notify hook that reads session JSONL files and sends
them to Respan as structured spans (chat, tool, reasoning).

Scope:
  --global   Install hook script + register notify in ~/.codex/config.toml
  --local    Write .codex/respan.json with customer_id, span_name, etc. (default)`;

  static examples = [
    'respan integrate codex-cli',
    'respan integrate codex-cli --disable',
    'respan integrate codex-cli --global',
    'respan integrate codex-cli --local --customer-id frank',
    'respan integrate codex-cli --attrs \'{"env":"prod"}\'',
    'respan integrate codex-cli --dry-run',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...integrateFlags,
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(IntegrateCodexCli);
    this.globalFlags = flags;

    try {
      const dryRun = flags['dry-run'];

      // ── Disable mode ─────────────────────────────────────────────
      if (flags.disable) {
        // Remove the notify line from ~/.codex/config.toml
        const configPath = expandHome('~/.codex/config.toml');
        const existing = readTextFile(configPath);
        const lines = existing.split('\n');
        const filtered = lines.filter((line) =>
          !/^\s*notify\s*=/.test(line) && !line.includes('respan integrate codex-cli'),
        );
        if (dryRun) {
          this.log(`[dry-run] Would update: ${configPath}`);
          this.log(filtered.join('\n'));
        } else {
          writeTextFile(configPath, filtered.join('\n'));
          this.log(`Removed notify hook: ${configPath}`);
        }
        this.log('Codex CLI tracing disabled. Run "respan integrate codex-cli" to re-enable.');
        return;
      }

      // ── Enable mode (default) ────────────────────────────────────
      // Verify the user is authenticated (key is read by hook from ~/.respan/)
      this.resolveApiKey();
      const projectId = flags['project-id'];
      const customerId = flags['customer-id'];
      const spanName = flags['span-name'];
      const workflowName = flags['workflow-name'];
      const attrs = parseAttrs(flags.attrs!);
      // Codex CLI default: both global + local
      const scope = resolveScope(flags);

      const doGlobal = scope === 'global' || scope === 'both';
      const doLocal = scope === 'local' || scope === 'both';

      // ── Global: hook script + notify registration ──────────────────
      if (doGlobal) {
        // 1. Write JS hook script (no Python dependency needed)
        const hookDir = expandHome('~/.respan/hooks');
        const hookPath = `${hookDir}/codex-cli.cjs`;
        if (dryRun) {
          this.log(`[dry-run] Would write hook script to: ${hookPath}`);
        } else {
          writeTextFile(hookPath, getJsHookScript('codex-cli'));
          fs.chmodSync(hookPath, 0o755);
          this.log(`Wrote hook script: ${hookPath}`);
        }

        // 2. Register notify hook in ~/.codex/config.toml
        const configPath = expandHome('~/.codex/config.toml');
        const existing = readTextFile(configPath);
        const notifyValue = `notify = ["node", "${hookPath}"]`;

        const updated = this.updateTomlNotify(existing, notifyValue);

        if (dryRun) {
          this.log(`[dry-run] Would update: ${configPath}`);
          this.log(updated);
        } else {
          writeTextFile(configPath, updated);
          this.log(`Updated config: ${configPath}`);
        }
      }

      // ── Local: .codex/respan.json ──────────────────────────────────
      if (doLocal) {
        const projectRoot = findProjectRoot();
        const respanConfigPath = `${projectRoot}/.codex/respan.json`;
        const respanConfig = readJsonFile(respanConfigPath);
        const newConfig: Record<string, unknown> = { ...respanConfig };

        if (customerId) {
          newConfig.customer_id = customerId;
        }
        if (spanName) {
          newConfig.span_name = spanName;
        }
        if (workflowName) {
          newConfig.workflow_name = workflowName;
        }
        if (projectId) {
          newConfig.project_id = projectId;
        }
        // Custom attrs go as top-level keys (unknown keys = custom properties)
        for (const [k, v] of Object.entries(attrs)) {
          newConfig[k] = v;
        }

        if (Object.keys(newConfig).length > 0) {
          if (dryRun) {
            this.log(`[dry-run] Would write: ${respanConfigPath}`);
            this.log(JSON.stringify(newConfig, null, 2));
          } else {
            writeJsonFile(respanConfigPath, newConfig);
            this.log(`Wrote Respan config: ${respanConfigPath}`);
          }
        }
      }

      // ── Done ────────────────────────────────────────────────────────
      this.log('');
      if (doGlobal && doLocal) {
        this.log('Codex CLI integration complete (global hook + project config).');
      } else if (doGlobal) {
        this.log('Codex CLI global hook installed.');
        this.log('Run without --global in a project to configure tracing there.');
      } else {
        this.log('Codex CLI tracing configured for this project.');
      }
      this.log('');
      this.log('Auth:   ~/.respan/credentials.json  (from `respan auth login`)');
      this.log('Config: .codex/respan.json                 (shareable, non-secret)');
      this.log('');
      this.log('Set properties via integrate flags or edit .codex/respan.json:');
      this.log('  respan integrate codex-cli --customer-id "frank" --span-name "my-app"');
      this.log('  respan integrate codex-cli --attrs \'{"team":"platform","env":"staging"}\'');
      this.log('');
      this.log('Override per-session with env vars:');
      this.log('  export RESPAN_CUSTOMER_ID="your-name"');
      this.log("  export RESPAN_METADATA='{\"task_id\":\"T-123\"}'");
      this.log('');
      this.log('Debug: CODEX_RESPAN_DEBUG=true → check ~/.codex/state/respan_hook.log');
    } catch (error) {
      this.handleError(error);
    }
  }

  /**
   * Update or insert the `notify` line in a TOML config string.
   *
   * TOML bare keys must appear before the first [table] header.
   * If a `notify` line already exists, replace it in-place.
   * Otherwise, insert the notify line before the first [table] header.
   */
  private updateTomlNotify(existing: string, notifyValue: string): string {
    const lines = existing.split('\n');

    // Check if notify line already exists
    const notifyIdx = lines.findIndex((line) =>
      /^\s*notify\s*=/.test(line),
    );

    if (notifyIdx >= 0) {
      // Replace existing notify line
      lines[notifyIdx] = notifyValue;
      return lines.join('\n');
    }

    // Find the first [table] header to insert before it
    const firstTableIdx = lines.findIndex((line) =>
      /^\s*\[/.test(line),
    );

    if (firstTableIdx >= 0) {
      // Insert notify + comment before the first table header
      lines.splice(firstTableIdx, 0,
        '# Respan observability (added by respan integrate codex-cli)',
        notifyValue,
        '',
      );
    } else {
      // No tables at all — append to end
      lines.push(
        '',
        '# Respan observability (added by respan integrate codex-cli)',
        notifyValue,
      );
    }

    return lines.join('\n');
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
