import * as fs from 'node:fs';
// execSync no longer needed — JS hooks don't require pip install
import { BaseCommand } from '../../lib/base-command.js';
import {
  integrateFlags,
  deepMerge,
  readJsonFile,
  writeJsonFile,
  writeTextFile,
  expandHome,
  parseAttrs,
  getJsHookScript,
  resolveScope,
  findProjectRoot,
  DEFAULT_BASE_URL,
} from '../../lib/integrate.js';

export default class IntegrateClaudeCode extends BaseCommand {
  static description = `Integrate Respan with Claude Code.

Installs a Stop hook that reads conversation transcripts and sends
them to Respan as structured spans (chat, tool, thinking).

Scope:
  --global   Install hook script + register in ~/.claude/settings.json
  --local    Write credentials + enable flag to .claude/settings.local.json
  (default)  Both: install hook globally + enable for current project`;

  static examples = [
    'respan integrate claude-code',
    'respan integrate claude-code --disable',
    'respan integrate claude-code --global',
    'respan integrate claude-code --local --project-id my-project',
    'respan integrate claude-code --attrs \'{"env":"prod"}\'',
    'respan integrate claude-code --dry-run',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...integrateFlags,
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(IntegrateClaudeCode);
    this.globalFlags = flags;

    try {
      const dryRun = flags['dry-run'];
      const scope = resolveScope(flags, 'both');
      const doLocal = scope === 'local' || scope === 'both';

      // ── Disable mode ─────────────────────────────────────────────
      if (flags.disable) {
        // Remove the Stop hook entry from global settings
        const globalSettingsPath = expandHome('~/.claude/settings.json');
        const globalSettings = readJsonFile(globalSettingsPath);
        const hooksSection = (globalSettings.hooks || {}) as Record<string, unknown>;
        if (Array.isArray(hooksSection.Stop)) {
          hooksSection.Stop = (hooksSection.Stop as Array<Record<string, unknown>>).filter((entry) => {
            const inner = Array.isArray(entry.hooks)
              ? (entry.hooks as Array<Record<string, unknown>>)
              : [];
            return !inner.some(
              (h) => typeof h.command === 'string' &&
                ((h.command as string).includes('respan') || (h.command as string).includes('hook.py') || (h.command as string).includes('claude-code')),
            );
          });
        }
        const merged = deepMerge(globalSettings, { hooks: hooksSection });
        if (dryRun) {
          this.log(`[dry-run] Would update: ${globalSettingsPath}`);
          this.log(JSON.stringify(merged, null, 2));
        } else {
          writeJsonFile(globalSettingsPath, merged);
          this.log(`Removed hook entry: ${globalSettingsPath}`);
        }
        this.log('Claude Code tracing disabled. Run "respan integrate claude-code" to re-enable.');
        return;
      }

      // ── Enable mode (default) ────────────────────────────────────
      // Verify the user is authenticated (key is read by hook from ~/.respan/)
      this.resolveApiKey();
      const baseUrl = flags['base-url']!;
      const projectId = flags['project-id'];
      const customerId = flags['customer-id'];
      const spanName = flags['span-name'];
      const workflowName = flags['workflow-name'];
      const attrs = parseAttrs(flags.attrs!);
      // Claude Code default: both global + local

      const doGlobal = scope === 'global' || scope === 'both';

      // ── Global: hook script + registration ────────────────────────
      if (doGlobal) {
        // 1. Write JS hook script (no Python dependency needed)
        const hookDir = expandHome('~/.respan/hooks');
        const hookPath = `${hookDir}/claude-code.cjs`;
        if (dryRun) {
          this.log(`[dry-run] Would write hook script to: ${hookPath}`);
        } else {
          writeTextFile(hookPath, getJsHookScript('claude-code'));
          fs.chmodSync(hookPath, 0o755);
          this.log(`Wrote hook script: ${hookPath}`);
        }

        // 2. Register Stop hook in global settings (no credentials here)
        const globalSettingsPath = expandHome('~/.claude/settings.json');
        const globalSettings = readJsonFile(globalSettingsPath);

        const hookEntry = {
          matcher: '',
          hooks: [{ type: 'command', command: `node ${hookPath}` }],
        };

        const hooksSection = (globalSettings.hooks || {}) as Record<string, unknown>;
        const stopHooks = Array.isArray(hooksSection.Stop)
          ? [...(hooksSection.Stop as Array<Record<string, unknown>>)]
          : [];

        const existingIdx = stopHooks.findIndex((entry) => {
          const inner = Array.isArray(entry.hooks)
            ? (entry.hooks as Array<Record<string, unknown>>)
            : [];
          return inner.some(
            (h) => typeof h.command === 'string' &&
              ((h.command as string).includes('respan') || (h.command as string).includes('hook.py') || (h.command as string).includes('claude-code.js')),
          );
        });

        if (existingIdx >= 0) {
          stopHooks[existingIdx] = hookEntry;
        } else {
          stopHooks.push(hookEntry);
        }

        const mergedGlobal = deepMerge(globalSettings, {
          hooks: { ...hooksSection, Stop: stopHooks },
        });

        if (dryRun) {
          this.log(`[dry-run] Would update: ${globalSettingsPath}`);
          this.log(JSON.stringify(mergedGlobal, null, 2));
        } else {
          writeJsonFile(globalSettingsPath, mergedGlobal);
          this.log(`Updated global settings: ${globalSettingsPath}`);
        }
      }

      // ── Local: credentials + enable in .claude/settings.local.json ─
      if (doLocal) {
        const projectRoot = findProjectRoot();
        const localSettingsPath = `${projectRoot}/.claude/settings.local.json`;
        const localSettings = readJsonFile(localSettingsPath);

        // settings.local.json: enable flag only (API key from ~/.respan/)
        const envBlock: Record<string, string> = {
          TRACE_TO_RESPAN: 'true',
        };

        const mergedLocal = deepMerge(localSettings, { env: envBlock });

        if (dryRun) {
          this.log(`[dry-run] Would update: ${localSettingsPath}`);
          this.log(JSON.stringify(mergedLocal, null, 2));
        } else {
          writeJsonFile(localSettingsPath, mergedLocal);
          this.log(`Updated project settings: ${localSettingsPath}`);
        }

        // respan.json: non-secret config (known fields + custom properties)
        const respanConfigPath = `${projectRoot}/.claude/respan.json`;
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

      // ── Done ──────────────────────────────────────────────────────
      this.log('');
      if (doGlobal && doLocal) {
        this.log('Claude Code integration complete (global hook + project config).');
      } else if (doGlobal) {
        this.log('Claude Code global hook installed.');
        this.log('Run without --global in a project to enable tracing there.');
      } else {
        this.log('Claude Code tracing enabled for this project.');
      }
      this.log('');
      this.log('Auth:   ~/.respan/credentials.json  (from `respan auth login`)');
      this.log('Config: .claude/respan.json                (shareable, non-secret)');
      this.log('');
      this.log('Set properties via integrate flags or edit .claude/respan.json:');
      this.log('  respan integrate claude-code --customer-id "frank" --span-name "my-app"');
      this.log('  respan integrate claude-code --attrs \'{"team":"platform","env":"staging"}\'');
      this.log('');
      this.log('Override per-session with env vars:');
      this.log('  export RESPAN_CUSTOMER_ID="your-name"');
      this.log("  export RESPAN_METADATA='{\"task_id\":\"T-123\"}'");
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
