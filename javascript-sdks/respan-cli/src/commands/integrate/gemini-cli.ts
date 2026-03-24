import * as fs from 'node:fs';
import * as path from 'node:path';
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

export default class IntegrateGeminiCli extends BaseCommand {
  static description = `Integrate Respan with Gemini CLI.

Installs an AfterModel hook that captures LLM request/response data
and sends it to Respan as structured spans with model, token counts,
and input/output.

Scope:
  --global   Write to ~/.gemini/settings.json (default)
  --local    Write to .gemini/settings.json in project root

Note: Gemini CLI ignores workspace-level telemetry settings, so
--global is the default.`;

  static examples = [
    'respan integrate gemini-cli',
    'respan integrate gemini-cli --local',
    'respan integrate gemini-cli --project-id my-project --attrs \'{"env":"prod"}\'',
    'respan integrate gemini-cli --dry-run',
  ];

  static flags = {
    ...BaseCommand.baseFlags,
    ...integrateFlags,
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(IntegrateGeminiCli);
    this.globalFlags = flags;

    try {
      // Verify the user is authenticated (key is read by hook from ~/.respan/)
      this.resolveApiKey();
      const projectId = flags['project-id'];
      const customerId = flags['customer-id'];
      const spanName = flags['span-name'];
      const workflowName = flags['workflow-name'];
      const attrs = parseAttrs(flags.attrs!);
      const dryRun = flags['dry-run'];
      const scope = resolveScope(flags, 'global');

      // ── 1. Install hook script ──────────────────────────────────
      const hookDir = expandHome('~/.respan/hooks');
      const hookPath = `${hookDir}/gemini-cli.cjs`;
      if (dryRun) {
        this.log(`[dry-run] Would write hook script to: ${hookPath}`);
      } else {
        writeTextFile(hookPath, getJsHookScript('gemini-cli'));
        fs.chmodSync(hookPath, 0o755);
        this.log(`Wrote hook script: ${hookPath}`);
      }

      // ── 2. Register hooks in settings.json ────────────────────────
      const settingsPath = scope === 'global'
        ? expandHome('~/.gemini/settings.json')
        : path.join(findProjectRoot(), '.gemini', 'settings.json');

      const existing = readJsonFile(settingsPath);

      const hookEntry = {
        hooks: [{ type: 'command', command: `node ${hookPath}` }],
      };

      const hooksSection = (existing.hooks || {}) as Record<string, unknown>;

      // Register the same hook script for AfterModel, BeforeTool, and AfterTool.
      // AfterModel captures streaming text; BeforeTool/AfterTool capture tool
      // names, arguments, and output for rich tool spans.
      const hookEvents = ['AfterModel', 'BeforeTool', 'AfterTool'] as const;
      const updatedHooks: Record<string, unknown> = { ...hooksSection };

      for (const eventName of hookEvents) {
        const eventHooks = Array.isArray(hooksSection[eventName])
          ? [...(hooksSection[eventName] as Array<Record<string, unknown>>)]
          : [];

        // Replace existing respan hook or add new one
        const existingIdx = eventHooks.findIndex((entry) => {
          const inner = Array.isArray(entry.hooks)
            ? (entry.hooks as Array<Record<string, unknown>>)
            : [];
          return inner.some(
            (h) => typeof h.command === 'string' &&
              ((h.command as string).includes('respan') || (h.command as string).includes('gemini_hook') || (h.command as string).includes('gemini-cli.js')),
          );
        });

        if (existingIdx >= 0) {
          eventHooks[existingIdx] = hookEntry;
        } else {
          eventHooks.push(hookEntry);
        }

        updatedHooks[eventName] = eventHooks;
      }

      // Remove legacy telemetry config (from older CLI versions) to avoid
      // Gemini CLI sending broken OTEL requests to the wrong endpoint.
      // The hooks handle all trace export now.
      const mergeSource: Record<string, unknown> = { hooks: updatedHooks };
      if ((existing as any).telemetry) {
        mergeSource.telemetry = undefined;
      }
      const merged = deepMerge(existing, mergeSource);
      // Explicitly remove telemetry key if it existed
      delete (merged as any).telemetry;

      // ── 3. Write respan.json with non-secret config ─────────────
      const configPath = expandHome('~/.gemini/respan.json');
      const respanConfig = readJsonFile(configPath);
      const newConfig: Record<string, unknown> = { ...respanConfig };

      const baseUrl = flags['base-url'];
      if (baseUrl && baseUrl !== DEFAULT_BASE_URL) {
        newConfig.base_url = baseUrl;
      }
      if (customerId) newConfig.customer_id = customerId;
      if (spanName) newConfig.span_name = spanName;
      if (workflowName) newConfig.workflow_name = workflowName;
      if (projectId) newConfig.project_id = projectId;
      for (const [k, v] of Object.entries(attrs)) {
        newConfig[k] = v;
      }

      if (dryRun) {
        this.log(`[dry-run] Would update: ${settingsPath}`);
        this.log(JSON.stringify(merged, null, 2));
        if (Object.keys(newConfig).length > 0) {
          this.log('');
          this.log(`[dry-run] Would write: ${configPath}`);
          this.log(JSON.stringify(newConfig, null, 2));
        }
      } else {
        writeJsonFile(settingsPath, merged);
        this.log(`Updated settings: ${settingsPath}`);
        if (Object.keys(newConfig).length > 0) {
          writeJsonFile(configPath, newConfig);
          this.log(`Wrote Respan config: ${configPath}`);
        }
      }

      this.log('');
      this.log(`Gemini CLI integration complete (${scope}).`);
      this.log('');
      this.log('Auth:   ~/.respan/credentials.json  (from `respan auth login`)');
      this.log('Config: ~/.gemini/respan.json               (shareable, non-secret)');
      this.log('');
      this.log('Set properties via integrate flags or edit ~/.gemini/respan.json:');
      this.log('  respan integrate gemini-cli --customer-id "frank" --span-name "my-app"');
      this.log('  respan integrate gemini-cli --attrs \'{"team":"platform","env":"staging"}\'');
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
