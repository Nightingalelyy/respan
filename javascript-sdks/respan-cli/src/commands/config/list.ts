import { BaseCommand } from '../../lib/base-command.js';
import { getConfig, getAllCredentials, getActiveProfile } from '../../lib/config.js';

export default class ConfigList extends BaseCommand {
  static description = 'List all configuration';
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { flags } = await this.parse(ConfigList);
    this.globalFlags = flags;
    const config = getConfig();
    const creds = getAllCredentials();
    const active = getActiveProfile();
    this.log(`Active profile: ${active}`);
    this.log(`Profiles: ${Object.keys(creds).join(', ') || '(none)'}`);
    if (config.defaults) {
      const { base_url, baseUrl, api_base_url, ...remainingDefaults } = config.defaults as Record<string, string>;
      if (Object.keys(remainingDefaults).length > 0) {
        this.log(`Defaults: ${JSON.stringify(remainingDefaults)}`);
      }
      if (base_url || baseUrl || api_base_url) {
        this.log(
          'Note: stored `base_url` config is ignored for CLI API commands. Use `respan auth login --base-url ...` to save an endpoint, or `--base-url` for a temporary override.',
        );
      }
    }
  }
}
