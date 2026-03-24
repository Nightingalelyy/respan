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
    if (config.defaults) this.log(`Defaults: ${JSON.stringify(config.defaults)}`);
  }
}
