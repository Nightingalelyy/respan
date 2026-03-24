import { BaseCommand } from '../../lib/base-command.js';
import { getActiveProfile, getCredential } from '../../lib/config.js';

export default class AuthStatus extends BaseCommand {
  static description = 'Show current authentication status';
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { flags } = await this.parse(AuthStatus);
    this.globalFlags = flags;
    const profile = flags.profile || getActiveProfile();
    const cred = getCredential(profile);
    if (!cred) {
      this.log('Not authenticated. Run `respan auth login`.');
      return;
    }
    this.log(`Profile: ${profile}`);
    this.log(`Type: ${cred.type}`);
    if (cred.type === 'api_key') this.log(`API Key: ${cred.apiKey.slice(0, 8)}...`);
    if (cred.type === 'jwt') this.log(`Email: ${cred.email}`);
    this.log(`Base URL: ${cred.baseUrl}`);
  }
}
