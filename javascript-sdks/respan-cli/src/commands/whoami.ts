import { BaseCommand } from '../lib/base-command.js';
import { getActiveProfile, getCredential } from '../lib/config.js';

export default class Whoami extends BaseCommand {
  static description = 'Show current user information';
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { flags } = await this.parse(Whoami);
    this.globalFlags = flags;
    const profile = flags.profile || getActiveProfile();
    const cred = getCredential(profile);
    if (!cred) {
      this.log('Not authenticated.');
      return;
    }
    const auth = this.getAuth();
    this.log(`Profile: ${profile}`);
    if (cred.type === 'jwt') this.log(`Email: ${cred.email}`);
    this.log(`Base URL: ${auth.baseUrl}`);
  }
}
