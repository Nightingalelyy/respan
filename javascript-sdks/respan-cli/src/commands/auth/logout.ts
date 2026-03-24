import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { deleteCredential, getActiveProfile } from '../../lib/config.js';

export default class AuthLogout extends BaseCommand {
  static description = 'Log out and remove stored credentials';
  static flags = {
    ...BaseCommand.baseFlags,
    profile: Flags.string({ description: 'Profile to log out' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(AuthLogout);
    this.globalFlags = flags;
    const profile = flags.profile || getActiveProfile();
    deleteCredential(profile);
    this.log(`Logged out of profile "${profile}".`);
  }
}
