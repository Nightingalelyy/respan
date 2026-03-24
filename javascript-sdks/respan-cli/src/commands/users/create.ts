import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class UsersCreate extends BaseCommand {
  static description = 'Create a new user (customer)';
  static flags = {
    ...BaseCommand.baseFlags,
    identifier: Flags.string({ description: 'Customer identifier', required: true }),
    name: Flags.string({ description: 'Customer name' }),
    email: Flags.string({ description: 'Customer email' }),
    metadata: Flags.string({ description: 'Metadata as JSON string' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(UsersCreate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const body: Record<string, unknown> = { customer_identifier: flags.identifier };
      if (flags.name) body.name = flags.name;
      if (flags.email) body.email = flags.email;
      if (flags.metadata) {
        try {
          body.metadata = JSON.parse(flags.metadata);
        } catch {
          this.error('Invalid JSON for --metadata');
        }
      }
      const data = await this.spin('Creating user', () => client.users.updateUser(body as any));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
