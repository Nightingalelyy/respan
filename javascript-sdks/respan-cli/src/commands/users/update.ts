import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class UsersUpdate extends BaseCommand {
  static description = 'Update a user (customer)';
  static args = { id: Args.string({ description: 'Customer identifier', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Customer name' }),
    email: Flags.string({ description: 'Customer email' }),
    metadata: Flags.string({ description: 'Metadata as JSON string' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(UsersUpdate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const body: Record<string, unknown> = { customer_identifier: args.id };
      if (flags.name) body.name = flags.name;
      if (flags.email) body.email = flags.email;
      if (flags.metadata) {
        try {
          body.metadata = JSON.parse(flags.metadata);
        } catch {
          this.error('Invalid JSON for --metadata');
        }
      }
      const data = await this.spin('Updating user', () => client.users.updateUser(body as any));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
