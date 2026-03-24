import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class LogsCreate extends BaseCommand {
  static description = 'Create a log span';
  static flags = {
    ...BaseCommand.baseFlags,
    input: Flags.string({ description: 'Input text or JSON', required: true }),
    output: Flags.string({ description: 'Output text or JSON' }),
    model: Flags.string({ description: 'Model name' }),
    metadata: Flags.string({ description: 'Metadata as JSON string' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(LogsCreate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const body: Record<string, unknown> = { input: flags.input };
      if (flags.output) body.output = flags.output;
      if (flags.model) body.model = flags.model;
      if (flags.metadata) {
        try {
          body.metadata = JSON.parse(flags.metadata);
        } catch {
          this.error('Invalid JSON for --metadata');
        }
      }
      const data = await this.spin('Creating span', () => client.logs.createSpan(body));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
