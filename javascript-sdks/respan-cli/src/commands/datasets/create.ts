import { Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class DatasetsCreate extends BaseCommand {
  static description = 'Create a new dataset';
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Dataset name', required: true }),
    description: Flags.string({ description: 'Dataset description' }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(DatasetsCreate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const body: Record<string, unknown> = { name: flags.name };
      if (flags.description) body.description = flags.description;
      const data = await this.spin('Creating dataset', () => client.datasets.createDataset(body as any));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
