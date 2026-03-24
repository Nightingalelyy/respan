import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class EvaluatorsUpdate extends BaseCommand {
  static description = 'Update an evaluator';
  static args = { id: Args.string({ description: 'Evaluator ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    name: Flags.string({ description: 'Evaluator name' }),
    description: Flags.string({ description: 'Evaluator description' }),
    config: Flags.string({ description: 'Evaluator config as JSON string' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(EvaluatorsUpdate);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const updateBody: Record<string, unknown> = {};
      if (flags.name) updateBody.name = flags.name;
      if (flags.description) updateBody.description = flags.description;
      if (flags.config) {
        try {
          Object.assign(updateBody, JSON.parse(flags.config));
        } catch {
          this.error('Invalid JSON for --config');
        }
      }
      const data = await this.spin('Updating evaluator', () => client.evaluators.updateEvaluator({
        evaluator_id: args.id,
        body: updateBody,
      }));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
