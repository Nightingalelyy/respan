import { Args, Flags } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class PromptsCreateVersion extends BaseCommand {
  static description = 'Create a new version of a prompt';
  static args = { 'prompt-id': Args.string({ description: 'Prompt ID', required: true }) };
  static flags = {
    ...BaseCommand.baseFlags,
    messages: Flags.string({ description: 'Messages as JSON array string', required: true }),
    model: Flags.string({ description: 'Model name' }),
    temperature: Flags.string({ description: 'Temperature value' }),
    'max-tokens': Flags.integer({ description: 'Max tokens' }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(PromptsCreateVersion);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      let messages: unknown;
      try {
        messages = JSON.parse(flags.messages);
      } catch {
        this.error('Invalid JSON for --messages');
      }
      // createVersion requires prompt_id, messages (string[]), and model (required)
      const messagesStr = Array.isArray(messages) ? (messages as any[]).map((m: any) => typeof m === 'string' ? m : JSON.stringify(m)) : [String(messages)];
      const createBody: Record<string, unknown> = {
        prompt_id: args['prompt-id'],
        messages: messagesStr,
        model: flags.model || 'gpt-4o',
      };
      if (flags.temperature) createBody.temperature = parseFloat(flags.temperature);
      if (flags['max-tokens']) createBody.max_tokens = flags['max-tokens'];

      const data = await this.spin('Creating prompt version', () => client.prompts.createVersion(createBody as any));
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
