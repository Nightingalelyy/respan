import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { getConfigValue } from '../../lib/config.js';

export default class ConfigGet extends BaseCommand {
  static description = 'Get a configuration value';
  static args = { key: Args.string({ description: 'Config key', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(ConfigGet);
    this.globalFlags = flags;
    const value = getConfigValue(args.key);
    this.log(value !== undefined ? String(value) : `Key "${args.key}" not set.`);
  }
}
