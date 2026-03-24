import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';
import { setConfigValue } from '../../lib/config.js';

export default class ConfigSet extends BaseCommand {
  static description = 'Set a configuration value';
  static args = {
    key: Args.string({ description: 'Config key', required: true }),
    value: Args.string({ description: 'Config value', required: true }),
  };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(ConfigSet);
    this.globalFlags = flags;
    setConfigValue(args.key, args.value);
    this.log(`Set ${args.key} = ${args.value}`);
  }
}
