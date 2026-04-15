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
    const normalizedKey = args.key.toLowerCase();
    if (['base_url', 'baseurl', 'api_base_url'].includes(normalizedKey)) {
      this.error(
        'CLI API base URLs are no longer managed with `respan config set`. Use `respan auth login --base-url ...` to save an endpoint, or `--base-url` on a command for a temporary override.',
        { exit: 1 },
      );
    }
    setConfigValue(args.key, args.value);
    this.log(`Set ${args.key} = ${args.value}`);
  }
}
