import assert from 'node:assert/strict';
import test from 'node:test';

import { SetupBaseCommand } from '../dist/lib/setup-base-command.js';

class SetupHarness extends SetupBaseCommand {
  async execute(opts) {
    await this.runSetup('tracing', opts);
  }

  async askApiKey() {
    return 'test-api-key';
  }

  async verifyApiKey() {
    return true;
  }

  async selectAgent() {
    this.events.push({ kind: 'selectAgent' });
    return 'codex-cli';
  }

  async installSkill() {
    this.events.push({ kind: 'installSkill' });
  }

  async launchAgent() {
    this.events.push({ kind: 'launchAgent' });
  }

  async notifySetup() {}

  getGitEmail() {
    return undefined;
  }

  logStep(step, label) {
    this.events.push({ kind: 'step', step, label });
  }

  log(message = '') {
    this.events.push({ kind: 'log', message });
  }
}

function createHarness() {
  const command = Object.create(SetupHarness.prototype);
  command.events = [];
  return command;
}

test('--no-instrument skips the agent picker and launch', async () => {
  const previousNoColor = process.env.NO_COLOR;
  process.env.NO_COLOR = '1';

  try {
    const command = createHarness();
    await command.execute({ agent: 'codex-cli', noInstrument: true });

    assert.deepEqual(
      command.events.filter(({ kind }) => kind === 'step'),
      [
        { kind: 'step', step: 1, label: 'API Key' },
        { kind: 'step', step: 2, label: 'Install skill' },
      ],
    );
    assert.equal(command.events.some(({ kind }) => kind === 'selectAgent'), false);
    assert.equal(command.events.some(({ kind }) => kind === 'launchAgent'), false);
    assert.equal(
      command.events.some(({ kind, message }) => kind === 'log' && message.includes('No agent selected')),
      false,
    );
  } finally {
    if (previousNoColor === undefined) delete process.env.NO_COLOR;
    else process.env.NO_COLOR = previousNoColor;
  }
});
