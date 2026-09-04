import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';

import { DEFAULT_BASE_URL } from '../dist/lib/integrate.js';
import {
  PI_PACKAGE_NAME,
  PI_PACKAGE_SPEC,
  buildPiRespanConfig,
  piConfigPath,
  piInstallArgs,
} from '../dist/lib/pi-integrate.js';

test('pi install args target the npm package spec for each scope', () => {
  assert.equal(PI_PACKAGE_NAME, '@respan/instrumentation-pi');
  assert.equal(PI_PACKAGE_SPEC, 'npm:@respan/instrumentation-pi');
  assert.deepEqual(piInstallArgs('global'), ['install', 'npm:@respan/instrumentation-pi']);
  assert.deepEqual(piInstallArgs('local'), ['install', '-l', 'npm:@respan/instrumentation-pi']);
});

test('pi config path follows the extension lookup locations', () => {
  assert.equal(
    piConfigPath('global', '/repo', '/home/me'),
    path.join('/home/me', '.pi', 'agent', 'respan.json'),
  );
  assert.equal(
    piConfigPath('local', '/repo', '/home/me'),
    path.join('/repo', '.pi', 'respan.json'),
  );
  // The home directory defaults to the current user's home.
  assert.ok(piConfigPath('global', '/repo').endsWith(path.join('.pi', 'agent', 'respan.json')));
});

test('buildPiRespanConfig writes snake_case keys and layers over existing config', () => {
  const existing = {
    workflow_name: 'old-workflow',
    metadata: { team: 'platform', env: 'dev' },
    extra: 'kept',
  };

  const config = buildPiRespanConfig(existing, {
    enabled: true,
    customerId: 'frank',
    projectId: 'proj-1',
    baseUrl: 'https://respan.example.com/api/',
    attrs: { env: 'prod' },
  });

  assert.deepEqual(config, {
    enabled: true,
    workflow_name: 'old-workflow',
    customer_id: 'frank',
    project_id: 'proj-1',
    base_url: 'https://respan.example.com/api',
    metadata: { team: 'platform', env: 'prod' },
    extra: 'kept',
  });
  // Existing input is not mutated.
  assert.deepEqual(existing.metadata, { team: 'platform', env: 'dev' });
});

test('buildPiRespanConfig omits the default base URL, empty metadata, and unset flags', () => {
  const config = buildPiRespanConfig({}, {
    enabled: true,
    baseUrl: DEFAULT_BASE_URL,
    attrs: {},
  });

  assert.deepEqual(config, { enabled: true });
});

test('buildPiRespanConfig overrides workflow_name and replaces non-object metadata', () => {
  const config = buildPiRespanConfig(
    { workflow_name: 'old', metadata: 'not-an-object' },
    { enabled: true, workflowName: 'email-agent', attrs: { task: 'triage' } },
  );

  assert.equal(config.workflow_name, 'email-agent');
  assert.deepEqual(config.metadata, { task: 'triage' });
});

test('buildPiRespanConfig writes trace_scope and keeps it across re-runs and disable', () => {
  const config = buildPiRespanConfig({}, { enabled: true, traceScope: 'session' });
  assert.deepEqual(config, { enabled: true, trace_scope: 'session' });

  // Re-running without the flag keeps the stored scope; passing it overrides.
  assert.equal(buildPiRespanConfig(config, { enabled: true, workflowName: 'email-agent' }).trace_scope, 'session');
  assert.equal(buildPiRespanConfig(config, { enabled: true, traceScope: 'run' }).trace_scope, 'run');
  assert.deepEqual(buildPiRespanConfig(config, { enabled: false }), { enabled: false, trace_scope: 'session' });
  // Not written unless requested.
  assert.equal('trace_scope' in buildPiRespanConfig({}, { enabled: true }), false);
});

test('buildPiRespanConfig disable keeps the rest of the config intact', () => {
  const config = buildPiRespanConfig(
    { enabled: true, customer_id: 'frank', base_url: 'https://respan.example.com/api' },
    { enabled: false },
  );

  assert.deepEqual(config, {
    enabled: false,
    customer_id: 'frank',
    base_url: 'https://respan.example.com/api',
  });
});

test('buildPiRespanConfig never writes credentials', () => {
  const config = buildPiRespanConfig({}, {
    enabled: true,
    customerId: 'frank',
    attrs: { env: 'prod' },
  });

  const serialized = JSON.stringify(config);
  assert.equal(/api[_-]?key|access[_-]?token/i.test(serialized), false);
});
