import assert from 'node:assert/strict';
import test from 'node:test';

import { TRACING_MD } from '../dist/lib/skill-refs.generated.js';

test('Auto tracing setup keeps the required Vercel AI SDK instrumentation', () => {
  const autoPath = TRACING_MD.match(/### Auto path([\s\S]*?)\n---\n\n### Full path/)?.[1];

  assert.ok(autoPath, 'expected an Auto path in the bundled tracing skill');
  assert.match(autoPath, /Vercel AI SDK/);
  assert.match(autoPath, /@respan\/instrumentation-vercel/);
  assert.match(autoPath, /@ai-sdk\/otel/);
  assert.match(autoPath, /EveInstrumentor/);
  assert.match(autoPath, /VercelAIInstrumentor/);
  assert.match(autoPath, /instrumentations:\s*\[new VercelAIInstrumentor\(\)\]/);
  assert.match(autoPath, /telemetry:\s*\{\s*isEnabled:\s*true/s);
  assert.match(autoPath, /experimental_telemetry:\s*\{\s*isEnabled:\s*true/s);
});
