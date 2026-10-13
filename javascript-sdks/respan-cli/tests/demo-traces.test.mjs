import assert from 'node:assert/strict';
import test from 'node:test';

import { DEMO_TRACES } from '../dist/lib/demo/trace-fixtures.js';

const SEMANTIC_NAME = /^(llm\.[a-z0-9.-]+|agent\.[a-z_]+|tool\.[a-z_]+|handoff\.[a-z]+_to_[a-z]+|embedding)$/;

function spansOf(fixture) {
  return fixture.body.resourceSpans.flatMap((rs) => rs.scopeSpans.flatMap((ss) => ss.spans));
}

function attrsOf(span) {
  return Object.fromEntries(
    (span.attributes ?? []).map((a) => [a.key, Object.values(a.value)[0]]),
  );
}

test('demo traces form one connected tree each', () => {
  assert.equal(DEMO_TRACES.length, 12);
  for (const fixture of DEMO_TRACES) {
    const spans = spansOf(fixture);
    const ids = new Set(spans.map((s) => s.spanId));
    assert.equal(spans.length, 17, fixture.workflow);
    assert.equal(ids.size, spans.length, fixture.workflow);
    assert.equal(spans.filter((s) => !s.parentSpanId).length, 1, fixture.workflow);
    for (const span of spans) {
      if (span.parentSpanId) assert.ok(ids.has(span.parentSpanId), `${fixture.workflow}: ${span.name}`);
    }
  }
});

test('demo child spans use the semantic span names', () => {
  for (const fixture of DEMO_TRACES) {
    for (const span of spansOf(fixture).filter((s) => s.parentSpanId)) {
      assert.match(span.name, SEMANTIC_NAME, `${fixture.workflow}: ${span.name}`);
      const attrs = attrsOf(span);
      if (span.name.startsWith('llm.')) {
        assert.equal(span.name, `llm.${attrs['gen_ai.request.model']}`);
      }
      if (span.name.startsWith('handoff.')) {
        assert.equal(attrs['respan.entity.log_type'], 'handoff');
        // The backend reads respan.entity.log_type only when no span kind is set.
        assert.equal(attrs['traceloop.span.kind'], undefined);
      }
    }
  }
});
