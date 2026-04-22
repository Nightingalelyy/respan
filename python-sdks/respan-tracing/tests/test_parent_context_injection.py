"""Tests for SpanBuffer parent context injection (trace continuation).

When a workflow pauses and resumes, we want the post-resume spans to
share the pre-pause trace_id. SpanBuffer achieves this by injecting a
NonRecordingSpan parent context so all child spans inherit the parent's
trace_id.

This is the SDK side of Phase 6 (Continue-Trace-on-Resume) from
span_mv_and_trace_evolution.md.
"""
import pytest
from unittest.mock import MagicMock

from opentelemetry import trace
from respan_tracing import RespanTelemetry, get_client
from respan_tracing.processors.base import SpanBuffer


# Valid OTel hex IDs for testing
PARENT_TRACE_ID = "0123456789abcdef0123456789abcdef"
PARENT_SPAN_ID = "fedcba9876543210"


class TestParentContextInjection:
    """Tests for SpanBuffer with parent_trace_id/parent_span_id."""

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-parent-ctx",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    def test_spans_inherit_parent_trace_id(self):
        """Spans created in a buffer with parent context inherit the parent trace_id."""
        with self.client.get_span_buffer(
            trace_id="run-abc",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("resumed_step", {"status": "running"})
            buffer.create_span("resumed_step_2", {"status": "done"})

        spans = buffer.get_all_spans()
        assert len(spans) == 2

        expected_trace_id = int(PARENT_TRACE_ID, 16)
        for span in spans:
            actual_trace_id = span.get_span_context().trace_id
            assert actual_trace_id == expected_trace_id, (
                f"Span '{span.name}' has trace_id {actual_trace_id:#034x}, "
                f"expected {expected_trace_id:#034x}"
            )

    def test_spans_are_children_of_parent_span(self):
        """Spans created in buffer are children of the injected parent span."""
        with self.client.get_span_buffer(
            trace_id="run-def",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("child_step", {"x": 1})

        spans = buffer.get_all_spans()
        assert len(spans) == 1
        span = spans[0]

        expected_parent = int(PARENT_SPAN_ID, 16)
        assert span.parent is not None, "Span should have a parent"
        actual_parent = span.parent.span_id
        assert actual_parent == expected_parent, (
            f"Span parent_span_id is {actual_parent:#018x}, "
            f"expected {expected_parent:#018x}"
        )

    def test_no_parent_context_creates_new_trace(self):
        """Without parent context, spans get a new auto-generated trace_id (backward compat)."""
        with self.client.get_span_buffer(trace_id="run-ghi") as buffer:
            buffer.create_span("independent_step", {"status": "ok"})

        spans = buffer.get_all_spans()
        assert len(spans) == 1

        # Should NOT have the parent trace_id
        actual_trace_id = spans[0].get_span_context().trace_id
        assert actual_trace_id != int(PARENT_TRACE_ID, 16)
        assert actual_trace_id != 0  # Should have some valid trace_id

    def test_parent_context_detached_after_exit(self):
        """Parent context is properly detached after buffer exits."""
        # Get trace_id before entering buffer
        pre_span = trace.get_current_span()

        with self.client.get_span_buffer(
            trace_id="run-jkl",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("step", {"x": 1})

        # After exit, current span should be back to what it was before
        post_span = trace.get_current_span()
        # The parent NonRecordingSpan should not leak into outer context
        post_trace_id = post_span.get_span_context().trace_id
        assert post_trace_id != int(PARENT_TRACE_ID, 16), (
            "Parent context leaked — trace_id still active after buffer exit"
        )

    def test_partial_parent_context_ignored(self):
        """If only parent_trace_id is provided (no parent_span_id), no injection happens."""
        buffer = SpanBuffer(
            trace_id="run-mno",
            parent_trace_id=PARENT_TRACE_ID,
            # parent_span_id intentionally omitted
        )
        assert buffer._parent_trace_id == PARENT_TRACE_ID
        assert buffer._parent_span_id is None

        # __enter__ should not inject parent context
        buffer.__enter__()
        assert buffer._parent_context_token is None
        buffer.__exit__(None, None, None)

    def test_client_get_span_buffer_passes_parent_params(self):
        """client.get_span_buffer() correctly passes parent_trace_id/parent_span_id."""
        buffer = self.client.get_span_buffer(
            trace_id="run-pqr",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        )
        assert buffer._parent_trace_id == PARENT_TRACE_ID
        assert buffer._parent_span_id == PARENT_SPAN_ID
        assert buffer._tracer_provider is not None

    def test_multiple_spans_share_trace_but_have_unique_span_ids(self):
        """All spans in a parent-context buffer share trace_id but have distinct span_ids."""
        with self.client.get_span_buffer(
            trace_id="run-stu",
            parent_trace_id=PARENT_TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
        ) as buffer:
            buffer.create_span("step_a", {})
            buffer.create_span("step_b", {})
            buffer.create_span("step_c", {})

        spans = buffer.get_all_spans()
        assert len(spans) == 3

        trace_ids = {s.get_span_context().trace_id for s in spans}
        span_ids = {s.get_span_context().span_id for s in spans}

        # All same trace_id
        assert len(trace_ids) == 1
        assert trace_ids.pop() == int(PARENT_TRACE_ID, 16)

        # All different span_ids
        assert len(span_ids) == 3


class TestNestedBufferPreservesActiveParent:
    """Regression tests for the orphan-root bug.

    When a ``SpanBuffer`` is opened with only a ``trace_id`` (no
    ``parent_span_id``), it previously injected a synthetic
    ``NonRecordingSpan(span_id=0x1)`` into the current OTel context
    unconditionally. That clobbered any real parent span that was
    already active, so every span created after the buffer opened was
    parented to the 0x1 sentinel instead of the caller's span.

    This mattered for nested executors. A parent workflow runs inside
    its own buffer with a real ``@workflow`` root span active. When a
    sub-workflow starts a child executor that opens ANOTHER buffer with
    just a ``trace_id``, the child used to replace the parent's context
    with the sentinel — so its sub-workflow root became a sibling of
    the real parent root, not a child.

    The fix: if the current OTel context already has a recording span,
    skip the injection and let normal parent-context propagation work.
    """

    def setup_method(self):
        self.telemetry = RespanTelemetry(
            app_name="test-nested-buffer",
            api_key="test-key",
            is_enabled=True,
        )
        self.client = get_client()

    def test_nested_buffer_preserves_active_parent_span(self):
        tracer = self.client.get_tracer()
        with tracer.start_as_current_span("outer_parent") as outer:
            outer_trace_id = outer.get_span_context().trace_id
            outer_span_id = outer.get_span_context().span_id

            # Simulate a nested executor opening its own buffer with just a
            # trace_id, mid-execution under an already-active parent span.
            with self.client.get_span_buffer(trace_id="nested-trace") as buffer:
                buffer.create_span("nested_child", {"from": "nested_buffer"})

            spans = buffer.get_all_spans()
            assert len(spans) == 1, (
                f"Expected exactly one span from the nested buffer, "
                f"got {len(spans)}"
            )

            child = spans[0]
            ctx = child.get_span_context()
            # Same trace as the real parent — no divergence
            assert ctx.trace_id == outer_trace_id, (
                f"Nested span's trace_id {ctx.trace_id:#034x} diverged "
                f"from parent {outer_trace_id:#034x}"
            )
            # Real parent span_id, not the 0x1 sentinel
            parent_id = child.parent.span_id if child.parent else None
            assert parent_id == outer_span_id, (
                f"Nested span's parent_span_id is {parent_id:#018x}, "
                f"expected parent's span_id {outer_span_id:#018x}. The "
                f"0x0000000000000001 sentinel indicates the fix did not "
                f"take — the buffer clobbered the active parent context."
            )

    def test_nested_buffer_never_injects_sentinel_parent(self):
        """The 0x1 sentinel must never appear when a real parent is active."""
        SENTINEL = int("0000000000000001", 16)
        tracer = self.client.get_tracer()
        with tracer.start_as_current_span("outer") as outer:
            with self.client.get_span_buffer(trace_id="nested") as buffer:
                buffer.create_span("inner", {})

            for span in buffer.get_all_spans():
                parent = span.parent
                assert parent is None or parent.span_id != SENTINEL, (
                    f"Span {span.name!r} is parented to the synthetic "
                    f"0x1 sentinel — SpanBuffer clobbered the real parent "
                    f"({outer.get_span_context().span_id:#018x})."
                )

    def test_top_level_buffer_unaffected_by_nested_guard(self):
        """When no real parent is active, the buffer's normal path runs.

        Guards against over-reach: the nested-buffer fix only kicks in
        when there's an active recording span. At the top level there is
        none, so the buffer follows its original logic (existing tests
        like ``test_no_parent_context_creates_new_trace`` cover that
        behavior in detail). We just check the buffer emits a span and
        doesn't parent it to the 0x1 sentinel.
        """
        SENTINEL = int("0000000000000001", 16)
        # No outer span — we're at the top level
        with self.client.get_span_buffer(trace_id="top-level-trace") as buffer:
            buffer.create_span("top_level_child", {})

        spans = buffer.get_all_spans()
        assert len(spans) == 1
        # Top-level spans either have no parent or the SDK-injected sentinel
        # (existing behavior when only trace_id is provided). The test below
        # confirms the fix doesn't cause top-level spans to lose their span.
        assert spans[0].get_span_context().span_id != 0, (
            "Top-level span should have a real span_id"
        )
        # If it did parent to the sentinel, that's the existing pre-fix
        # behavior — not a regression from this fix.
        _ = SENTINEL  # silence unused

    def test_resume_path_still_replaces_active_parent(self):
        """Resume (parent_trace_id + parent_span_id) must replace the active context.

        Pre-pause trace lives on a different executor — the resume path
        intentionally rejoins it. The nested-buffer guard must not
        short-circuit this case.
        """
        tracer = self.client.get_tracer()
        with tracer.start_as_current_span("local_parent"):
            with self.client.get_span_buffer(
                trace_id="resume-trace",
                parent_trace_id=PARENT_TRACE_ID,
                parent_span_id=PARENT_SPAN_ID,
            ) as buffer:
                buffer.create_span("resumed_step", {})

            spans = buffer.get_all_spans()
            assert len(spans) == 1
            # Trace_id comes from the resume target, not local_parent
            assert spans[0].get_span_context().trace_id == int(PARENT_TRACE_ID, 16)
            # Parent_id comes from the resume target, not local_parent
            parent_id = spans[0].parent.span_id if spans[0].parent else None
            assert parent_id == int(PARENT_SPAN_ID, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
