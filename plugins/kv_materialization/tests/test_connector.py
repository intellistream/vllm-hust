"""Non-device lifecycle tests for the external connector."""

from __future__ import annotations

import pytest
from kv_materialization_plugin.audit import AuditLog
from kv_materialization_plugin.connector import (
    DynamicSimpleCPUOffloadConnector,
    _advance_recompute_progress,
)
from kv_materialization_plugin.decision import (
    MaterializationDecision,
    MaterializationObservation,
)
from kv_materialization_plugin.metadata import TimingSampleMetadata
from kv_materialization_plugin.telemetry import TelemetryWindow


def test_recompute_progress_waits_for_the_complete_cpu_hit() -> None:
    """Chunked prefill cannot report completion after only its first step."""
    remaining, consumed, completed = _advance_recompute_progress(4096, 1024)
    assert (remaining, consumed, completed) == (3072, 1024, False)

    remaining, consumed, completed = _advance_recompute_progress(remaining, 4096)
    assert (remaining, consumed, completed) == (0, 3072, True)


def test_recompute_progress_ignores_tokens_after_the_hit_prefix() -> None:
    """Uncached suffix/output work is not charged as prefix recomputation."""
    assert _advance_recompute_progress(256, 300) == (0, 256, True)


def test_recompute_progress_rejects_invalid_counts() -> None:
    """Impossible scheduler progress cannot silently corrupt telemetry."""
    with pytest.raises(ValueError, match="non-negative"):
        _advance_recompute_progress(256, -1)


def test_request_completion_releases_active_materialization_state() -> None:
    """A completed request cannot keep later dynamic decisions in fallback."""
    connector = object.__new__(DynamicSimpleCPUOffloadConnector)
    connector._decisions = {"req-1": object()}
    connector._decision_hit_tokens = {"req-1": 256}
    connector._decision_times = {"req-1": 1.0}
    connector._recompute_remaining_tokens = {"req-1": 256}
    connector._new_recompute_attempts = {"req-1"}

    connector._clear_request_state("req-1")

    assert connector._decisions == {}
    assert connector._decision_hit_tokens == {}
    assert connector._decision_times == {}
    assert connector._recompute_remaining_tokens == {}
    assert connector._new_recompute_attempts == set()


def test_worker_sample_returns_actual_cost_to_scheduler_telemetry() -> None:
    """The non-device transport closes the scheduler-to-worker loop."""
    connector = object.__new__(DynamicSimpleCPUOffloadConnector)
    connector._decision_times = {"req-1": 10.0}
    connector._decisions = {
        "req-1": MaterializationDecision(
            mode="load",
            reason="predicted_load_is_lower",
            predicted_load_ms=5.0,
            predicted_recompute_ms=20.0,
        )
    }
    connector._decision_hit_tokens = {"req-1": 256}
    connector._recompute_remaining_tokens = {}
    connector._new_recompute_attempts = set()
    connector._recompute_queue_wait_ms = {}
    connector._telemetry = TelemetryWindow()
    connector._audit = AuditLog()
    connector._audit.start(
        "req-1",
        256,
        2,
        connector._decisions["req-1"],
        observation=MaterializationObservation(
            hit_tokens=256,
            hit_blocks=2,
            load_total_ms=5.0,
            load_service_ms=4.0,
            load_queue_wait_ms=1.0,
            load_observation_age_ms=1.0,
            load_sample_count=1,
            recompute_total_ms=20.0,
            recompute_service_ms=18.0,
            recompute_queue_wait_ms=2.0,
            recompute_observation_age_ms=1.0,
            recompute_sample_count=1,
        ),
    )

    connector._consume_samples(
        [TimingSampleMetadata("req-1", 2, 4.0, 1024, 1.0)],
        "cpu_kv_load",
        is_load=True,
    )

    record = connector._audit.records()[0]
    assert record.actual_branch == "cpu_kv_load"
    assert record.actual_cost_ms is not None
    assert record.queue_wait_ms == pytest.approx(1.0)
    observation = connector._telemetry.snapshot(256, 2)
    assert observation.load_service_ms == pytest.approx(4.0)
    assert observation.load_queue_wait_ms == pytest.approx(1.0)
