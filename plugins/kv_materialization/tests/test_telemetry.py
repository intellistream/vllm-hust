"""CPU-only tests for the recent telemetry window."""

from __future__ import annotations

import time

import pytest
from kv_materialization_plugin.decision import (
    MaterializationDecisionConfig,
    choose_materialization,
)
from kv_materialization_plugin.telemetry import TelemetryWindow


def test_snapshot_uses_recent_mean_samples() -> None:
    """Recent samples provide mean path-completion estimates for a bucket."""
    telemetry = TelemetryWindow(max_samples=4)
    telemetry.observe_load(
        2, total_ms=5.0, service_ms=4.0, kv_bytes=1000, queue_wait_ms=1.0
    )
    telemetry.observe_load(
        2, total_ms=9.0, service_ms=6.0, kv_bytes=1200, queue_wait_ms=3.0
    )
    telemetry.observe_load(
        2, total_ms=16.0, service_ms=8.0, kv_bytes=1300, queue_wait_ms=8.0
    )
    telemetry.observe_recompute(
        256, total_ms=22.0, service_ms=20.0, queue_wait_ms=2.0
    )

    observation = telemetry.snapshot(256, 2, kv_bytes=1100)

    assert observation.load_sample_count == 3
    assert observation.load_total_ms == pytest.approx(10.0)
    assert observation.load_service_ms == pytest.approx(6.0)
    assert observation.load_queue_wait_ms == pytest.approx(4.0)
    assert observation.load_extra_wait_ms == pytest.approx(4.0)
    assert observation.recompute_sample_count == 1
    assert observation.recompute_total_ms == pytest.approx(22.0)
    assert observation.recompute_service_ms == pytest.approx(20.0)
    assert observation.recompute_queue_wait_ms == pytest.approx(2.0)
    assert observation.recompute_extra_wait_ms == pytest.approx(2.0)


def test_snapshot_does_not_reuse_a_different_size_bucket() -> None:
    """Measurements for another prefix size cannot drive a decision."""
    telemetry = TelemetryWindow()
    telemetry.observe_load(2, total_ms=5.0, service_ms=4.0)
    telemetry.observe_recompute(256, total_ms=22.0, service_ms=20.0)

    observation = telemetry.snapshot(512, 4)

    assert observation.load_total_ms is None
    assert observation.recompute_total_ms is None
    assert observation.load_sample_count == 0
    assert observation.recompute_sample_count == 0


def test_same_load_block_bucket_does_not_reuse_recompute_tokens() -> None:
    """Load and recompute retain their actual, different size keys."""
    telemetry = TelemetryWindow()
    telemetry.observe_load(2, total_ms=5.0, service_ms=4.0, queue_wait_ms=1.0)
    telemetry.observe_recompute(
        256, total_ms=22.0, service_ms=20.0, queue_wait_ms=2.0
    )

    observation = telemetry.snapshot(300, 2)

    assert observation.load_total_ms == pytest.approx(5.0)
    assert observation.load_sample_count == 1
    assert observation.recompute_total_ms is None
    assert observation.recompute_sample_count == 0


def test_same_recompute_token_bucket_does_not_reuse_load_blocks() -> None:
    """A recompute token match cannot fill a missing load-size bucket."""
    telemetry = TelemetryWindow()
    telemetry.observe_load(2, total_ms=5.0, service_ms=4.0, queue_wait_ms=1.0)
    telemetry.observe_recompute(
        256, total_ms=22.0, service_ms=20.0, queue_wait_ms=2.0
    )

    observation = telemetry.snapshot(256, 3)

    assert observation.load_total_ms is None
    assert observation.load_sample_count == 0
    assert observation.recompute_total_ms == pytest.approx(22.0)
    assert observation.recompute_sample_count == 1


def test_snapshot_aggregates_admission_positions_in_the_matched_workload() -> None:
    """The estimator captures the workload instead of one greedy queue depth."""
    telemetry = TelemetryWindow()
    telemetry.observe_load(
        2, 5.0, 4.0, queue_wait_ms=1.0, active_path_count=0
    )
    telemetry.observe_load(
        2, 35.0, 10.0, queue_wait_ms=25.0, active_path_count=3
    )
    telemetry.observe_recompute(
        256, 22.0, 20.0, queue_wait_ms=2.0, active_path_count=0
    )
    telemetry.observe_recompute(
        256, 18.0, 16.0, queue_wait_ms=2.0, active_path_count=1
    )

    observation = telemetry.snapshot(
        256,
        2,
        active_load_count=3,
        active_recompute_count=1,
    )

    assert observation.active_materialization_count == 4
    assert observation.active_load_count == 3
    assert observation.active_recompute_count == 1
    assert observation.load_total_ms == pytest.approx(20.0)
    assert observation.recompute_total_ms == pytest.approx(20.0)


def test_snapshot_keeps_active_counts_as_diagnostics() -> None:
    """Active counts are audited but do not fragment the workload estimate."""
    telemetry = TelemetryWindow()
    telemetry.observe_load(2, 5.0, 4.0, queue_wait_ms=1.0)
    telemetry.observe_recompute(256, 22.0, 20.0, queue_wait_ms=2.0)

    observation = telemetry.snapshot(
        256,
        2,
        active_load_count=2,
        active_recompute_count=1,
    )

    assert observation.active_materialization_count == 3
    assert observation.load_total_ms == pytest.approx(5.0)
    assert observation.recompute_total_ms == pytest.approx(22.0)
    assert observation.load_sample_count == 1
    assert observation.recompute_sample_count == 1


def test_many_overlapping_requests_can_keep_choosing_recompute() -> None:
    """A lower workload-wide recompute mean applies beyond the first request."""
    telemetry = TelemetryWindow()
    for depth in range(8):
        for _ in range(3):
            telemetry.observe_load(
                2,
                200.0 + depth * 5.0,
                180.0,
                queue_wait_ms=20.0 + depth * 5.0,
                active_path_count=depth,
            )
            telemetry.observe_recompute(
                256,
                150.0 + depth * 2.0,
                140.0,
                queue_wait_ms=10.0 + depth * 2.0,
                active_path_count=depth,
            )

    active_load_count = 0
    active_recompute_count = 0
    decisions: list[str] = []
    config = MaterializationDecisionConfig(
        enabled=True,
        min_copy_samples=3,
        min_recompute_samples=3,
    )
    for _ in range(8):
        decision = choose_materialization(
            telemetry.snapshot(
                256,
                2,
                active_load_count=active_load_count,
                active_recompute_count=active_recompute_count,
            ),
            config,
        )
        decisions.append(decision.mode)
        active_load_count += decision.mode == "load"
        active_recompute_count += decision.mode == "recompute"

    assert decisions == ["recompute"] * 8
    assert active_recompute_count == 8


def test_many_overlapping_requests_can_keep_choosing_load() -> None:
    """A lower workload-wide load mean applies at every admission position."""
    telemetry = TelemetryWindow()
    for depth in range(8):
        telemetry.observe_load(
            2,
            20.0 + depth,
            18.0,
            queue_wait_ms=2.0 + depth,
            active_path_count=depth,
        )
        telemetry.observe_recompute(
            256,
            50.0 + depth,
            45.0,
            queue_wait_ms=5.0 + depth,
            active_path_count=depth,
        )

    config = MaterializationDecisionConfig(enabled=True)
    decisions = [
        choose_materialization(
            telemetry.snapshot(
                256,
                2,
                active_load_count=depth,
                active_recompute_count=0,
            ),
            config,
        ).mode
        for depth in range(8)
    ]

    assert decisions == ["load"] * 8


def test_workload_mean_avoids_greedy_low_depth_load_choices() -> None:
    """A cheap first load cannot override worse load behavior for the batch."""
    telemetry = TelemetryWindow()
    for depth, total_ms in enumerate((50.0, 150.0, 160.0)):
        telemetry.observe_load(
            2,
            total_ms,
            total_ms - 1.0,
            queue_wait_ms=1.0,
            active_path_count=depth,
        )
    for depth, total_ms in enumerate((90.0, 100.0, 110.0)):
        telemetry.observe_recompute(
            256,
            total_ms,
            total_ms - 1.0,
            queue_wait_ms=1.0,
            active_path_count=depth,
        )

    active_load_count = 0
    active_recompute_count = 0
    decisions: list[str] = []
    config = MaterializationDecisionConfig(enabled=True)
    for _ in range(5):
        decision = choose_materialization(
            telemetry.snapshot(
                256,
                2,
                active_load_count=active_load_count,
                active_recompute_count=active_recompute_count,
            ),
            config,
        )
        decisions.append(decision.mode)
        active_load_count += decision.mode == "load"
        active_recompute_count += decision.mode == "recompute"

    assert decisions == ["recompute"] * 5


def test_snapshot_excludes_stale_samples_before_the_median() -> None:
    """One fresh sample cannot make expired history look valid."""
    now = time.time()
    telemetry = TelemetryWindow()
    telemetry.observe_load(2, 100.0, 90.0, timestamp=now - 10.0)
    telemetry.observe_load(2, 5.0, 4.0, timestamp=now)
    telemetry.observe_recompute(256, 22.0, 20.0, timestamp=now)

    observation = telemetry.snapshot(256, 2, max_age_ms=1000.0)

    assert observation.load_sample_count == 1
    assert observation.load_total_ms == pytest.approx(5.0)


def test_cached_stats_are_invalidated_by_a_new_sample() -> None:
    telemetry = TelemetryWindow()
    telemetry.observe_load(2, 5.0, 4.0, queue_wait_ms=1.0)
    assert telemetry.snapshot(256, 2).load_total_ms == pytest.approx(5.0)

    telemetry.observe_load(2, 15.0, 14.0, queue_wait_ms=1.0)

    assert telemetry.snapshot(256, 2).load_total_ms == pytest.approx(10.0)


def test_bucket_window_evicts_the_oldest_samples() -> None:
    """Each exact-size bucket retains only the configured recent window."""
    telemetry = TelemetryWindow(max_samples=2)
    telemetry.observe_load(2, 100.0, 90.0, queue_wait_ms=10.0)
    telemetry.observe_load(2, 20.0, 18.0, queue_wait_ms=2.0)
    telemetry.observe_load(2, 10.0, 9.0, queue_wait_ms=1.0)

    observation = telemetry.snapshot(256, 2)

    assert observation.load_sample_count == 2
    assert observation.load_total_ms == pytest.approx(15.0)


def test_mixed_sizes_remain_independent_during_overlap() -> None:
    """Concurrent diagnostics do not merge distinct exact-size buckets."""
    telemetry = TelemetryWindow()
    telemetry.observe_load(2, 5.0, 4.0, queue_wait_ms=1.0)
    telemetry.observe_load(4, 15.0, 13.0, queue_wait_ms=2.0)
    telemetry.observe_recompute(256, 22.0, 20.0, queue_wait_ms=2.0)
    telemetry.observe_recompute(512, 12.0, 10.0, queue_wait_ms=2.0)

    short = telemetry.snapshot(
        256,
        2,
        active_load_count=1,
        active_recompute_count=1,
    )
    long = telemetry.snapshot(
        512,
        4,
        active_load_count=1,
        active_recompute_count=1,
    )

    assert choose_materialization(
        short, MaterializationDecisionConfig(enabled=True)
    ).mode == "load"
    assert choose_materialization(
        long, MaterializationDecisionConfig(enabled=True)
    ).mode == "recompute"


def test_cached_stats_expire_at_the_sample_freshness_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = TelemetryWindow()
    telemetry.observe_load(
        2,
        5.0,
        4.0,
        queue_wait_ms=1.0,
        timestamp=100.0,
    )
    monkeypatch.setattr(
        "kv_materialization_plugin.telemetry.time.time", lambda: 100.5
    )
    assert telemetry.snapshot(
        256, 2, max_age_ms=1000.0
    ).load_sample_count == 1

    monkeypatch.setattr(
        "kv_materialization_plugin.telemetry.time.time", lambda: 101.1
    )
    assert telemetry.snapshot(
        256, 2, max_age_ms=1000.0
    ).load_sample_count == 0


def test_snapshot_rejects_an_invalid_age_limit() -> None:
    """Invalid freshness configuration cannot bypass validation."""
    with pytest.raises(ValueError, match="max_age_ms"):
        TelemetryWindow().snapshot(256, 2, max_age_ms=-1.0)

    with pytest.raises(ValueError, match="active_load_count"):
        TelemetryWindow().snapshot(256, 2, active_load_count=-1)


def test_snapshot_has_no_estimate_for_empty_buckets() -> None:
    """Cold-start telemetry is explicit and left for the fallback gate."""
    observation = TelemetryWindow().snapshot(256, 2)

    assert observation.load_total_ms is None
    assert observation.recompute_total_ms is None
    assert observation.load_sample_count == 0
    assert observation.recompute_sample_count == 0


def test_negative_measurements_are_rejected() -> None:
    """Telemetry cannot silently accept impossible values."""
    telemetry = TelemetryWindow()

    with pytest.raises(ValueError):
        telemetry.observe_load(2, total_ms=1.0, service_ms=-1.0)

    with pytest.raises(ValueError):
        telemetry.observe_load(True, total_ms=1.0, service_ms=1.0)


def test_calibration_state_round_trips(tmp_path) -> None:
    """Forced-mode telemetry can be loaded by a later dynamic run."""
    path = tmp_path / "calibration.json"
    source = TelemetryWindow()
    source.observe_load(
        2, total_ms=5.0, service_ms=4.0, active_path_count=3
    )
    source.observe_recompute(
        256, total_ms=22.0, service_ms=20.0, active_path_count=1
    )
    source.save_json(path)

    restored = TelemetryWindow()
    restored.load_json(path)
    observation = restored.snapshot(
        256,
        2,
        active_load_count=3,
        active_recompute_count=1,
    )

    assert observation.load_total_ms == pytest.approx(5.0)
    assert observation.recompute_total_ms == pytest.approx(22.0)


def test_old_calibration_json_remains_readable(tmp_path) -> None:
    """Calibration written before active-path diagnostics remains readable."""
    path = tmp_path / "old-calibration.json"
    path.write_text(
        '{"load":[{"size":2,"total_ms":5.0,"service_ms":4.0,'
        '"extra_wait_ms":1.0,"timestamp":1.0,"kv_bytes":0,'
        '"queue_wait_ms":1.0}],"recompute":[]}',
        encoding="utf-8",
    )
    telemetry = TelemetryWindow()
    telemetry.load_json(path)

    assert telemetry.snapshot(256, 2).load_total_ms == pytest.approx(5.0)
    assert telemetry.snapshot(
        256, 2, active_load_count=1
    ).load_total_ms == pytest.approx(5.0)
