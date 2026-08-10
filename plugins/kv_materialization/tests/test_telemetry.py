"""CPU-only tests for the recent telemetry window."""

from __future__ import annotations

import time

import pytest
from kv_materialization_plugin.telemetry import TelemetryWindow


def test_snapshot_uses_recent_median_samples() -> None:
    """Recent samples provide end-to-end estimates for a matching bucket."""
    telemetry = TelemetryWindow(max_samples=4)
    telemetry.observe_load(
        2, total_ms=5.0, service_ms=4.0, kv_bytes=1000, queue_wait_ms=1.0
    )
    telemetry.observe_load(
        2, total_ms=9.0, service_ms=6.0, kv_bytes=1200, queue_wait_ms=3.0
    )
    telemetry.observe_recompute(
        256, total_ms=22.0, service_ms=20.0, queue_wait_ms=2.0
    )

    observation = telemetry.snapshot(256, 2, kv_bytes=1100)

    assert observation.load_sample_count == 2
    assert observation.load_total_ms == pytest.approx(7.0)
    assert observation.load_service_ms == pytest.approx(5.0)
    assert observation.load_queue_wait_ms == pytest.approx(2.0)
    assert observation.load_extra_wait_ms == pytest.approx(2.0)
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


def test_snapshot_rejects_an_invalid_age_limit() -> None:
    """Invalid freshness configuration cannot bypass validation."""
    with pytest.raises(ValueError, match="max_age_ms"):
        TelemetryWindow().snapshot(256, 2, max_age_ms=-1.0)


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
    source.observe_load(2, total_ms=5.0, service_ms=4.0)
    source.observe_recompute(256, total_ms=22.0, service_ms=20.0)
    source.save_json(path)

    restored = TelemetryWindow()
    restored.load_json(path)
    observation = restored.snapshot(256, 2)

    assert observation.load_total_ms == pytest.approx(5.0)
    assert observation.recompute_total_ms == pytest.approx(22.0)
