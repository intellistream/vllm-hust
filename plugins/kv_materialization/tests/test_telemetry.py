"""CPU-only tests for the recent telemetry window."""

from __future__ import annotations

import pytest
from kv_materialization_plugin.telemetry import TelemetryWindow


def test_snapshot_uses_recent_median_samples() -> None:
    """Recent samples provide end-to-end estimates for a matching bucket."""
    telemetry = TelemetryWindow(max_samples=4)
    telemetry.observe_load(2, total_ms=5.0, service_ms=4.0, kv_bytes=1000)
    telemetry.observe_load(2, total_ms=9.0, service_ms=6.0, kv_bytes=1200)
    telemetry.observe_recompute(256, total_ms=22.0, service_ms=20.0)

    observation = telemetry.snapshot(256, 2, kv_bytes=1100)

    assert observation.load_sample_count == 2
    assert observation.load_total_ms == pytest.approx(7.0)
    assert observation.recompute_sample_count == 1
    assert observation.recompute_total_ms == pytest.approx(22.0)


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
