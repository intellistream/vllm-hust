"""CPU-only tests for the recent telemetry window."""

from __future__ import annotations

import pytest
from kv_materialization_plugin.telemetry import TelemetryWindow


def test_snapshot_uses_recent_median_samples() -> None:
    """Recent samples provide queue and service estimates for a matching bucket."""
    telemetry = TelemetryWindow(max_samples=4)
    telemetry.observe_copy(2, service_ms=4.0, queue_wait_ms=1.0, kv_bytes=1000)
    telemetry.observe_copy(2, service_ms=6.0, queue_wait_ms=3.0, kv_bytes=1200)
    telemetry.observe_recompute(256, service_ms=20.0, queue_wait_ms=2.0)

    observation = telemetry.snapshot(256, 2, kv_bytes=1100)

    assert observation.copy_sample_count == 2
    assert observation.copy_service_ms == pytest.approx(5.0)
    assert observation.copy_queue_wait_ms == pytest.approx(2.0)
    assert observation.recompute_sample_count == 1
    assert observation.recompute_service_ms == pytest.approx(20.0)


def test_snapshot_has_no_estimate_for_empty_buckets() -> None:
    """Cold-start telemetry is explicit and left for the fallback gate."""
    observation = TelemetryWindow().snapshot(256, 2)

    assert observation.copy_service_ms is None
    assert observation.recompute_service_ms is None
    assert observation.copy_sample_count == 0
    assert observation.recompute_sample_count == 0


def test_negative_measurements_are_rejected() -> None:
    """Telemetry cannot silently accept impossible values."""
    telemetry = TelemetryWindow()

    with pytest.raises(ValueError):
        telemetry.observe_copy(2, service_ms=-1.0)
