"""CPU-only tests for the dynamic materialization decision."""

from __future__ import annotations

import math

import pytest
from kv_materialization_plugin.decision import (
    MaterializationDecisionConfig,
    MaterializationObservation,
    choose_materialization,
)


def make_observation(**overrides: object) -> MaterializationObservation:
    """Build a valid observation for decision tests."""
    values: dict[str, object] = {
        "hit_tokens": 256,
        "hit_blocks": 2,
        "kv_bytes": 1024,
        "copy_queue_wait_ms": 1.0,
        "copy_service_ms": 4.0,
        "copy_observation_age_ms": 10.0,
        "copy_sample_count": 3,
        "recompute_queue_wait_ms": 2.0,
        "recompute_service_ms": 20.0,
        "recompute_observation_age_ms": 10.0,
        "recompute_sample_count": 3,
    }
    values.update(overrides)
    return MaterializationObservation(**values)  # type: ignore[arg-type]


def test_dynamic_load_uses_queue_and_service_time() -> None:
    """Load wins only after both queue and service costs are included."""
    decision = choose_materialization(
        make_observation(),
        MaterializationDecisionConfig(
            enabled=True,
            min_copy_samples=2,
            min_recompute_samples=2,
        ),
    )

    assert decision.mode == "load"
    assert decision.predicted_load_ms == pytest.approx(5.0)
    assert decision.predicted_recompute_ms == pytest.approx(22.0)
    assert decision.fallback is False


def test_equal_prediction_deterministically_chooses_recompute() -> None:
    """Equal predictions use one deterministic side of the boundary."""
    decision = choose_materialization(
        make_observation(
            copy_queue_wait_ms=2.0,
            copy_service_ms=20.0,
        ),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "recompute"
    assert decision.reason == "predicted_recompute_is_not_slower"


@pytest.mark.parametrize("forced_mode", ["load", "recompute"])
def test_forced_mode_does_not_require_observations(forced_mode: str) -> None:
    """Forced modes remain usable during cold start."""
    decision = choose_materialization(
        make_observation(
            copy_queue_wait_ms=None,
            copy_service_ms=None,
            copy_observation_age_ms=None,
            recompute_queue_wait_ms=None,
            recompute_service_ms=None,
            recompute_observation_age_ms=None,
        ),
        MaterializationDecisionConfig(
            forced_mode=forced_mode  # type: ignore[arg-type]
        ),
    )

    assert decision.mode == forced_mode
    assert decision.fallback is False


def test_disabled_dynamic_preserves_load_fallback() -> None:
    """The default configuration retains the native load behavior."""
    decision = choose_materialization(
        make_observation(
            copy_queue_wait_ms=None,
            copy_service_ms=None,
            copy_observation_age_ms=None,
            recompute_queue_wait_ms=None,
            recompute_service_ms=None,
            recompute_observation_age_ms=None,
        ),
        MaterializationDecisionConfig(),
    )

    assert decision.mode == "load"
    assert decision.reason == "disabled"


@pytest.mark.parametrize(
    "field, value",
    [
        ("copy_sample_count", 0),
        ("recompute_sample_count", 0),
        ("copy_observation_age_ms", 6000.0),
        ("recompute_service_ms", math.nan),
        ("copy_queue_wait_ms", -1.0),
    ],
)
def test_invalid_or_stale_observation_falls_back(field: str, value: object) -> None:
    """Invalid runtime telemetry never raises or makes a dynamic choice."""
    decision = choose_materialization(
        make_observation(**{field: value}),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "load"
    assert decision.fallback is True
    assert decision.invalid_fields


def test_bandwidth_is_used_when_direct_copy_service_is_missing() -> None:
    """Bandwidth is an explicit fallback estimator, not a nominal constant."""
    decision = choose_materialization(
        make_observation(
            copy_service_ms=None,
            copy_bandwidth_bytes_per_ms=100.0,
        ),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "load"
    assert decision.estimate_source == "bandwidth"
    assert decision.predicted_load_ms == pytest.approx(11.24)


def test_no_hit_does_not_enter_materialization_decision() -> None:
    """A zero hit is represented as a fallback/no-op decision."""
    decision = choose_materialization(
        make_observation(hit_tokens=0, hit_blocks=0),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.fallback is True
    assert decision.reason == "no_complete_cpu_hit"


def test_invalid_config_is_rejected() -> None:
    """Configuration errors fail before runtime decisions begin."""
    with pytest.raises(ValueError, match="fallback mode"):
        MaterializationDecisionConfig(fallback_mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_observation_age_ms"):
        MaterializationDecisionConfig(max_observation_age_ms=-1.0)
