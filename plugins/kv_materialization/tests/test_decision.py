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
        "load_total_ms": 5.0,
        "load_queue_wait_ms": 1.0,
        "load_observation_age_ms": 10.0,
        "load_sample_count": 3,
        "recompute_total_ms": 22.0,
        "recompute_queue_wait_ms": 2.0,
        "recompute_observation_age_ms": 10.0,
        "recompute_sample_count": 3,
    }
    values.update(overrides)
    return MaterializationObservation(**values)  # type: ignore[arg-type]


def test_dynamic_load_uses_end_to_end_time() -> None:
    """Load wins when its recent end-to-end cost is lower."""
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
            load_total_ms=22.0,
        ),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "recompute"
    assert decision.reason == "predicted_recompute_is_not_slower"


def test_overlapping_materialization_falls_back() -> None:
    """M1 does not reuse a single-request estimate during overlap."""
    decision = choose_materialization(
        make_observation(active_materialization_count=1),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "load"
    assert decision.fallback is True
    assert decision.reason == "unsupported_concurrent_context"
    assert decision.invalid_fields == ("active_materialization_count",)


def test_only_first_overlapping_request_uses_dynamic_estimate() -> None:
    """Later overlap cannot repeat the first request's recompute decision."""
    config = MaterializationDecisionConfig(enabled=True)
    first = choose_materialization(
        make_observation(load_total_ms=30.0, recompute_total_ms=10.0),
        config,
    )
    second = choose_materialization(
        make_observation(
            load_total_ms=30.0,
            recompute_total_ms=10.0,
            active_materialization_count=1,
        ),
        config,
    )

    assert first.mode == "recompute"
    assert first.fallback is False
    assert second.mode == "load"
    assert second.fallback is True


@pytest.mark.parametrize("forced_mode", ["load", "recompute"])
def test_forced_mode_does_not_require_observations(forced_mode: str) -> None:
    """Forced modes remain usable during cold start."""
    decision = choose_materialization(
        make_observation(
            load_total_ms=None,
            load_observation_age_ms=None,
            recompute_total_ms=None,
            recompute_observation_age_ms=None,
            active_materialization_count=10,
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
            load_total_ms=None,
            load_observation_age_ms=None,
            recompute_total_ms=None,
            recompute_observation_age_ms=None,
        ),
        MaterializationDecisionConfig(),
    )

    assert decision.mode == "load"
    assert decision.reason == "disabled"


@pytest.mark.parametrize(
    "field, value",
    [
        ("load_sample_count", 0),
        ("recompute_sample_count", 0),
        ("load_observation_age_ms", 6000.0),
        ("recompute_total_ms", math.nan),
        ("load_total_ms", -1.0),
        ("load_queue_wait_ms", -1.0),
        ("active_materialization_count", -1),
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


def test_end_to_end_prediction_is_recorded() -> None:
    """The decision exposes the end-to-end estimate source."""
    decision = choose_materialization(
        make_observation(
            load_total_ms=11.24,
        ),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "load"
    assert decision.estimate_source == "end_to_end_median"
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


def test_extreme_prefix_length_is_valid_when_observations_are_explicit() -> None:
    """A large complete prefix is still a normal two-way decision input."""
    decision = choose_materialization(
        make_observation(
            hit_tokens=1_048_576,
            hit_blocks=8192,
            kv_bytes=1 << 40,
            load_total_ms=900.0,
            recompute_total_ms=1200.0,
        ),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "load"
    assert decision.fallback is False


def test_missing_phase_queue_observation_is_a_confidence_fallback() -> None:
    """Old or incomplete calibration cannot silently drive dynamic choice."""
    decision = choose_materialization(
        make_observation(load_queue_wait_ms=None),
        MaterializationDecisionConfig(enabled=True),
    )

    assert decision.mode == "load"
    assert decision.reason == "insufficient_observation_confidence"
    assert decision.confidence_guard == "recent_samples_and_phase_timestamps"
