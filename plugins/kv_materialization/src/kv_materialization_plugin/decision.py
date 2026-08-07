"""Pure decision logic for CPU KV load versus prefix recompute."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

MaterializationMode = Literal["load", "recompute"]


def _finite_nonnegative(value: object) -> bool:
    """Return whether value is a finite non-negative number."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


@dataclass(frozen=True, slots=True)
class MaterializationDecisionConfig:
    """Configuration for the two-way materialization decision."""

    enabled: bool = False
    forced_mode: MaterializationMode | None = None
    fallback_mode: MaterializationMode = "load"
    min_copy_samples: int = 1
    min_recompute_samples: int = 1
    max_observation_age_ms: float = 5000.0

    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        if self.forced_mode not in (None, "load", "recompute"):
            raise ValueError(f"Invalid forced mode: {self.forced_mode!r}")
        if self.fallback_mode not in ("load", "recompute"):
            raise ValueError(f"Invalid fallback mode: {self.fallback_mode!r}")
        if self.min_copy_samples < 0 or self.min_recompute_samples < 0:
            raise ValueError("Minimum sample counts must be non-negative")
        if not _finite_nonnegative(self.max_observation_age_ms):
            raise ValueError("max_observation_age_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MaterializationObservation:
    """Runtime measurements used by the decision function."""

    hit_tokens: int
    hit_blocks: int
    kv_bytes: int = 0
    copy_queue_wait_ms: float | None = None
    copy_service_ms: float | None = None
    copy_bandwidth_bytes_per_ms: float | None = None
    copy_observation_age_ms: float | None = None
    copy_sample_count: int = 0
    recompute_queue_wait_ms: float | None = None
    recompute_service_ms: float | None = None
    recompute_observation_age_ms: float | None = None
    recompute_sample_count: int = 0


@dataclass(frozen=True, slots=True)
class MaterializationDecision:
    """Auditable output of the materialization decision."""

    mode: MaterializationMode
    reason: str
    predicted_load_ms: float | None = None
    predicted_recompute_ms: float | None = None
    fallback: bool = False
    invalid_fields: tuple[str, ...] = ()
    estimate_source: str = "unavailable"


def _fallback(
    config: MaterializationDecisionConfig,
    reason: str,
    invalid_fields: tuple[str, ...] = (),
) -> MaterializationDecision:
    """Build a documented fallback result."""
    return MaterializationDecision(
        mode=config.fallback_mode,
        reason=reason,
        fallback=True,
        invalid_fields=invalid_fields,
    )


def _validate_observation(
    observation: MaterializationObservation,
    config: MaterializationDecisionConfig,
) -> tuple[str, ...]:
    """Return invalid observation fields."""
    invalid: list[str] = []
    if not isinstance(observation.hit_tokens, int) or observation.hit_tokens <= 0:
        invalid.append("hit_tokens")
    if not isinstance(observation.hit_blocks, int) or observation.hit_blocks <= 0:
        invalid.append("hit_blocks")
    if not isinstance(observation.kv_bytes, int) or observation.kv_bytes < 0:
        invalid.append("kv_bytes")

    for field_name in ("copy_queue_wait_ms", "recompute_queue_wait_ms"):
        if not _finite_nonnegative(getattr(observation, field_name)):
            invalid.append(field_name)

    if observation.copy_service_ms is None:
        has_bandwidth = (
            observation.kv_bytes > 0
            and _finite_nonnegative(observation.copy_bandwidth_bytes_per_ms)
            and float(observation.copy_bandwidth_bytes_per_ms) > 0.0
        )
        if not has_bandwidth:
            invalid.append("copy_estimate")
    elif not _finite_nonnegative(observation.copy_service_ms):
        invalid.append("copy_service_ms")

    if not _finite_nonnegative(observation.recompute_service_ms):
        invalid.append("recompute_service_ms")

    if observation.copy_sample_count < config.min_copy_samples:
        invalid.append("copy_sample_count")
    if observation.recompute_sample_count < config.min_recompute_samples:
        invalid.append("recompute_sample_count")

    for field_name in (
        "copy_observation_age_ms",
        "recompute_observation_age_ms",
    ):
        age = getattr(observation, field_name)
        if not _finite_nonnegative(age) or float(age) > config.max_observation_age_ms:
            invalid.append(field_name)
    return tuple(invalid)


def choose_materialization(
    observation: MaterializationObservation,
    config: MaterializationDecisionConfig,
) -> MaterializationDecision:
    """Choose CPU KV load or prefix recompute.

    The function is intentionally device-independent. It never performs cache
    lookup, allocation, synchronization, or device work.
    """
    if observation.hit_tokens <= 0 or observation.hit_blocks <= 0:
        return _fallback(config, "no_complete_cpu_hit")

    if config.forced_mode is not None:
        return MaterializationDecision(
            mode=config.forced_mode,
            reason=f"forced_{config.forced_mode}",
        )

    if not config.enabled:
        return MaterializationDecision(mode=config.fallback_mode, reason="disabled")

    invalid_fields = _validate_observation(observation, config)
    if invalid_fields:
        return _fallback(config, "invalid_or_missing_observation", invalid_fields)

    if observation.copy_service_ms is not None:
        copy_service_ms = float(observation.copy_service_ms)
        estimate_source = "service_time"
    else:
        assert observation.copy_bandwidth_bytes_per_ms is not None
        copy_service_ms = observation.kv_bytes / float(
            observation.copy_bandwidth_bytes_per_ms
        )
        estimate_source = "bandwidth"

    assert observation.copy_queue_wait_ms is not None
    assert observation.recompute_queue_wait_ms is not None
    assert observation.recompute_service_ms is not None
    predicted_load_ms = float(observation.copy_queue_wait_ms) + copy_service_ms
    predicted_recompute_ms = float(observation.recompute_queue_wait_ms) + float(
        observation.recompute_service_ms
    )

    if not all(math.isfinite(x) for x in (predicted_load_ms, predicted_recompute_ms)):
        return _fallback(config, "non_finite_prediction", ("prediction",))

    if predicted_load_ms < predicted_recompute_ms:
        return MaterializationDecision(
            mode="load",
            reason="predicted_load_is_lower",
            predicted_load_ms=predicted_load_ms,
            predicted_recompute_ms=predicted_recompute_ms,
            estimate_source=estimate_source,
        )
    return MaterializationDecision(
        mode="recompute",
        reason="predicted_recompute_is_not_slower",
        predicted_load_ms=predicted_load_ms,
        predicted_recompute_ms=predicted_recompute_ms,
        estimate_source=estimate_source,
    )
