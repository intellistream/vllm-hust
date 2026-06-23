# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for hierarchical KV-cache shadow metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _get_config_value(
    extra_config: dict[str, Any],
    key: str,
    env_key: str,
    default: Any,
) -> Any:
    if key in extra_config:
        return extra_config[key]
    return os.getenv(env_key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


@dataclass(frozen=True, slots=True)
class HierarchicalShadowConfig:
    """Configuration for cache-observability-only shadow mode.

    Shadow mode must not change request scheduling. It only records the
    scheduler-visible residency mix and a lightweight load/compute estimate.
    Values can be supplied through ``kv_connector_extra_config`` using the
    ``hierarchical_*`` keys, or through the matching environment variables.
    """

    enabled: bool = False
    metrics_path: str | None = None
    kv_bytes_per_block: int = 0
    h2d_bandwidth_gbps: float = 0.0
    compute_ms_per_token: float = 0.0
    opportunity_ratio_threshold: float = 1.0

    @classmethod
    def from_extra_config(
        cls,
        extra_config: dict[str, Any],
        *,
        default_kv_bytes_per_block: int = 0,
    ) -> HierarchicalShadowConfig:
        return cls(
            enabled=_as_bool(
                _get_config_value(
                    extra_config,
                    "hierarchical_shadow_mode",
                    "VLLM_HIERARCHICAL_KV_SHADOW_MODE",
                    False,
                )
            ),
            metrics_path=_get_config_value(
                extra_config,
                "hierarchical_metrics_path",
                "VLLM_HIERARCHICAL_KV_METRICS_PATH",
                None,
            )
            or None,
            kv_bytes_per_block=_as_int(
                _get_config_value(
                    extra_config,
                    "hierarchical_kv_bytes_per_block",
                    "VLLM_HIERARCHICAL_KV_BYTES_PER_BLOCK",
                    default_kv_bytes_per_block,
                ),
                default_kv_bytes_per_block,
            ),
            h2d_bandwidth_gbps=_as_float(
                _get_config_value(
                    extra_config,
                    "hierarchical_h2d_bandwidth_gbps",
                    "VLLM_HIERARCHICAL_KV_H2D_BANDWIDTH_GBPS",
                    0.0,
                )
            ),
            compute_ms_per_token=_as_float(
                _get_config_value(
                    extra_config,
                    "hierarchical_compute_ms_per_token",
                    "VLLM_HIERARCHICAL_KV_COMPUTE_MS_PER_TOKEN",
                    0.0,
                )
            ),
            opportunity_ratio_threshold=_as_float(
                _get_config_value(
                    extra_config,
                    "hierarchical_opportunity_ratio_threshold",
                    "VLLM_HIERARCHICAL_KV_OPPORTUNITY_RATIO_THRESHOLD",
                    1.0,
                ),
                1.0,
            ),
        )
