# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load/compute estimators for hierarchical KV-cache shadow mode."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from vllm.v1.kv_offload.hierarchical.config import HierarchicalShadowConfig


@dataclass(frozen=True, slots=True)
class RequestHierarchyEstimate:
    request_id: str
    total_blocks: int
    device_hit_blocks: int
    host_hit_blocks: int
    loading_blocks: int
    storing_blocks: int
    absent_blocks: int
    non_loadable_host_blocks: int
    estimated_load_bytes: int
    estimated_load_ms: float
    estimated_compute_ms: float
    load_compute_ratio: float
    has_scheduling_opportunity: bool

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        if math.isinf(self.load_compute_ratio):
            data["load_compute_ratio"] = "inf"
        return data


class HierarchicalLoadComputeEstimator:
    def __init__(self, config: HierarchicalShadowConfig) -> None:
        self.config = config

    def estimate(
        self,
        *,
        request_id: str,
        total_blocks: int,
        device_hit_blocks: int,
        host_hit_blocks: int,
        loading_blocks: int,
        storing_blocks: int,
        absent_blocks: int,
        non_loadable_host_blocks: int,
        compute_tokens: int,
    ) -> RequestHierarchyEstimate:
        estimated_load_bytes = host_hit_blocks * self.config.kv_bytes_per_block
        if self.config.h2d_bandwidth_gbps > 0 and estimated_load_bytes > 0:
            estimated_load_ms = (
                estimated_load_bytes
                / (self.config.h2d_bandwidth_gbps * 1_000_000_000)
                * 1000
            )
        else:
            estimated_load_ms = 0.0

        estimated_compute_ms = max(compute_tokens, 0) * self.config.compute_ms_per_token
        if estimated_compute_ms > 0:
            load_compute_ratio = estimated_load_ms / estimated_compute_ms
        elif estimated_load_ms > 0:
            load_compute_ratio = math.inf
        else:
            load_compute_ratio = 0.0

        has_scheduling_opportunity = (
            host_hit_blocks > 0
            and (
                loading_blocks > 0
                or absent_blocks > 0
                or non_loadable_host_blocks > 0
                or load_compute_ratio >= self.config.opportunity_ratio_threshold
            )
        )

        return RequestHierarchyEstimate(
            request_id=request_id,
            total_blocks=total_blocks,
            device_hit_blocks=device_hit_blocks,
            host_hit_blocks=host_hit_blocks,
            loading_blocks=loading_blocks,
            storing_blocks=storing_blocks,
            absent_blocks=absent_blocks,
            non_loadable_host_blocks=non_loadable_host_blocks,
            estimated_load_bytes=estimated_load_bytes,
            estimated_load_ms=estimated_load_ms,
            estimated_compute_ms=estimated_compute_ms,
            load_compute_ratio=load_compute_ratio,
            has_scheduling_opportunity=has_scheduling_opportunity,
        )

