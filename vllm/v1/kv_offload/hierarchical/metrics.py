# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics recorder for hierarchical KV-cache shadow mode."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any

from vllm.v1.kv_offload.hierarchical.config import HierarchicalShadowConfig
from vllm.v1.kv_offload.hierarchical.estimator import (
    HierarchicalLoadComputeEstimator,
    RequestHierarchyEstimate,
)
from vllm.v1.kv_offload.hierarchical.residency import ResidencyTracker


@dataclass(slots=True)
class HierarchicalMetricsSummary:
    records: int = 0
    total_blocks: int = 0
    device_hit_blocks: int = 0
    host_hit_blocks: int = 0
    loading_blocks: int = 0
    storing_blocks: int = 0
    absent_blocks: int = 0
    opportunity_records: int = 0


class HierarchicalKVCacheMetrics:
    def __init__(
        self,
        config: HierarchicalShadowConfig,
        *,
        residency: ResidencyTracker | None = None,
        estimator: HierarchicalLoadComputeEstimator | None = None,
    ) -> None:
        self.config = config
        self.residency = residency or ResidencyTracker()
        self.estimator = estimator or HierarchicalLoadComputeEstimator(config)
        self.summary = HierarchicalMetricsSummary()

    def record_estimate(
        self,
        estimate: RequestHierarchyEstimate,
        *,
        event: str = "request_estimate",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.summary.records += 1
        self.summary.total_blocks += estimate.total_blocks
        self.summary.device_hit_blocks += estimate.device_hit_blocks
        self.summary.host_hit_blocks += estimate.host_hit_blocks
        self.summary.loading_blocks += estimate.loading_blocks
        self.summary.storing_blocks += estimate.storing_blocks
        self.summary.absent_blocks += estimate.absent_blocks
        if estimate.has_scheduling_opportunity:
            self.summary.opportunity_records += 1

        if self.config.metrics_path is None:
            return

        record: dict[str, Any] = {
            "type": "hierarchical_kv_shadow",
            "event": event,
            "timestamp": time.monotonic(),
            **estimate.to_json(),
        }
        if extra:
            record.update(extra)

        parent = os.path.dirname(self.config.metrics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.config.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def snapshot(self) -> HierarchicalMetricsSummary:
        return replace(self.summary)
