# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.base import (
    OffloadingManager,
    make_offload_key,
)
from vllm.v1.kv_offload.hierarchical.config import HierarchicalShadowConfig
from vllm.v1.kv_offload.hierarchical.estimator import (
    HierarchicalLoadComputeEstimator,
)
from vllm.v1.kv_offload.hierarchical.metrics import HierarchicalKVCacheMetrics
from vllm.v1.kv_offload.hierarchical.residency import (
    BlockRef,
    ResidencyState,
    ResidencyTracker,
)


def _key(value: int):
    return make_offload_key(str(value).encode(), 0)


def test_residency_tracker_marks_and_removes_keys():
    tracker = ResidencyTracker()
    refs = [BlockRef.from_offload_key(_key(1)), BlockRef.from_offload_key(_key(2))]

    tracker.mark(refs, ResidencyState.READY_HOST)
    assert tracker.snapshot_counts(refs)[ResidencyState.READY_HOST] == 2

    tracker.mark([refs[0]], ResidencyState.LOADING_H2D)
    assert tracker.get(refs[0]) == ResidencyState.LOADING_H2D
    assert tracker.get(refs[1]) == ResidencyState.READY_HOST

    tracker.mark([refs[1]], ResidencyState.ABSENT)
    assert tracker.get(refs[1]) == ResidencyState.ABSENT


def test_estimator_flags_load_compute_opportunity():
    config = HierarchicalShadowConfig(
        enabled=True,
        kv_bytes_per_block=1024,
        h2d_bandwidth_gbps=1.0,
        compute_ms_per_token=0.1,
        opportunity_ratio_threshold=1.0,
    )
    estimate = HierarchicalLoadComputeEstimator(config).estimate(
        request_id="req",
        total_blocks=4,
        device_hit_blocks=1,
        host_hit_blocks=2,
        loading_blocks=0,
        storing_blocks=0,
        absent_blocks=1,
        non_loadable_host_blocks=0,
        compute_tokens=1,
    )

    assert estimate.estimated_load_bytes == 2048
    assert estimate.estimated_load_ms > 0
    assert estimate.load_compute_ratio > 0
    assert estimate.has_scheduling_opportunity


def test_metrics_writes_jsonl(tmp_path):
    metrics_path = tmp_path / "hier.jsonl"
    config = HierarchicalShadowConfig(
        enabled=True,
        metrics_path=str(metrics_path),
        kv_bytes_per_block=1024,
        h2d_bandwidth_gbps=1.0,
        compute_ms_per_token=0.1,
    )
    metrics = HierarchicalKVCacheMetrics(config)
    estimate = metrics.estimator.estimate(
        request_id="req",
        total_blocks=2,
        device_hit_blocks=1,
        host_hit_blocks=1,
        loading_blocks=0,
        storing_blocks=0,
        absent_blocks=0,
        non_loadable_host_blocks=0,
        compute_tokens=1,
    )

    metrics.record_estimate(estimate, extra={"num_tokens": 8})

    record = json.loads(metrics_path.read_text().strip())
    assert record["type"] == "hierarchical_kv_shadow"
    assert record["request_id"] == "req"
    assert record["host_hit_blocks"] == 1
    assert record["num_tokens"] == 8
    assert metrics.snapshot().records == 1


class _FakeSpec:
    def __init__(self, metrics_path: str):
        self.extra_config = {
            "hierarchical_shadow_mode": True,
            "hierarchical_metrics_path": metrics_path,
            "hierarchical_kv_bytes_per_block": 1024,
            "hierarchical_h2d_bandwidth_gbps": 1.0,
            "hierarchical_compute_ms_per_token": 0.1,
        }
        self.block_size_factor = 1
        self.hash_block_size = 4
        self.gpu_block_size = (4,)
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=1),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
        )
        self.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(
                        block_size=4,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    )
                )
            ],
        )
        self.manager = MagicMock(spec=OffloadingManager)
        self.manager.lookup.return_value = True
        self.manager.touch.return_value = None

    def get_manager(self):
        return self.manager


def test_offloading_scheduler_shadow_records_lookup_without_changing_result(tmp_path):
    scheduler = OffloadingConnectorScheduler(_FakeSpec(str(tmp_path / "hier.jsonl")))
    request = SimpleNamespace(
        request_id="req",
        num_tokens=16,
        block_hashes=[str(i).encode() for i in range(4)],
        kv_transfer_params=None,
    )

    matched_tokens, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=4
    )

    assert matched_tokens == 12
    assert load_async

    summary = scheduler.hierarchical_metrics.snapshot()
    assert summary.records == 1
    assert summary.total_blocks == 4
    assert summary.device_hit_blocks == 1
    assert summary.host_hit_blocks == 3

    record = json.loads((tmp_path / "hier.jsonl").read_text().strip())
    assert record["device_hit_blocks"] == 1
    assert record["host_hit_blocks"] == 3
    assert record["absent_blocks"] == 0

    scheduler.manager.lookup.assert_called()
    scheduler.manager.touch.assert_called()


def test_offloading_scheduler_shadow_records_deferred_lookup(tmp_path):
    scheduler = OffloadingConnectorScheduler(_FakeSpec(str(tmp_path / "hier.jsonl")))
    scheduler.manager.lookup.return_value = None
    request = SimpleNamespace(
        request_id="req",
        num_tokens=8,
        block_hashes=[b"0", b"1"],
        kv_transfer_params=None,
    )

    matched_tokens, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert matched_tokens is None
    assert not load_async
    record = json.loads((tmp_path / "hier.jsonl").read_text().strip())
    assert record["lookup_deferred"]
