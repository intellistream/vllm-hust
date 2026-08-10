"""Non-device tests for scheduler/worker metadata transport."""

from __future__ import annotations

import pytest
from kv_materialization_plugin.metadata import (
    DynamicCPUOffloadMetadata,
    DynamicCPUOffloadWorkerMetadata,
    TimingSampleMetadata,
    as_worker_metadata,
)

from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)


def test_dynamic_metadata_preserves_native_transfer_fields() -> None:
    """Plugin metadata adds observations without changing native transfer data."""
    base = SimpleCPUOffloadMetadata(
        load_event=7,
        load_gpu_blocks=[1, 2],
        load_cpu_blocks=[11, 12],
        load_event_to_reqs={7: ["load-1"]},
        store_event=8,
        store_gpu_blocks=[3],
        store_cpu_blocks=[13],
        need_flush=True,
    )

    dynamic = DynamicCPUOffloadMetadata.from_base(
        base,
        recompute_requests={"recompute-1": 256},
        reset_recompute_requests={"recompute-1"},
        completed_recompute_requests={"recompute-1"},
        load_block_counts={"load-1": 2},
        decision_times={"load-1": 10.0, "recompute-1": 10.0},
    )

    assert dynamic.load_event == base.load_event
    assert dynamic.load_gpu_blocks == base.load_gpu_blocks
    assert dynamic.load_cpu_blocks == base.load_cpu_blocks
    assert dynamic.load_event_to_reqs == base.load_event_to_reqs
    assert dynamic.store_event == base.store_event
    assert dynamic.store_gpu_blocks == base.store_gpu_blocks
    assert dynamic.store_cpu_blocks == base.store_cpu_blocks
    assert dynamic.need_flush is True
    assert dynamic.recompute_requests == {"recompute-1": 256}
    assert dynamic.reset_recompute_requests == {"recompute-1"}
    assert dynamic.completed_recompute_requests == {"recompute-1"}
    assert dynamic.load_block_counts == {"load-1": 2}
    assert dynamic.decision_times == {"load-1": 10.0, "recompute-1": 10.0}


def test_worker_metadata_aggregates_without_double_counting_requests() -> None:
    """Rank metadata is concatenated for later request-level critical-path use."""
    first = DynamicCPUOffloadWorkerMetadata(
        completed_store_events={7: 1},
        copy_samples=[TimingSampleMetadata("req-1", 2, 4.0)],
    )
    second = DynamicCPUOffloadWorkerMetadata(
        completed_store_events={7: 1, 8: 1},
        copy_samples=[TimingSampleMetadata("req-1", 2, 6.0)],
        recompute_samples=[TimingSampleMetadata("req-2", 256, 20.0)],
    )

    merged = first.aggregate(second)

    assert merged.completed_store_events == {7: 2, 8: 1}
    assert [sample.service_ms for sample in merged.copy_samples] == [4.0, 6.0]
    assert [sample.request_id for sample in merged.recompute_samples] == ["req-2"]


def test_worker_metadata_wraps_native_completion_counts() -> None:
    """Native store completion state survives plugin sample transport."""
    base = SimpleCPUOffloadWorkerMetadata(completed_store_events={9: 1})
    sample = TimingSampleMetadata("req-1", 2, 4.0)

    wrapped = as_worker_metadata(base, [sample], [])

    assert wrapped is not None
    assert wrapped.completed_store_events == {9: 1}
    assert wrapped.copy_samples == [sample]
    assert as_worker_metadata(None, [], []) is None


def test_worker_metadata_rejects_unrelated_aggregate_type() -> None:
    """Mixing connector metadata types fails loudly."""
    metadata = DynamicCPUOffloadWorkerMetadata(completed_store_events={})
    native = SimpleCPUOffloadWorkerMetadata(completed_store_events={})

    with pytest.raises(TypeError, match="different worker metadata"):
        metadata.aggregate(native)
