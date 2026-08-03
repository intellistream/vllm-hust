# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    DirectionalTransferStats,
    OffloadingWorkerMetadata,
    TransferStats,
)
from vllm.v1.kv_recovery_profile import (
    KV_RECOVERY_PROFILE_BINDING,
    MAX_H2D_RECEIPTS_PER_WORKER_STEP,
    KVRecoveryH2DReceipt,
    KVRecoveryIdentity,
)

pytestmark = pytest.mark.cpu_test


def make_receipt(job_id: int = 42) -> KVRecoveryH2DReceipt:
    trace_id = "1" * 32
    lifecycle_id = f"{trace_id}:e:0"
    identity = KVRecoveryIdentity(
        run_id="0" * 32,
        trace_id=trace_id,
        engine_lifecycle_id=lifecycle_id,
        runtime_request_id="request-0",
        recovery_epoch=1,
        episode_id=f"{lifecycle_id}:k:1",
        base_preempted_event_id=f"{'b' * 32}:e:0",
    )
    return KVRecoveryH2DReceipt(
        binding=KV_RECOVERY_PROFILE_BINDING,
        connector_job_id=job_id,
        transfer_id=f"{'a' * 32}:t:{job_id}",
        identity=identity,
        block_set_id="b" * 64,
        process_uuid="a" * 32,
        rank=0,
        world_size=1,
        clock_domain_id="c" * 32,
        communication_done_event_id=f"{'a' * 32}:e:{job_id}",
        restore_done_profile_record_id=f"{'a' * 32}:k:{job_id}",
        timestamp_ns=10,
        bytes_moved=128,
    )


def test_aggregate_sums_counts():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 2, 7: 2}


def test_aggregate_disjoint_jobs():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={43: 1, 8: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 1, 7: 1, 43: 1, 8: 1}


def test_aggregate_multiple_workers():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1, 8: 1})
    meta3 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 8: 1})
    result = meta1.aggregate(meta2).aggregate(meta3)
    assert result.completed_jobs == {42: 3, 43: 2, 7: 2, 8: 2}


def test_aggregate_transfer_stats():
    meta1 = OffloadingWorkerMetadata(
        transfer_stats=TransferStats(
            load=DirectionalTransferStats(bytes=10, time=0.5, sizes=[10])
        )
    )
    meta2 = OffloadingWorkerMetadata(
        transfer_stats=TransferStats(
            load=DirectionalTransferStats(bytes=20, time=1.0, sizes=[20, 30])
        )
    )

    result = meta1.aggregate(meta2)

    assert result.transfer_stats.load.bytes == 30
    assert result.transfer_stats.load.time == 1.5
    assert result.transfer_stats.load.sizes == [10, 20, 30]


def test_add_h2d_receipt_does_not_change_completion_counts():
    meta = OffloadingWorkerMetadata(completed_jobs={42: 1})
    receipt = make_receipt()

    assert meta.add_kv_recovery_h2d_receipt(receipt)
    assert meta.completed_jobs == {42: 1}
    assert meta.kv_recovery_h2d_receipts == (receipt,)
    assert not meta.kv_recovery_h2d_receipts_truncated


def test_aggregate_preserves_duplicate_receipts_for_exact_one_validation():
    receipt = make_receipt()
    meta1 = OffloadingWorkerMetadata(
        completed_jobs={42: 1},
        kv_recovery_h2d_receipts=(receipt,),
    )
    meta2 = OffloadingWorkerMetadata(
        completed_jobs={42: 1},
        kv_recovery_h2d_receipts=(receipt,),
    )

    result = meta1.aggregate(meta2)

    assert result.completed_jobs == {42: 2}
    assert result.kv_recovery_h2d_receipts == (receipt, receipt)
    assert not result.kv_recovery_h2d_receipts_truncated
    assert meta1.completed_jobs == {42: 1}
    assert meta2.completed_jobs == {42: 1}


def test_aggregate_bounds_receipts_without_affecting_serving_metadata():
    receipt = make_receipt()
    meta1 = OffloadingWorkerMetadata(
        completed_jobs={42: 1},
        kv_recovery_h2d_receipts=(receipt,) * MAX_H2D_RECEIPTS_PER_WORKER_STEP,
    )
    meta2 = OffloadingWorkerMetadata(
        completed_jobs={42: 1},
        kv_recovery_h2d_receipts=(receipt,),
    )

    result = meta1.aggregate(meta2)

    assert result.completed_jobs == {42: 2}
    assert len(result.kv_recovery_h2d_receipts) == (MAX_H2D_RECEIPTS_PER_WORKER_STEP)
    assert result.kv_recovery_h2d_receipts_truncated
