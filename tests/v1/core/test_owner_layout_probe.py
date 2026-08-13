# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.v1.core.sched.owner_layout_probe import (
    MAX_BYTES_ENV,
    MAX_RECORDS_ENV,
    PROBE_DIR_ENV,
    RUN_ID_ENV,
    OwnerLayoutProbe,
)
from vllm.v1.core.sched.ownership import (
    OwnerCacheGroupSnapshot,
    OwnerCachePoolSnapshot,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
    OwnerReceipt,
    OwnerReceiptBatch,
)


def _lease(request_id: str, owner: int, step: int = 3) -> OwnerLeaseToken:
    return OwnerLeaseToken(
        key=OwnerLeaseKey(request_id=request_id, owner_epoch=1),
        owner_id=owner,
        step_seq=step,
        command_seq=1,
        runnable_num_tokens=64,
    )


def test_probe_records_mixed_and_zero_owner_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(PROBE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(RUN_ID_ENV, "owner-cell")
    probe = OwnerLayoutProbe.from_env(world_size=4)
    assert probe is not None
    probe.record_step(
        step_seq=3,
        leases=[_lease("req-b", 2), _lease("req-a", 0)],
        num_scheduled_tokens={"req-a": 5, "req-b": 2},
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "request-owner-layout-owner-cell.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records[0] == {
        "kind": "header",
        "run_id": "owner-cell",
        "schema": "g5-request-owner-lifecycle-observation/v4",
        "world_size": 4,
    }
    step = records[1]
    assert [item["request_id"] for item in step["assignments"]] == [
        "req-a",
        "req-b",
    ]
    assert step["owner_row_counts"] == [5, 0, 2, 0]
    assert step["owner_request_counts"] == [1, 0, 1, 0]
    assert step["scheduled_request_count"] == 2
    assert step["total_scheduled_tokens"] == 7
    assert step["zero_row_owner_ranks"] == [1, 3]
    assert step["owner_cache_pools"] is None
    assert step["commands"] == []
    assert step["receipts"] == []


def test_probe_records_matching_command_and_receipt(tmp_path) -> None:
    probe = OwnerLayoutProbe(
        path=tmp_path / "probe.jsonl",
        run_id="extend-cell",
        world_size=2,
        max_records=4,
        max_bytes=4096,
    )
    key = OwnerLeaseKey(request_id="req", owner_epoch=1)
    command = OwnerCommand(
        key=key,
        owner_id=0,
        command_seq=5,
        kind=OwnerCommandKind.EXTEND,
        required_num_tokens=10,
    )
    receipt = OwnerReceipt(
        key=key,
        owner_id=0,
        command_seq=5,
        accepted=True,
        runnable_num_tokens=10,
    )
    probe.record_step(
        step_seq=7,
        leases=[],
        num_scheduled_tokens={},
        commands=[command],
        receipt_batches=[
            OwnerReceiptBatch(owner_rank=0, emitted_step_seq=7, events=(receipt,)),
            OwnerReceiptBatch(owner_rank=1, emitted_step_seq=7, events=()),
        ],
    )
    record = json.loads((tmp_path / "probe.jsonl").read_text().splitlines()[1])
    assert record["commands"] == [
        {
            "command_seq": 5,
            "kind": "EXTEND",
            "owner_epoch": 1,
            "owner_rank": 0,
            "request_id": "req",
            "required_num_tokens": 10,
        }
    ]
    assert record["receipts"] == [
        {
            "accepted": True,
            "command_seq": 5,
            "error": None,
            "owner_epoch": 1,
            "owner_rank": 0,
            "released": False,
            "request_id": "req",
            "runnable_num_tokens": 10,
        }
    ]


def _pool(rank: int, *, free_blocks: int) -> OwnerCachePoolSnapshot:
    return OwnerCachePoolSnapshot(
        owner_rank=rank,
        total_blocks=16,
        free_blocks=free_blocks,
        bytes_per_block=4096,
        groups=(
            OwnerCacheGroupSnapshot(
                group_index=0,
                spec_kind="full",
                effective_tokens_per_block=128,
                allocated_blocks=16 - free_blocks,
                resident_blocks=16 - free_blocks,
            ),
        ),
    )


def test_probe_records_block_id_free_physical_capacity(tmp_path) -> None:
    probe = OwnerLayoutProbe(
        path=tmp_path / "probe.jsonl",
        run_id="capacity-cell",
        world_size=2,
        max_records=4,
        max_bytes=4096,
    )
    probe.record_step(
        step_seq=1,
        leases=[_lease("req", 0, step=1)],
        num_scheduled_tokens={"req": 1},
        cache_pool_snapshots={0: _pool(0, free_blocks=12), 1: _pool(1, free_blocks=16)},
    )
    records = [
        json.loads(line) for line in (tmp_path / "probe.jsonl").read_text().splitlines()
    ]
    pools = records[1]["owner_cache_pools"]
    assert [pool["owner_rank"] for pool in pools] == [0, 1]
    assert pools[0] == {
        "owner_rank": 0,
        "total_blocks": 16,
        "free_blocks": 12,
        "bytes_per_block": 4096,
        "groups": [
            {
                "group_index": 0,
                "spec_kind": "full",
                "effective_tokens_per_block": 128,
                "allocated_blocks": 4,
                "resident_blocks": 4,
            }
        ],
    }
    assert "block_id" not in repr(pools)


def test_probe_rejects_partial_or_mislabeled_capacity(tmp_path) -> None:
    probe = OwnerLayoutProbe(
        path=tmp_path / "probe.jsonl",
        run_id="capacity-cell",
        world_size=2,
        max_records=4,
        max_bytes=4096,
    )
    with pytest.raises(ValueError, match="cover every owner rank"):
        probe.record_step(
            step_seq=1,
            leases=[],
            num_scheduled_tokens={},
            cache_pool_snapshots={0: _pool(0, free_blocks=16)},
        )

    with pytest.raises(ValueError, match="does not match snapshot owner"):
        probe.record_step(
            step_seq=2,
            leases=[],
            num_scheduled_tokens={},
            cache_pool_snapshots={
                0: _pool(1, free_blocks=16),
                1: _pool(0, free_blocks=16),
            },
        )


def test_probe_fails_closed_on_identity_or_bounds(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(PROBE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(RUN_ID_ENV, "owner-cell")
    monkeypatch.setenv(MAX_RECORDS_ENV, "1")
    probe = OwnerLayoutProbe.from_env(world_size=2)
    assert probe is not None
    with pytest.raises(RuntimeError, match="exceeded 1 records"):
        probe.record_step(
            step_seq=1,
            leases=[_lease("req", 0, step=1)],
            num_scheduled_tokens={"req": 1},
        )

    monkeypatch.setenv(RUN_ID_ENV, "bad/run")
    with pytest.raises(ValueError, match=RUN_ID_ENV):
        OwnerLayoutProbe.from_env(world_size=2)


def test_probe_rejects_mismatched_lease_coverage(tmp_path) -> None:
    probe = OwnerLayoutProbe(
        path=tmp_path / "probe.jsonl",
        run_id="owner-cell",
        world_size=2,
        max_records=4,
        max_bytes=4096,
    )
    with pytest.raises(ValueError, match="exactly match"):
        probe.record_step(
            step_seq=3,
            leases=[_lease("req-a", 0)],
            num_scheduled_tokens={"req-b": 1},
        )


def test_probe_fails_closed_before_exceeding_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(PROBE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(RUN_ID_ENV, "owner-cell")
    monkeypatch.setenv(MAX_BYTES_ENV, "1")
    with pytest.raises(RuntimeError, match="exceeded 1 bytes"):
        OwnerLayoutProbe.from_env(world_size=2)
