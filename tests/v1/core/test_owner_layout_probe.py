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
from vllm.v1.core.sched.ownership import OwnerLeaseKey, OwnerLeaseToken


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
        "schema": "g4-request-owner-layout-observation/v1",
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
