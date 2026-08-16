# SPDX-License-Identifier: Apache-2.0

"""Scheduler proof that RESTORE can ride an unrelated token-bearing step."""

import time

import pytest

from tests.v1.core.test_scheduler_owner_admission import (
    _apply_receipts,
    _make_scheduler,
    _receipt,
    _request,
)
from vllm.v1.core.sched.ownership import OwnerCommandKind, OwnerLeaseKey

pytestmark = pytest.mark.cpu_test


def test_restore_command_rides_unrelated_running_work() -> None:
    scheduler = _make_scheduler(
        max_num_scheduled_tokens=8,
        enable_request_owned_kv_offload=True,
        request_owned_decode_reservation_tokens=64,
    )
    cold = _request("cold")
    other = _request("other")
    scheduler.add_request(cold)
    scheduler.add_request(other)
    reserve_out = scheduler.schedule()
    reserves = [
        command
        for command in reserve_out.owner_commands
        if command.kind is OwnerCommandKind.RESERVE
    ]
    assert len(reserves) == 2
    events_by_rank = {}
    for command in reserves:
        events_by_rank.setdefault(command.owner_id, []).append(
            _receipt(
                command.key,
                command.owner_id,
                command.command_seq,
                runnable=command.required_num_tokens,
            )
        )
    _apply_receipts(scheduler, reserve_out, events_by_rank)
    scheduler.schedule()

    scheduler._preempt_request_owned(cold, time.monotonic())
    preempt_out = scheduler.schedule()
    assert any(
        command.kind is OwnerCommandKind.PREEMPT
        for command in preempt_out.owner_commands
    )
    for command in preempt_out.owner_commands:
        scheduler._apply_owner_receipt(
            _receipt(
                command.key,
                command.owner_id,
                command.command_seq,
                runnable=command.required_num_tokens,
            )
        )

    restore_out = scheduler.schedule()

    restore = next(
        command
        for command in restore_out.owner_commands
        if command.kind is OwnerCommandKind.RESTORE
    )
    assert restore.key == OwnerLeaseKey("cold", 0)
    assert restore_out.num_scheduled_tokens.get("other", 0) > 0
    assert "cold" not in restore_out.num_scheduled_tokens
