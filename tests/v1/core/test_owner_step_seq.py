# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused tests for the scheduler's request-owner execution fence."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.ownership import (
    OwnerLeaseKey,
    OwnerReceipt,
    OwnerReceiptBatch,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_schedule_stamps_every_call_with_current_step() -> None:
    scheduler = create_scheduler()

    # A zero-token schedule call is still an execution/control-plane step.
    empty = scheduler.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert empty.step_seq == 1
    assert empty.step_seq == scheduler.current_step

    (request,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    assert scheduled.total_num_scheduled_tokens == 8
    assert scheduled.step_seq == 2
    assert scheduled.step_seq == scheduler.current_step

    # The owner fence is not the deferred-free KV fence.
    assert scheduler.sched_step_seq == 0


def _ingress_scheduler(*, enabled: bool = True, world_size: int = 2) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler_config = SimpleNamespace(enable_request_owned_attention=enabled)
    scheduler.parallel_config = SimpleNamespace(world_size=world_size)
    return scheduler


def _runner_output(batches=None) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        owner_receipt_batches=batches,
    )


def _batch(rank: int, step_seq: int, events=()) -> OwnerReceiptBatch:
    return OwnerReceiptBatch(
        owner_rank=rank,
        emitted_step_seq=step_seq,
        events=tuple(events),
    )


def test_receipt_ingress_accepts_exact_zero_event_envelope() -> None:
    scheduler = _ingress_scheduler()
    step = SchedulerOutput.make_empty()
    step.step_seq = 7
    scheduler._validate_request_owned_receipt_ingress(
        step,
        _runner_output([_batch(1, 7), _batch(0, 7)]),
    )


@pytest.mark.parametrize(
    "batches, match",
    [
        (None, "missing all-worker"),
        ([_batch(0, 7)], "exactly one batch"),
        ([_batch(0, 7), _batch(0, 7)], "exactly one batch"),
        ([_batch(0, 7), _batch(2, 7)], "exactly one batch"),
        ([_batch(False, 7), _batch(1, 7)], "exactly one batch"),
        ([_batch(0, 6), _batch(1, 7)], "emitted_step_seq"),
        ([_batch(0, True), _batch(1, 7)], "emitted_step_seq"),
    ],
)
def test_receipt_ingress_rejects_structural_violations(batches, match) -> None:
    scheduler = _ingress_scheduler()
    step = SchedulerOutput.make_empty()
    step.step_seq = 7
    with pytest.raises(RuntimeError, match=match):
        scheduler._validate_request_owned_receipt_ingress(step, _runner_output(batches))


def test_receipt_ingress_refuses_to_ignore_resource_events() -> None:
    scheduler = _ingress_scheduler()
    step = SchedulerOutput.make_empty()
    step.step_seq = 7
    event = OwnerReceipt(
        key=OwnerLeaseKey("req", 0),
        owner_id=1,
        command_seq=1,
        accepted=False,
        error="insufficient capacity to reserve",
    )
    with pytest.raises(RuntimeError, match="G2 owner-local allocator"):
        scheduler._validate_request_owned_receipt_ingress(
            step,
            _runner_output([_batch(0, 7), _batch(1, 7, [event])]),
        )


def test_update_rejects_missing_receipts_before_other_scheduler_state() -> None:
    scheduler = _ingress_scheduler()
    step = SchedulerOutput.make_empty()
    step.step_seq = 1
    with pytest.raises(RuntimeError, match="missing all-worker"):
        scheduler.update_from_output(step, _runner_output())


def test_receipt_ingress_default_off_is_unchanged() -> None:
    scheduler = _ingress_scheduler(enabled=False)
    scheduler._validate_request_owned_receipt_ingress(
        SchedulerOutput.make_empty(), _runner_output()
    )
