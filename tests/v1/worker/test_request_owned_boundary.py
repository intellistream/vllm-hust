# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the G1 control-only worker envelope boundary."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
    PublicationViolationError,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


class _FakeWorker:
    def __init__(self, output=EMPTY_MODEL_RUNNER_OUTPUT, before_return=None) -> None:
        self.output = output
        self.before_return = before_return
        self.calls = 0

    def execute_model(self, scheduler_output):
        self.calls += 1
        if self.before_return is not None:
            self.before_return(scheduler_output)
        return self.output


class _FakeAsyncOutput(AsyncModelRunnerOutput):
    def get_output(self) -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT


def _wrapper(rank: int, worker: _FakeWorker) -> WorkerWrapperBase:
    wrapper = WorkerWrapperBase(global_rank=rank)
    wrapper.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_request_owned_attention=True)
    )
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    return wrapper


def _output(step_seq: int = 1) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.step_seq = step_seq
    return output


def _reserve(owner_id: int, command_seq: int = 1) -> OwnerCommand:
    return OwnerCommand(
        key=OwnerLeaseKey("req", 0),
        owner_id=owner_id,
        command_seq=command_seq,
        kind=OwnerCommandKind.RESERVE,
        requested_through=8,
    )


def test_targeted_reserve_refuses_while_other_rank_emits_empty_batch() -> None:
    step = _output(step_seq=3)
    step.owner_commands = [_reserve(owner_id=1)]

    rank0_worker = _FakeWorker()
    rank0 = _wrapper(0, rank0_worker)
    rank0_result = rank0.execute_model(step)
    rank0_batch = rank0_result.owner_receipt_batches[0]
    assert rank0_batch.owner_rank == 0
    assert rank0_batch.emitted_step_seq == 3
    assert rank0_batch.events == ()

    rank1 = _wrapper(1, _FakeWorker())
    rank1_result = rank1.execute_model(step)
    rank1_batch = rank1_result.owner_receipt_batches[0]
    assert rank1_batch.owner_rank == 1
    assert rank1_batch.emitted_step_seq == 3
    assert len(rank1_batch.events) == 1
    receipt = rank1_batch.events[0]
    assert not receipt.accepted
    assert receipt.error == "insufficient capacity to reserve"
    assert receipt.free_capacity == 0


def test_command_processing_precedes_underlying_worker(monkeypatch) -> None:
    step = _output()
    step.owner_commands = [_reserve(owner_id=1)]
    order = []
    original_apply = AttentionLeaseManager.apply

    def record_apply(manager, command):
        order.append("apply")
        return original_apply(manager, command)

    monkeypatch.setattr(AttentionLeaseManager, "apply", record_apply)
    worker = _FakeWorker(before_return=lambda _: order.append("worker"))
    wrapper = _wrapper(1, worker)
    wrapper.execute_model(step)
    assert worker.calls == 1
    assert order == ["apply", "worker"]


def test_scheduled_tokens_fail_before_underlying_worker() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    step = _output()
    step.total_num_scheduled_tokens = 1

    with pytest.raises(RuntimeError, match="replicated KV"):
        wrapper.execute_model(step)
    assert worker.calls == 0
    assert wrapper._request_owned_control_manager is None


@pytest.mark.parametrize(
    "total, per_request",
    [
        (False, {}),
        (0.0, {}),
        (0, {"req": 1}),
        (1, {}),
        (0, {"req": True}),
        (0, {"req": -1, "other": 1}),
    ],
)
def test_inconsistent_token_envelope_fails_before_worker(total, per_request) -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    step = _output()
    step.total_num_scheduled_tokens = total
    step.num_scheduled_tokens = per_request
    with pytest.raises(RuntimeError, match="inconsistent token schedule"):
        wrapper.execute_model(step)
    assert worker.calls == 0
    assert wrapper._request_owned_control_manager is None


def test_empty_singleton_is_never_mutated() -> None:
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None
    wrapper = _wrapper(0, _FakeWorker())
    result = wrapper.execute_model(_output())
    assert result is not EMPTY_MODEL_RUNNER_OUTPUT
    assert result.owner_receipt_batches is not None
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None


def test_none_and_async_outputs_fail_explicitly() -> None:
    step = _output()
    step.owner_commands = [_reserve(owner_id=0)]
    none_worker = _FakeWorker(output=None)
    none_wrapper = _wrapper(0, none_worker)
    with pytest.raises(RuntimeError, match="split sampling"):
        none_wrapper.execute_model(step)
    assert none_wrapper._request_owned_control_manager is None

    async_worker = _FakeWorker(output=_FakeAsyncOutput())
    async_wrapper = _wrapper(0, async_worker)
    with pytest.raises(RuntimeError, match="step-keyed receipt FIFO"):
        async_wrapper.execute_model(step)
    assert async_wrapper._request_owned_control_manager is None


@pytest.mark.parametrize("step_seq", [0, -1, True, False, 1.5, "1", None])
def test_invalid_step_fails_before_underlying_worker(step_seq) -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    with pytest.raises(RuntimeError, match="positive non-bool"):
        wrapper.execute_model(_output(step_seq=step_seq))
    assert worker.calls == 0


def test_foreign_commands_and_publications_are_ignored() -> None:
    step = _output(step_seq=4)
    command = _reserve(owner_id=1)
    step.owner_commands = [command]
    step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=command.key,
            owner_id=1,
            step_seq=4,
            command_seq=1,
            runnable_through=8,
        )
    ]
    worker = _FakeWorker()
    result = _wrapper(0, worker).execute_model(step)
    assert worker.calls == 1
    assert result.owner_receipt_batches[0].events == ()


def test_local_publication_without_physical_grant_fails_before_worker() -> None:
    step = _output(step_seq=4)
    command = _reserve(owner_id=0)
    step.owner_commands = [command]
    step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=command.key,
            owner_id=0,
            step_seq=4,
            command_seq=1,
            runnable_through=8,
        )
    ]
    worker = _FakeWorker()
    with pytest.raises(PublicationViolationError, match="no lease"):
        _wrapper(0, worker).execute_model(step)
    assert worker.calls == 0


def test_default_off_path_is_unchanged() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_attention = False
    result = wrapper.execute_model(SchedulerOutput.make_empty())
    assert result is EMPTY_MODEL_RUNNER_OUTPUT
    assert worker.calls == 1
