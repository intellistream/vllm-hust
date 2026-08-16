# SPDX-License-Identifier: Apache-2.0

"""CPU lifecycle tests for early request-owned H2D restore submission."""

from collections.abc import Callable

import pytest

from tests.v1.worker.test_request_owned_boundary import (
    _FakeStore,
    _FakeWorker,
    _output,
    _real_reserve,
    _real_wrapper,
    _SyncOffloadingWorker,
    _wrapper,
)
from vllm.v1.core.sched.ownership import (
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ModelRunnerOutput,
    OwnerSamplingBatch,
)
from vllm.v1.worker.request_owned_offload import RequestOwnedBulkOffloadAdapter

pytestmark = pytest.mark.cpu_test


class _BulkRestoreWorker(_FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.restore_zero_plans: list[tuple[tuple[int, ...], ...]] = []

    def execute_request_owned_bulk_restore(self, work) -> None:
        for item in work:
            self.restore_zero_plans.append(item.zero_block_ids)
            item.execute_after_zero()


class _DeferredBulkRestoreWorker(_BulkRestoreWorker):
    """Ascend-shaped owner-sampling worker with deferred terminal output."""

    def __init__(self, owner_rank: int = 0) -> None:
        super().__init__()
        self.owner_rank = owner_rank
        self.output = None
        self.sample_calls = 0
        self.sample_grammar = object()
        self.return_none_from_sample = False

    def sample_tokens(self, grammar_output):
        self.sample_calls += 1
        self.sample_grammar = grammar_output
        if self.return_none_from_sample:
            return None
        metadata = self.metadata_handoffs[-1]
        output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
        output.owner_sampling_batches = [
            OwnerSamplingBatch(
                owner_rank=self.owner_rank,
                emitted_step_seq=metadata.step_seq,
                row_ids=(),
            )
        ]
        return output


class _DeferredLoadOffloadingWorker(OffloadingWorker):
    """D2H completes immediately; H2D completes only at its exact wait."""

    def __init__(
        self,
        timeline: list[str],
        before_load_wait: Callable[[], None] | None = None,
    ) -> None:
        self.timeline = timeline
        self.before_load_wait = before_load_wait
        self.finished: list[TransferResult] = []
        self.pending_loads: set[int] = set()

    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        self.finished.append(TransferResult(job_id=job_id, success=True))
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        self.timeline.append("h2d-submit")
        self.pending_loads.add(job_id)
        return True

    def get_finished(self) -> list[TransferResult]:
        finished, self.finished = self.finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        load_ids = job_ids.intersection(self.pending_loads)
        if not load_ids:
            return
        if self.before_load_wait is not None:
            self.before_load_wait()
        self.timeline.append("h2d-wait")
        self.pending_loads.difference_update(load_ids)
        self.finished.extend(
            TransferResult(job_id=job_id, success=True) for job_id in sorted(load_ids)
        )


class _ConcurrentOffloadingWorker(OffloadingWorker):
    """Let an H2D wait discover both H2D and an older D2H completion."""

    def __init__(self) -> None:
        self.pending: dict[int, str] = {}
        self.finished: list[TransferResult] = []

    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        self.pending[job_id] = "store"
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        self.pending[job_id] = "load"
        return True

    def get_finished(self) -> list[TransferResult]:
        finished, self.finished = self.finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        finish_all = any(self.pending.get(job_id) == "load" for job_id in job_ids)
        selected = (
            set(self.pending) if finish_all else job_ids.intersection(self.pending)
        )
        for job_id in sorted(selected):
            self.pending.pop(job_id)
            self.finished.append(TransferResult(job_id=job_id, success=True))


def _prepare_cold_request(
    worker: _BulkRestoreWorker,
    transfer_worker: OffloadingWorker,
    *,
    include_other: bool = False,
):
    wrapper = _real_wrapper(worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    wrapper.vllm_config.scheduler_config.enable_request_owned_kv_offload = True
    wrapper._request_owned_offload_adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=0,
        manager=CPUOffloadingManager(num_blocks=16),
        worker=transfer_worker,
    )
    store = wrapper._request_owned_kv_store
    assert store is not None
    key = OwnerLeaseKey("req", 0)

    reserve_step = _output(step_seq=1)
    reserve_step.owner_commands = [_real_reserve(owner_id=0, command_seq=1)]
    if include_other:
        reserve_step.owner_commands.append(
            _real_reserve(owner_id=0, command_seq=2, request_id="other")
        )
    wrapper.execute_model(reserve_step)
    assert store.mark_computed(key, 6)

    preempt_step = _output(step_seq=2)
    preempt_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=3 if include_other else 2,
            kind=OwnerCommandKind.PREEMPT,
            required_num_tokens=10,
        )
    ]
    wrapper.execute_model(preempt_step)
    assert store.snapshot(key) is None
    return wrapper, store, key


def _restore_command(key: OwnerLeaseKey, command_seq: int) -> OwnerCommand:
    return OwnerCommand(
        key=key,
        owner_id=0,
        command_seq=command_seq,
        kind=OwnerCommandKind.RESTORE,
        required_num_tokens=10,
    )


def _mixed_restore_step(key: OwnerLeaseKey):
    step = _output(step_seq=3)
    step.owner_commands = [_restore_command(key, command_seq=4)]
    step.num_scheduled_tokens = {"other": 1}
    step.total_num_scheduled_tokens = 1
    step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=OwnerLeaseKey("other", 0),
            owner_id=0,
            step_seq=3,
            command_seq=2,
            runnable_num_tokens=10,
        )
    ]
    return step


def test_background_restore_submits_before_unrelated_forward_then_waits() -> None:
    timeline: list[str] = []
    worker = _BulkRestoreWorker()
    worker.before_return = lambda _: timeline.append("forward")
    transfer = _DeferredLoadOffloadingWorker(
        timeline,
        before_load_wait=lambda: timeline.index("forward"),
    )
    wrapper, store, key = _prepare_cold_request(worker, transfer, include_other=True)
    timeline.clear()

    batch = wrapper.execute_model(_mixed_restore_step(key)).owner_receipt_batches[0]

    assert timeline == ["h2d-submit", "forward", "h2d-wait"]
    assert batch.events[0].accepted
    assert batch.events[0].pending_dma == 0
    assert store.is_restore_ready(key)


def test_deferred_restore_preserves_real_grammar_until_sampling() -> None:
    timeline: list[str] = []
    worker = _DeferredBulkRestoreWorker()
    worker.output = EMPTY_MODEL_RUNNER_OUTPUT
    worker.before_return = lambda _: timeline.append("forward")
    transfer = _DeferredLoadOffloadingWorker(timeline)
    wrapper, store, key = _prepare_cold_request(worker, transfer, include_other=True)
    timeline.clear()
    worker.output = None
    original_sample = worker.sample_tokens
    grammar = object()

    def sample_tokens(received_grammar):
        assert received_grammar is grammar
        timeline.append("sample")
        return original_sample(received_grammar)

    worker.sample_tokens = sample_tokens
    assert wrapper.execute_model(_mixed_restore_step(key)) is None
    assert timeline == ["h2d-submit", "forward"]
    assert wrapper._request_owned_restore_guard is None
    assert wrapper._request_owned_deferred.restore_guard is not None

    result = wrapper.sample_tokens(grammar)

    assert timeline == ["h2d-submit", "forward", "sample", "h2d-wait"]
    assert result.owner_receipt_batches[0].events[0].pending_dma == 0
    assert store.is_restore_ready(key)
    assert wrapper._request_owned_deferred is None


def test_failed_deferred_restore_fences_dma_and_fail_stops() -> None:
    timeline: list[str] = []
    worker = _DeferredBulkRestoreWorker()
    worker.output = EMPTY_MODEL_RUNNER_OUTPUT
    transfer = _DeferredLoadOffloadingWorker(timeline)
    wrapper, store, key = _prepare_cold_request(worker, transfer, include_other=True)
    timeline.clear()
    worker.output = None
    worker.return_none_from_sample = True

    assert wrapper.execute_model(_mixed_restore_step(key)) is None
    with pytest.raises(RuntimeError, match="sample_tokens returned None"):
        wrapper.sample_tokens(object())

    assert timeline == ["h2d-submit", "h2d-wait"]
    assert store.snapshot(key) is None
    assert wrapper._request_owned_restore_guard is None
    assert wrapper._request_owned_fail_stop is not None
    with pytest.raises(RuntimeError, match="irreversible fail-stop"):
        wrapper.sample_tokens(object())


def test_restore_poll_holds_concurrent_drain_receipt_for_its_controller() -> None:
    worker = _BulkRestoreWorker()
    transfer = _ConcurrentOffloadingWorker()
    wrapper = _real_wrapper(worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    wrapper.vllm_config.scheduler_config.enable_request_owned_kv_offload = True
    wrapper._request_owned_offload_adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=0,
        manager=CPUOffloadingManager(num_blocks=32),
        worker=transfer,
    )
    store = wrapper._request_owned_kv_store
    assert store is not None
    cold = OwnerLeaseKey("cold", 0)
    draining = OwnerLeaseKey("draining", 0)
    other = OwnerLeaseKey("other", 0)

    reserve = _output(step_seq=1)
    reserve.owner_commands = [
        _real_reserve(owner_id=0, command_seq=1, request_id="cold"),
        _real_reserve(owner_id=0, command_seq=2, request_id="draining"),
        _real_reserve(owner_id=0, command_seq=3, request_id="other"),
    ]
    wrapper.execute_model(reserve)
    assert store.mark_computed(cold, 6)
    assert store.mark_computed(draining, 6)

    make_cold = _output(step_seq=2)
    make_cold.owner_commands = [
        OwnerCommand(
            key=cold,
            owner_id=0,
            command_seq=4,
            kind=OwnerCommandKind.PREEMPT,
            required_num_tokens=10,
        )
    ]
    assert wrapper.execute_model(make_cold).owner_receipt_batches[0].events
    assert store.snapshot(cold) is None

    start_drain = _output(step_seq=3)
    start_drain.owner_commands = [
        OwnerCommand(
            key=draining,
            owner_id=0,
            command_seq=5,
            kind=OwnerCommandKind.PREEMPT,
            required_num_tokens=10,
        )
    ]
    start_drain.num_scheduled_tokens = {"other": 1}
    start_drain.total_num_scheduled_tokens = 1
    start_drain.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=other,
            owner_id=0,
            step_seq=3,
            command_seq=3,
            runnable_num_tokens=10,
        )
    ]
    drain_batch = wrapper.execute_model(start_drain).owner_receipt_batches[0]
    assert drain_batch.events == ()
    assert drain_batch.pending_dma == 1
    assert store.snapshot(draining) is not None

    restore = _output(step_seq=4)
    restore.owner_commands = [_restore_command(cold, command_seq=6)]
    restore.num_scheduled_tokens = {"other": 1}
    restore.total_num_scheduled_tokens = 1
    restore.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=other,
            owner_id=0,
            step_seq=4,
            command_seq=3,
            runnable_num_tokens=10,
        )
    ]
    final_batch = wrapper.execute_model(restore).owner_receipt_batches[0]

    assert [(event.key, event.command_seq) for event in final_batch.events] == [
        (draining, 5),
        (cold, 6),
    ]
    assert final_batch.pending_dma == 0
    assert store.snapshot(draining) is None
    assert store.is_restore_ready(cold)
    assert transfer.pending == {}


def test_restore_without_useful_work_falls_back_to_exact_wait() -> None:
    timeline: list[str] = []
    worker = _BulkRestoreWorker()
    worker.before_return = lambda _: timeline.append("heartbeat")
    transfer = _DeferredLoadOffloadingWorker(timeline)
    wrapper, store, key = _prepare_cold_request(worker, transfer)
    timeline.clear()

    step = _output(step_seq=3)
    step.owner_commands = [_restore_command(key, command_seq=3)]
    batch = wrapper.execute_model(step).owner_receipt_batches[0]

    assert timeline == ["h2d-submit", "heartbeat", "h2d-wait"]
    assert batch.events[0].pending_dma == 0
    assert store.is_restore_ready(key)


def test_restore_target_cannot_be_scheduled_before_h2d_receipt() -> None:
    worker = _BulkRestoreWorker()
    wrapper, store, key = _prepare_cold_request(worker, _SyncOffloadingWorker())
    step = _output(step_seq=3)
    step.owner_commands = [_restore_command(key, command_seq=3)]
    step.num_scheduled_tokens = {"req": 1}
    step.total_num_scheduled_tokens = 1
    step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=key,
            owner_id=0,
            step_seq=3,
            command_seq=3,
            runnable_num_tokens=10,
        )
    ]

    with pytest.raises(RuntimeError, match="cannot schedule its target"):
        wrapper.execute_model(step)
    assert store.snapshot(key) is None


def test_background_restore_failure_fences_h2d_before_recycling() -> None:
    timeline: list[str] = []
    worker = _BulkRestoreWorker()
    transfer = _DeferredLoadOffloadingWorker(timeline)
    wrapper, store, key = _prepare_cold_request(worker, transfer)
    timeline.clear()

    def fail_after_submit(_scheduler_output) -> None:
        timeline.append("forward-failed")
        raise RuntimeError("late model failure")

    worker.before_return = fail_after_submit
    step = _output(step_seq=3)
    step.owner_commands = [_restore_command(key, command_seq=3)]
    with pytest.raises(RuntimeError, match="late model failure"):
        wrapper.execute_model(step)

    assert timeline == ["h2d-submit", "forward-failed", "h2d-wait"]
    assert store.snapshot(key) is None
    assert wrapper._request_owned_restore_guard is None
    assert wrapper._request_owned_fail_stop is None


def test_bulk_restore_rolls_back_on_late_terminal_failure_and_retries() -> None:
    worker = _BulkRestoreWorker()
    wrapper, store, key = _prepare_cold_request(worker, _SyncOffloadingWorker())
    step = _output(step_seq=3)
    step.owner_commands = [_restore_command(key, command_seq=3)]

    original_pool_snapshot = store.pool_snapshot

    def fail_terminal_snapshot():
        raise RuntimeError("late receipt failure")

    store.pool_snapshot = fail_terminal_snapshot
    with pytest.raises(RuntimeError, match="late receipt failure"):
        wrapper.execute_model(step)
    assert store.snapshot(key) is None
    assert wrapper._request_owned_fail_stop is None

    store.pool_snapshot = original_pool_snapshot
    receipt = wrapper.execute_model(step).owner_receipt_batches[0].events[0]
    assert receipt.accepted
    assert receipt.pending_dma == 0
    assert store.is_restore_ready(key)


def test_restore_commits_deferred_sampling_step_synchronously() -> None:
    worker = _DeferredBulkRestoreWorker()
    worker.output = EMPTY_MODEL_RUNNER_OUTPUT
    wrapper, store, key = _prepare_cold_request(worker, _SyncOffloadingWorker())
    worker.output = None
    step = _output(step_seq=3)
    step.owner_commands = [_restore_command(key, command_seq=3)]

    worker.return_none_from_sample = True
    with pytest.raises(RuntimeError, match="sample_tokens returned None"):
        wrapper.execute_model(step)
    assert store.snapshot(key) is None
    assert wrapper._request_owned_deferred is None

    worker.return_none_from_sample = False
    result = wrapper.execute_model(step)
    assert result.owner_sampling_batches == [
        OwnerSamplingBatch(owner_rank=0, emitted_step_seq=3, row_ids=())
    ]
    assert result.owner_receipt_batches[0].events[0].pending_dma == 0
    assert store.is_restore_ready(key)


def test_remote_rank_accepts_mixed_global_restore_step() -> None:
    worker = _DeferredBulkRestoreWorker(owner_rank=1)
    store = _FakeStore(1)
    wrapper = _wrapper(1, worker, store)
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    step = _output(step_seq=7)
    step.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("remote-restore", 0),
            owner_id=0,
            command_seq=4,
            kind=OwnerCommandKind.RESTORE,
            required_num_tokens=10,
        ),
        OwnerCommand(
            key=OwnerLeaseKey("remote-extend", 0),
            owner_id=0,
            command_seq=5,
            kind=OwnerCommandKind.EXTEND,
            required_num_tokens=12,
        ),
    ]

    result = wrapper.execute_model(step)

    assert worker.calls == 1
    assert worker.sample_calls == 1
    assert store.calls == ["build", "mark", "flush", "pool_snapshot"]
    assert result.owner_receipt_batches[0].events == ()


def test_resume_activates_only_after_background_restore_completion() -> None:
    worker = _BulkRestoreWorker()
    wrapper, store, key = _prepare_cold_request(worker, _SyncOffloadingWorker())
    restore_step = _output(step_seq=3)
    restore_step.owner_commands = [_restore_command(key, command_seq=3)]
    restore_receipt = wrapper.execute_model(restore_step).owner_receipt_batches[0]
    assert restore_receipt.events[0].pending_dma == 0
    assert store.is_restore_ready(key)

    resume_step = _output(step_seq=4)
    resume_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=4,
            kind=OwnerCommandKind.RESERVE,
            required_num_tokens=10,
            allocation=OwnerAllocationDescriptor(
                key=key,
                num_prompt_tokens=10,
                num_computed_tokens=6,
                num_tokens=10,
                status=OwnerAdmissionStatus.PREEMPTED,
            ),
        )
    ]
    receipt = wrapper.execute_model(resume_step).owner_receipt_batches[0].events[0]
    assert receipt.accepted
    assert not store.is_restore_ready(key)
