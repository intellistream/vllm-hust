# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the G3 request-owned deferred sampling lifecycle.

Covers the ``enable_request_owned_sampling`` wrapper seam: strict flag
gating, the pending deferred record set when the underlying
``execute_model`` returns ``None``, the explicit ``sample_tokens``
completion (mark -> flush -> emit -> commit), mark-once/replay semantics,
and the irreversible post-mark fail-stop latch.  The injected fake store
logs every wrapper call so the exact ordering of store operations is
asserted directly.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerCachePoolSnapshot,
    OwnerLeaseKey,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.request_owned_kv import (
    AllocationResult,
    DeferredFreeResult,
    RequestOwnedStepEntry,
    RequestOwnedStepMarkResult,
    RequestOwnedStepMetadata,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test

#: Sentinel for the ``_wrapper`` sampling parameter: leaves the G3 flag
#: attribute absent so ``None`` itself can be tested as an invalid value.
_ABSENT = object()


class _FakeAsyncOutput(AsyncModelRunnerOutput):
    def get_output(self) -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT


class _FakeWorker:
    """Worker with independent execute_model / sample_tokens outputs."""

    def __init__(
        self,
        execute_output=EMPTY_MODEL_RUNNER_OUTPUT,
        sample_output=EMPTY_MODEL_RUNNER_OUTPUT,
    ) -> None:
        self.execute_output = execute_output
        self.sample_output = sample_output
        self.execute_calls = 0
        self.sample_calls = 0
        self.sample_grammar = None
        self.metadata_handoffs: list[RequestOwnedStepMetadata] = []

    def execute_model(self, scheduler_output):
        self.execute_calls += 1
        return self.execute_output

    def sample_tokens(self, grammar_output):
        self.sample_calls += 1
        self.sample_grammar = grammar_output
        return self.sample_output

    def set_request_owned_step_metadata(self, metadata):
        self.metadata_handoffs.append(metadata)


class _FakeStore:
    """Wrapper-ordering fake for the physical store.

    Commands accept unless their kind is in ``reject``; ``build_step_metadata``
    returns empty-entry metadata unless ``reject_build``; ``mark_computed_batch``
    accepts (logged as ``"mark"``) unless ``reject_mark``.  Every call is
    logged so tests can assert that mark happens exactly once and that the
    receipt snapshot is taken only after a successful flush.
    """

    def __init__(
        self,
        owner_rank: int = 0,
        reject=(),
        reject_build: bool = False,
        reject_mark: bool = False,
    ) -> None:
        self.owner_rank = owner_rank
        self.reject = set(reject)
        self.reject_build = reject_build
        self.reject_mark = reject_mark
        self.calls: list[str] = []
        self.mark_calls = 0
        self.last_mark_metadata: RequestOwnedStepMetadata | None = None

    def reserve(self, command):
        self.calls.append("reserve")
        return self._allocation(command)

    def extend(self, command):
        self.calls.append("extend")
        return self._allocation(command)

    def preempt(self, command):
        self.calls.append("preempt")
        return self._free(command)

    def release(self, command):
        self.calls.append("release")
        return self._free(command)

    def restore(self, command):
        self.calls.append("restore")
        return DeferredFreeResult(
            accepted=False,
            key=command.key,
            error="RESTORE is out of scope for the physical KV store",
        )

    def build_step_metadata(
        self,
        step_seq,
        tokens,
        request_token_counts,
        scheduled_spec_decode_tokens,
    ):
        self.calls.append("build")
        if self.reject_build:
            return SimpleNamespace(
                accepted=False,
                step_seq=step_seq,
                metadata=None,
                error="fake build failure",
            )
        return SimpleNamespace(
            accepted=True,
            step_seq=step_seq,
            metadata=RequestOwnedStepMetadata(
                step_seq=step_seq, owner_rank=self.owner_rank, entries=()
            ),
            error=None,
        )

    def mark_computed_batch(self, metadata, committed_num_tokens=None):
        self.calls.append("mark")
        self.mark_calls += 1
        self.last_mark_metadata = metadata
        if self.reject_mark:
            return RequestOwnedStepMarkResult(
                accepted=False,
                step_seq=metadata.step_seq,
                error="fake mark failure",
            )
        return RequestOwnedStepMarkResult(accepted=True, step_seq=metadata.step_seq)

    def flush(self):
        self.calls.append("flush")
        return ()

    def pool_snapshot(self):
        self.calls.append("pool_snapshot")
        return OwnerCachePoolSnapshot(
            owner_rank=self.owner_rank, total_blocks=32, free_blocks=32
        )

    def _allocation(self, command):
        if command.kind in self.reject:
            return AllocationResult(
                accepted=False, key=command.key, error="physical reserve failure"
            )
        return AllocationResult(accepted=True, key=command.key)

    def _free(self, command):
        if command.kind in self.reject:
            return DeferredFreeResult(
                accepted=False, key=command.key, error="physical free failure"
            )
        return DeferredFreeResult(accepted=True, key=command.key, deferred=True)


def _wrapper(
    rank: int = 0,
    worker: _FakeWorker | None = None,
    store: _FakeStore | None = None,
    sampling: object = _ABSENT,
) -> WorkerWrapperBase:
    """Build a request-owned wrapper; ``sampling=_ABSENT`` leaves the G3
    flag absent (default off), otherwise it is set verbatim on the
    scheduler config so strict bool gating can be exercised."""
    wrapper = WorkerWrapperBase(global_rank=rank)
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64),
        scheduler_config=SimpleNamespace(
            enable_request_owned_attention=True,
            max_num_batched_tokens=16,
            max_num_seqs=4,
        ),
    )
    if sampling is not _ABSENT:
        wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = sampling
    wrapper.worker = worker if worker is not None else _FakeWorker()
    wrapper.mm_receiver_cache = None
    wrapper._request_owned_kv_store = store if store is not None else _FakeStore(rank)
    return wrapper


def _output(
    step_seq: int = 1, total: int = 0, per_request: dict | None = None
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.step_seq = step_seq
    output.total_num_scheduled_tokens = total
    output.num_scheduled_tokens = dict(per_request or {})
    return output


def _token_step(step_seq: int = 1) -> SchedulerOutput:
    return _output(step_seq=step_seq, total=1, per_request={"req": 1})


# -- synchronous token output: shared terminal path ---------------------------


def test_sync_token_output_takes_terminal_path_and_marks_once() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)

    result = wrapper.execute_model(_token_step())

    assert worker.execute_calls == 1
    assert store.mark_calls == 1
    assert store.calls == ["build", "mark", "flush", "pool_snapshot"]
    assert wrapper._request_owned_deferred is None
    assert wrapper._request_owned_control_manager is not None
    # The output is copied: the shared empty singleton is never mutated and
    # the receipt batch is attached only after the successful completion.
    assert result is not EMPTY_MODEL_RUNNER_OUTPUT
    assert result.owner_receipt_batches is not None
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None
    batch = result.owner_receipt_batches[0]
    assert batch.emitted_step_seq == 1
    assert batch.cache_pool.owner_rank == 0
    assert batch.cache_pool.total_blocks == 32


def test_sync_mark_rejection_fails_without_commit() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0, reject_mark=True)
    wrapper = _wrapper(0, worker, store, sampling=True)

    with pytest.raises(RuntimeError, match="computed batch mark failed"):
        wrapper.execute_model(_token_step())

    assert worker.execute_calls == 1
    assert store.calls == ["build", "mark"]
    assert "flush" not in store.calls
    assert wrapper._request_owned_control_manager is None
    # A mark rejection is atomic, not fail-stop: no latch is set.
    assert wrapper._request_owned_fail_stop is None


# -- deferred lifecycle: execute_model -> None, then sample_tokens -----------


def test_deferred_execute_none_then_sample_tokens_completes() -> None:
    worker = _FakeWorker(execute_output=None)
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)
    grammar = object()

    result = wrapper.execute_model(_token_step())

    assert result is None
    assert worker.execute_calls == 1
    assert store.mark_calls == 0
    assert store.calls == ["build"]
    pending = wrapper._request_owned_deferred
    assert pending is not None
    assert pending.step_seq == 1
    assert isinstance(pending.trial_manager, AttentionLeaseManager)
    assert pending.metadata.step_seq == 1
    assert wrapper._request_owned_control_manager is None

    result = wrapper.sample_tokens(grammar)

    assert worker.sample_calls == 1
    assert worker.sample_grammar is grammar
    assert store.mark_calls == 1
    assert store.calls == ["build", "mark", "flush", "pool_snapshot"]
    assert wrapper._request_owned_deferred is None
    assert wrapper._request_owned_control_manager is not None
    # Receipts exist only after the full terminal path succeeded.
    assert result is not EMPTY_MODEL_RUNNER_OUTPUT
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None
    batch = result.owner_receipt_batches[0]
    assert batch.emitted_step_seq == 1
    assert batch.cache_pool.total_blocks == 32


def test_execute_model_while_pending_raises_and_keeps_pending() -> None:
    worker = _FakeWorker(execute_output=None)
    wrapper = _wrapper(0, worker, sampling=True)
    wrapper.execute_model(_token_step())
    pending = wrapper._request_owned_deferred

    with pytest.raises(RuntimeError, match="pending deferred sampling step"):
        wrapper.execute_model(_token_step(step_seq=2))

    assert worker.execute_calls == 1
    assert wrapper._request_owned_deferred is pending


# -- strict flag gating -------------------------------------------------------


def test_sampling_off_rejects_token_bearing_before_worker() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker, sampling=False)

    with pytest.raises(RuntimeError, match="control-only"):
        wrapper.execute_model(_token_step())

    assert worker.execute_calls == 0
    assert wrapper._request_owned_deferred is None


def test_sampling_absent_defaults_off() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)

    with pytest.raises(RuntimeError, match="control-only"):
        wrapper.execute_model(_token_step())

    assert worker.execute_calls == 0


@pytest.mark.parametrize("bad", [1, "true", 0.0, None, []])
def test_non_bool_sampling_flag_fails_closed(bad) -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker, sampling=bad)

    with pytest.raises(RuntimeError, match="must be a bool"):
        wrapper.execute_model(_output())

    assert worker.execute_calls == 0


def test_sampling_off_sample_tokens_fails_closed() -> None:
    wrapper = _wrapper(0, sampling=False)

    with pytest.raises(RuntimeError, match="without enable_request_owned_sampling"):
        wrapper.sample_tokens(object())


# -- mark exactly once / replay ----------------------------------------------


def test_sample_tokens_without_pending_raises() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker, sampling=True)

    with pytest.raises(RuntimeError, match="requires a pending deferred step"):
        wrapper.sample_tokens(object())

    assert worker.sample_calls == 0


def test_duplicate_sample_tokens_after_success_raises() -> None:
    worker = _FakeWorker(execute_output=None)
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)
    wrapper.execute_model(_token_step())
    wrapper.sample_tokens(object())
    assert store.mark_calls == 1

    with pytest.raises(RuntimeError, match="requires a pending deferred step"):
        wrapper.sample_tokens(object())

    assert worker.sample_calls == 1
    assert store.mark_calls == 1


@pytest.mark.parametrize(
    "bad_output, match",
    [
        (None, "returned None"),
        (_FakeAsyncOutput(), "async model runner outputs from sample_tokens"),
        (object(), "unexpected"),
    ],
)
def test_deferred_bad_sample_output_fails_and_keeps_pending(bad_output, match) -> None:
    worker = _FakeWorker(execute_output=None, sample_output=bad_output)
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)
    wrapper.execute_model(_token_step())
    pending = wrapper._request_owned_deferred

    with pytest.raises(RuntimeError, match=match):
        wrapper.sample_tokens(object())

    assert wrapper._request_owned_deferred is pending
    assert wrapper._request_owned_control_manager is None
    assert store.mark_calls == 0


# -- mark rejection is atomic and retryable ----------------------------------


def test_mark_rejection_keeps_pending_and_is_retryable() -> None:
    worker = _FakeWorker(execute_output=None)
    store = _FakeStore(0, reject_mark=True)
    wrapper = _wrapper(0, worker, store, sampling=True)
    wrapper.execute_model(_token_step())
    pending = wrapper._request_owned_deferred

    with pytest.raises(RuntimeError, match="computed batch mark failed"):
        wrapper.sample_tokens(object())

    assert wrapper._request_owned_deferred is pending
    assert wrapper._request_owned_control_manager is None
    assert wrapper._request_owned_fail_stop is None
    assert store.calls == ["build", "mark"]
    assert "flush" not in store.calls
    # Still only the pending-step rejection on the next execute: not fail-stop.
    with pytest.raises(RuntimeError, match="pending deferred sampling step"):
        wrapper.execute_model(_token_step(step_seq=2))
    assert wrapper._request_owned_fail_stop is None

    # The rejected mark is atomic: the same pending step can be retried once
    # the store admits it, and then the terminal path completes normally.
    store.reject_mark = False
    result = wrapper.sample_tokens(object())
    assert store.mark_calls == 2
    assert wrapper._request_owned_deferred is None
    assert result.owner_receipt_batches is not None


# -- irreversible post-mark fail-stop ----------------------------------------


def test_post_mark_flush_failure_latches_fail_stop() -> None:
    worker = _FakeWorker(execute_output=None)
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)
    wrapper.execute_model(_token_step())

    def exploding_flush():
        store.calls.append("flush")
        raise RuntimeError("pool flush exploded")

    store.flush = exploding_flush
    with pytest.raises(RuntimeError, match="pool flush exploded"):
        wrapper.sample_tokens(object())

    assert store.mark_calls == 1
    assert wrapper._request_owned_fail_stop is not None
    assert "already marked and cannot be retried" in wrapper._request_owned_fail_stop
    assert wrapper._request_owned_control_manager is None

    # Every further request-owned call fails closed on the latch; the mark
    # is never retried.
    with pytest.raises(RuntimeError, match="irreversible fail-stop"):
        wrapper.sample_tokens(object())
    with pytest.raises(RuntimeError, match="irreversible fail-stop"):
        wrapper.execute_model(_token_step(step_seq=2))
    assert worker.sample_calls == 1
    assert worker.execute_calls == 1
    assert store.mark_calls == 1


def test_post_mark_pool_snapshot_failure_latches_fail_stop() -> None:
    worker = _FakeWorker(execute_output=None)
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)
    wrapper.execute_model(_token_step())

    def exploding_snapshot():
        store.calls.append("pool_snapshot")
        raise RuntimeError("snapshot exploded")

    store.pool_snapshot = exploding_snapshot
    with pytest.raises(RuntimeError, match="snapshot exploded"):
        wrapper.sample_tokens(object())

    assert store.mark_calls == 1
    assert "flush" in store.calls
    assert wrapper._request_owned_fail_stop is not None
    assert wrapper._request_owned_control_manager is None
    with pytest.raises(RuntimeError, match="irreversible fail-stop"):
        wrapper.execute_model(_token_step(step_seq=2))


def test_sync_post_mark_failure_latches_fail_stop() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)

    def exploding_flush():
        store.calls.append("flush")
        raise RuntimeError("pool flush exploded")

    store.flush = exploding_flush
    with pytest.raises(RuntimeError, match="pool flush exploded"):
        wrapper.execute_model(_token_step())

    assert store.mark_calls == 1
    assert wrapper._request_owned_fail_stop is not None
    assert wrapper._request_owned_control_manager is None
    with pytest.raises(RuntimeError, match="irreversible fail-stop"):
        wrapper.execute_model(_token_step(step_seq=2))


# -- zero-token heartbeat metadata mark --------------------------------------


def test_zero_token_step_with_sampling_on_marks_empty_metadata() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store, sampling=True)

    result = wrapper.execute_model(_output(step_seq=1))

    assert worker.execute_calls == 1
    assert store.mark_calls == 1
    assert store.last_mark_metadata is not None
    assert store.last_mark_metadata.step_seq == 1
    assert store.last_mark_metadata.entries == ()
    assert result.owner_receipt_batches[0].emitted_step_seq == 1


def test_default_off_path_unchanged_and_never_marks() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)

    result = wrapper.execute_model(_output(step_seq=1))

    assert worker.execute_calls == 1
    assert store.mark_calls == 0
    assert "mark" not in store.calls
    assert result.owner_receipt_batches[0].emitted_step_seq == 1
    assert wrapper._request_owned_deferred is None


def test_speculative_completion_derives_verified_logical_commit() -> None:
    key = OwnerLeaseKey("spec", 2)
    metadata = RequestOwnedStepMetadata(
        step_seq=9,
        owner_rank=0,
        entries=(
            RequestOwnedStepEntry(
                key=key,
                allocation_generation=1,
                pre_step_num_computed_tokens=20,
                post_step_num_tokens=24,
                tables=((),),
                delta=((),),
                num_speculative_tokens=3,
            ),
        ),
    )
    output = ModelRunnerOutput(
        req_ids=["spec"],
        req_id_to_index={"spec": 0},
        sampled_token_ids=[[31, 99]],
    )

    assert WorkerWrapperBase._request_owned_committed_num_tokens(metadata, output) == {
        key: 22
    }

    output.sampled_token_ids = []
    with pytest.raises(RuntimeError, match="align 1:1"):
        WorkerWrapperBase._request_owned_committed_num_tokens(metadata, output)


def test_speculative_completion_rejects_misaligned_identity_before_mark() -> None:
    key = OwnerLeaseKey("spec", 2)
    metadata = RequestOwnedStepMetadata(
        step_seq=9,
        owner_rank=0,
        entries=(
            RequestOwnedStepEntry(
                key=key,
                allocation_generation=1,
                pre_step_num_computed_tokens=20,
                post_step_num_tokens=24,
                tables=((),),
                delta=((),),
                num_speculative_tokens=3,
            ),
        ),
    )
    output = ModelRunnerOutput(
        req_ids=["other"],
        req_id_to_index={"spec": 0},
        sampled_token_ids=[[31, 32, 99]],
    )
    store = _FakeStore(0)
    wrapper = _wrapper(0, store=store, sampling=True)

    with pytest.raises(RuntimeError, match="bijective and aligned"):
        wrapper._complete_request_owned_step(
            step_seq=9,
            trial_manager=AttentionLeaseManager(owner_rank=0, capacity=64),
            metadata=metadata,
            output=output,
        )

    assert store.mark_calls == 0
    assert "mark" not in store.calls


# -- default delegation ------------------------------------------------------


def test_sample_tokens_delegates_when_request_owned_attention_disabled() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_attention = False
    grammar = object()

    result = wrapper.sample_tokens(grammar)

    assert worker.sample_calls == 1
    assert worker.sample_grammar is grammar
    assert result is EMPTY_MODEL_RUNNER_OUTPUT
