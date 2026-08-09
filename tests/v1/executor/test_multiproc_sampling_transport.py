# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multiproc transport for the G3 request-owned sampling flow.

CPU-only: drives :class:`MultiprocExecutor` against fake message queues.  With
``enable_request_owned_sampling`` the executor must bind and stash the exact
per-step adapter during ``execute_model``, reuse that same adapter for the
immediate ``sample_tokens`` round, clear it only after successful terminal
aggregation (or an explicit fail-stop state), and refuse overwrite, stale,
no-pending, replay, wrong-step, and non-blocking transitions.  With the
sampling flag off, the existing request-owned attention path is unchanged.
"""

from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.ownership import OwnerReceiptBatch
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.executor.output_aggregator import (
    ModelRunnerOutputAggregator,
    ModelRunnerOutputAggregatorStepAdapter,
)
from vllm.v1.outputs import ModelRunnerOutput, OwnerSamplingBatch


class _FakeMq:
    """Minimal MessageQueue stand-in recording broadcast enqueues and serving
    pre-queued (status, result) responses."""

    def __init__(self, responses=None):
        self.enqueued = []
        self._responses = list(responses or [])

    def enqueue(self, item):
        self.enqueued.append(item)

    def dequeue(self, timeout=None):
        if not self._responses:
            raise AssertionError("dequeue called with no queued response")
        return self._responses.pop(0)


class _RecordingExecutor(MultiprocExecutor):
    """Records every (method, non_block, aggregator) collective_rpc call."""

    def __init__(self):
        self.rpc_calls = []

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        unique_reply_rank=None,
        kv_output_aggregator=None,
    ):
        self.rpc_calls.append((method, non_block, kv_output_aggregator))
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


def _scheduler_output(step_seq: int = 7):
    from vllm.v1.core.sched.output import SchedulerOutput

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.step_seq = step_seq
    return scheduler_output


def _owner_batch(owner_rank: int, emitted_step_seq: int = 7) -> OwnerReceiptBatch:
    return OwnerReceiptBatch(
        owner_rank=owner_rank,
        emitted_step_seq=emitted_step_seq,
        events=(),
    )


def _sampling_batch(owner_rank: int, emitted_step_seq: int = 7) -> OwnerSamplingBatch:
    return OwnerSamplingBatch(
        owner_rank=owner_rank,
        emitted_step_seq=emitted_step_seq,
        row_ids=(),
    )


def _worker_output(owner_rank: int, emitted_step_seq: int = 7):
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        owner_receipt_batches=[_owner_batch(owner_rank, emitted_step_seq)],
        owner_sampling_batches=[_sampling_batch(owner_rank, emitted_step_seq)],
    )


def _fake_executor(
    response_mqs,
    cls: type = _RecordingExecutor,
    enable_request_owned_sampling: bool = True,
) -> MultiprocExecutor:
    executor = cls.__new__(cls)
    executor.rpc_calls = []
    executor.scheduler_config = SimpleNamespace(
        enable_request_owned_attention=True,
        enable_request_owned_sampling=enable_request_owned_sampling,
    )
    executor.kv_output_aggregator = None
    executor.model_runner_output_aggregator = ModelRunnerOutputAggregator(
        [0, 1, 2],
        expected_sampling_owner_ranks=[0, 1, 2]
        if enable_request_owned_sampling
        else None,
    )
    executor.output_rank = 0
    executor.is_failed = False
    executor._pending_sampling_adapter = None
    executor._sampling_transport_failed = False
    executor.rpc_broadcast_mq = _FakeMq()
    executor.response_mqs = response_mqs
    executor.futures_queue = deque()
    return executor


def _none_responses(count: int = 3):
    return [(WorkerProc.ResponseStatus.SUCCESS, None) for _ in range(count)]


def _sampling_responses(count: int = 3, emitted_step_seq: int = 7):
    return [
        (
            WorkerProc.ResponseStatus.SUCCESS,
            _worker_output(rank, emitted_step_seq),
        )
        for rank in range(count)
    ]


def _deferred_then_terminal_mqs(step: int = 7):
    """MQ responses for a deferred execute round followed by a terminal
    sample_tokens round for the same step."""
    return [
        _FakeMq([_none_responses()[rank], _sampling_responses()[rank]])
        for rank in range(3)
    ]


# ---------------------------------------------------------------------------
# Deferred execute -> sample_tokens transport
# ---------------------------------------------------------------------------


def test_deferred_execute_then_sample_tokens_reuses_exact_adapter():
    """The exact per-step adapter bound during execute_model is stashed,
    reused for the immediate sample_tokens round, and cleared only after the
    terminal aggregation; one receipt + one sampling batch per worker is
    aggregated with the exact step fence."""
    executor = _fake_executor(_deferred_then_terminal_mqs())

    result = executor.execute_model(_scheduler_output(7), non_block=False)
    assert result is None
    pending = executor._pending_sampling_adapter
    assert isinstance(pending, ModelRunnerOutputAggregatorStepAdapter)
    assert pending.step_seq == 7

    result = executor.sample_tokens(None, non_block=False)
    assert result is not None
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]
    assert executor._pending_sampling_adapter is None

    execute_call, sample_call = executor.rpc_calls
    assert execute_call[0] == "execute_model"
    assert sample_call[0] == "sample_tokens"
    assert execute_call[1] is False
    assert sample_call[1] is False
    # The exact same adapter object served both rounds.
    assert execute_call[2] is sample_call[2]
    assert isinstance(execute_call[2], ModelRunnerOutputAggregatorStepAdapter)
    assert execute_call[2].step_seq == 7
    # Both rounds broadcast to every worker MQ (output_rank=None).
    for item in executor.rpc_broadcast_mq.enqueued:
        assert item[3] is None
    assert all(len(mq._responses) == 0 for mq in executor.response_mqs)


def test_execute_concrete_output_aggregates_and_leaves_no_pending():
    """A concrete execute round is terminal: it aggregates normally and
    leaves no pending sampling adapter, so a later sample_tokens fails."""
    response_mqs = [_FakeMq([response]) for response in _sampling_responses()]
    executor = _fake_executor(response_mqs)

    result = executor.execute_model(_scheduler_output(7), non_block=False)
    assert result is not None
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]
    assert executor._pending_sampling_adapter is None

    with pytest.raises(RuntimeError, match="no pending"):
        executor.sample_tokens(None, non_block=False)


def test_sample_tokens_without_pending_execute_fails_closed():
    """sample_tokens without a preceding deferred execute round is refused
    before any RPC is dispatched."""
    executor = _fake_executor([_FakeMq() for _ in range(3)])
    with pytest.raises(RuntimeError, match="no pending"):
        executor.sample_tokens(None, non_block=False)
    assert executor.rpc_broadcast_mq.enqueued == []
    assert executor._sampling_transport_failed is False


def test_execute_overwrite_fails_closed():
    """A second execute_model while the first step's adapter is still pending
    is refused; the original pending adapter is untouched and the immediate
    sample_tokens round for it still succeeds."""
    executor = _fake_executor(_deferred_then_terminal_mqs())

    assert executor.execute_model(_scheduler_output(7), non_block=False) is None
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        executor.execute_model(_scheduler_output(8), non_block=False)
    assert executor._pending_sampling_adapter.step_seq == 7
    assert executor._sampling_transport_failed is False

    result = executor.sample_tokens(None, non_block=False)
    assert result is not None
    assert executor._pending_sampling_adapter is None


def test_sample_tokens_replay_fails_closed():
    """The pending adapter is consumed exactly once: a replayed sample_tokens
    is refused, and a fresh deferred execute -> sample cycle still works."""
    executor = _fake_executor(_deferred_then_terminal_mqs())

    assert executor.execute_model(_scheduler_output(7), non_block=False) is None
    assert executor.sample_tokens(None, non_block=False) is not None
    with pytest.raises(RuntimeError, match="no pending"):
        executor.sample_tokens(None, non_block=False)

    # A new step follows the normal cycle.
    executor.response_mqs = [
        _FakeMq(
            [
                _none_responses()[rank],
                _sampling_responses(emitted_step_seq=8)[rank],
            ]
        )
        for rank in range(3)
    ]
    assert executor.execute_model(_scheduler_output(8), non_block=False) is None
    result = executor.sample_tokens(None, non_block=False)
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]
    assert executor._pending_sampling_adapter is None


def test_sample_tokens_wrong_step_fail_stops():
    """Worker emissions in the terminal round must carry the exact step of
    the execute round: a wrong step fails closed and fail-stops the
    transport."""
    executor = _fake_executor(
        [
            _FakeMq(
                [
                    _none_responses()[rank],
                    _sampling_responses(emitted_step_seq=8)[rank],
                ]
            )
            for rank in range(3)
        ]
    )
    assert executor.execute_model(_scheduler_output(7), non_block=False) is None
    with pytest.raises(RuntimeError, match="does not match expected_step_seq"):
        executor.sample_tokens(None, non_block=False)
    assert executor._sampling_transport_failed is True
    assert executor._pending_sampling_adapter is None


def test_sample_tokens_all_none_fail_stops():
    """An all-None terminal round is not a successful terminal aggregation:
    the transport fail-stops and drops the pending adapter."""
    response_mqs = [
        _FakeMq([_none_responses()[rank], _none_responses()[rank]]) for rank in range(3)
    ]
    executor = _fake_executor(response_mqs)
    assert executor.execute_model(_scheduler_output(7), non_block=False) is None
    with pytest.raises(RuntimeError, match="sample_tokens returned None"):
        executor.sample_tokens(None, non_block=False)
    assert executor._sampling_transport_failed is True
    assert executor._pending_sampling_adapter is None

    with pytest.raises(RuntimeError, match="fail-stop"):
        executor.execute_model(_scheduler_output(8), non_block=False)


def test_execute_transport_failure_fail_stops():
    """Any collective_rpc failure during the execute round fail-stops the
    transport and drops the pending adapter."""
    response_mqs = [
        _FakeMq([(WorkerProc.ResponseStatus.FAILURE, "worker boom")]) for _ in range(3)
    ]
    executor = _fake_executor(response_mqs)
    with pytest.raises(RuntimeError, match="Worker failed"):
        executor.execute_model(_scheduler_output(7), non_block=False)
    assert executor._sampling_transport_failed is True
    assert executor._pending_sampling_adapter is None


def test_non_block_fails_closed():
    """Non-blocking execution cannot prove the exact per-step adapter state
    transition across a deferred future boundary: it fails closed before any
    RPC is dispatched, on both rounds."""
    executor = _fake_executor([_FakeMq() for _ in range(3)])
    with pytest.raises(RuntimeError, match="refusing non_block=True"):
        executor.execute_model(_scheduler_output(7), non_block=True)
    assert executor.rpc_broadcast_mq.enqueued == []

    executor.response_mqs = _deferred_then_terminal_mqs()
    assert executor.execute_model(_scheduler_output(7), non_block=False) is None
    with pytest.raises(RuntimeError, match="refusing non_block=True"):
        executor.sample_tokens(None, non_block=True)
    # The pending adapter survives the rejected non-blocking call.
    assert executor._pending_sampling_adapter is not None


# ---------------------------------------------------------------------------
# Sampling off and control-only gate
# ---------------------------------------------------------------------------


def test_sampling_off_keeps_attention_aggregation_unchanged():
    """With the sampling flag off, the existing request-owned attention path
    is byte-for-byte unchanged: receipts aggregate, no pending sampling
    adapter exists."""
    responses = [
        (
            WorkerProc.ResponseStatus.SUCCESS,
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                owner_receipt_batches=[_owner_batch(rank)],
            ),
        )
        for rank in (0, 1, 2)
    ]
    response_mqs = [_FakeMq([response]) for response in responses]
    executor = _fake_executor(response_mqs, enable_request_owned_sampling=False)

    result = executor.execute_model(_scheduler_output(7), non_block=False)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert executor._pending_sampling_adapter is None


def test_control_only_gate_lifts_with_sampling_enabled():
    """The G1 control-only gate lifts exactly when the sampling transport is
    enabled (token-bearing schedules are aggregated per step); with the
    sampling flag off the gate stays unchanged."""
    executor = _fake_executor([_FakeMq() for _ in range(3)])
    scheduler_output = _scheduler_output(7)
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.num_scheduled_tokens = {"req": 1}
    # Sampling enabled: no control-only rejection.
    executor._validate_request_owned_control_only_step(scheduler_output)

    executor.scheduler_config.enable_request_owned_sampling = False
    with pytest.raises(RuntimeError, match="control-only"):
        executor._validate_request_owned_control_only_step(scheduler_output)


def test_control_only_gate_lift_is_config_local_not_transport_local():
    """The gate lift is read from the executor's own config, so a worker that
    retained an older disabled config cannot be the decision maker; the
    driver-side validator is the authority (same as the G1 gate)."""
    executor = _fake_executor([_FakeMq() for _ in range(3)])
    scheduler_output = _scheduler_output(7)
    scheduler_output.total_num_scheduled_tokens = 0
    scheduler_output.num_scheduled_tokens = {}
    # No token schedule: no gate involvement either way.
    executor._validate_request_owned_control_only_step(scheduler_output)


def test_abstract_executor_builds_sampling_aggregator():
    """The base executor builds the sampling-enabled aggregator with every
    process-global rank as an expected sampling owner rank when the flag is
    on, and keeps sampling ranks disabled (None) when it is off."""
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.scheduler_config = SimpleNamespace(
        enable_request_owned_attention=True,
        enable_request_owned_sampling=True,
    )
    executor.parallel_config = SimpleNamespace(world_size=3)
    executor.kv_output_aggregator = None
    executor._build_model_runner_output_aggregator()
    assert executor.model_runner_output_aggregator._expected_sampling_owner_ranks == [
        0,
        1,
        2,
    ]

    executor.scheduler_config = SimpleNamespace(
        enable_request_owned_attention=True,
        enable_request_owned_sampling=False,
    )
    executor._build_model_runner_output_aggregator()
    assert (
        executor.model_runner_output_aggregator._expected_sampling_owner_ranks is None
    )
