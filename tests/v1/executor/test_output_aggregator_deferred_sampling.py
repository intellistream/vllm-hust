# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred-round semantics of the G3 owner-sampling aggregator.

CPU-only: exercises :class:`ModelRunnerOutputAggregator` across the deferred
``execute_model -> None -> sample_tokens`` flow.  With the sampling contract
enabled, an all-``None`` transport round (every worker deferred sampling) must
return ``None`` without demanding receipt or sampling envelopes, so the same
per-step adapter can be reused for the immediate terminal round; a mixed
``None``/non-``None`` round and the sampling-disabled aggregator must keep
failing closed exactly as before.
"""

import pytest

from vllm.v1.core.sched.owner_layout import GlobalRowId
from vllm.v1.core.sched.ownership import (
    OwnerLeaseKey,
    OwnerReceiptBatch,
)
from vllm.v1.executor.output_aggregator import ModelRunnerOutputAggregator
from vllm.v1.outputs import ModelRunnerOutput, OwnerSamplingBatch


def _row(
    request_id: str = "req-0",
    owner_epoch: int = 1,
    position: int = 3,
    lane: int = 0,
) -> GlobalRowId:
    return GlobalRowId(
        OwnerLeaseKey(request_id=request_id, owner_epoch=owner_epoch),
        position,
        lane,
    )


def _samp(owner_rank: int, req_ids: tuple[str, ...] = (), step: int = 7):
    return OwnerSamplingBatch(
        owner_rank=owner_rank,
        emitted_step_seq=step,
        row_ids=tuple(_row(rid, position=3) for rid in req_ids),
    )


def _output(owner_rank: int, req_ids: tuple[str, ...] = (), step: int = 7):
    req_ids = list(req_ids)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: idx for idx, rid in enumerate(req_ids)},
        sampled_token_ids=[[1] for _ in req_ids],
        owner_receipt_batches=[
            OwnerReceiptBatch(owner_rank=owner_rank, emitted_step_seq=step, events=())
        ],
        owner_sampling_batches=[_samp(owner_rank, tuple(req_ids), step=step)],
    )


def _aggregator(*ranks: int, sampling_ranks: list[int] | None = None):
    return ModelRunnerOutputAggregator(
        list(ranks), expected_sampling_owner_ranks=sampling_ranks
    )


def test_all_none_deferred_round_returns_none():
    """The deferred execute round: every transport slot is None, so the
    aggregator returns None without demanding receipt or sampling envelopes."""
    aggregator = _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2])
    assert aggregator.aggregate([None, None, None]) is None


def test_all_none_deferred_round_via_step_adapter():
    """The all-None round must not demand the bound step fence either: no
    envelopes exist yet to fence against."""
    adapter = _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).for_step(7)
    assert adapter.aggregate([None, None, None]) is None


def test_same_adapter_reused_for_terminal_round():
    """One per-step adapter serves both rounds: the all-None deferred round
    and then the terminal round carrying one receipt + one sampling batch per
    worker, fenced to the same exact step."""
    adapter = _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).for_step(7)
    assert adapter.aggregate([None, None, None]) is None

    outputs = [_output(rank, step=7) for rank in (0, 1, 2)]
    result = adapter.aggregate(outputs)
    assert result is not None
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]
    assert result.req_ids == []


def test_terminal_round_after_deferred_round_wrong_step_fails_closed():
    """Reusing the adapter does not relax the step fence: a terminal round
    emitting a different step than the bound execute round fails closed."""
    adapter = _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).for_step(7)
    assert adapter.aggregate([None, None, None]) is None

    outputs = [_output(rank, step=8) for rank in (0, 1, 2)]
    with pytest.raises(RuntimeError, match="does not match expected_step_seq"):
        adapter.aggregate(outputs)


def test_all_none_deferred_round_validates_sampling_cardinality():
    """The all-None fast path must still enforce the exact sampling transport
    cardinality: a truncated or padded deferred round is never accepted."""
    with pytest.raises(RuntimeError, match="expected 3"):
        _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).aggregate([None])
    with pytest.raises(RuntimeError, match="expected 3"):
        _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).aggregate(
            [None, None, None, None]
        )


def test_all_none_deferred_round_validates_owner_cardinality():
    """The all-None fast path also enforces the owner receipt contract
    cardinality when the sampling ranks alone would match the slot count."""
    with pytest.raises(RuntimeError, match="owner receipt contract"):
        ModelRunnerOutputAggregator(
            [0, 1, 2], expected_sampling_owner_ranks=[0, 1]
        ).aggregate([None, None])


def test_all_none_deferred_round_empty_contract_fails_closed():
    """An enabled-but-empty sampling contract fails closed even on an
    all-None round: any transport slot is a contract violation."""
    with pytest.raises(RuntimeError, match="expected 0"):
        ModelRunnerOutputAggregator([], expected_sampling_owner_ranks=[]).aggregate(
            [None]
        )


def test_mixed_none_round_fails_closed():
    """A mixed None/non-None round can never be a deferred round: the None
    slot cannot carry its envelopes, so it fails closed exactly as before."""
    outputs = [_output(0, step=7), None, _output(2, step=7)]
    with pytest.raises(RuntimeError, match="exactly one OwnerSamplingBatch"):
        _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_mixed_none_round_fails_closed_via_step_adapter():
    outputs = [None, None, _output(2, step=7)]
    with pytest.raises(RuntimeError, match="exactly one OwnerSamplingBatch"):
        _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).for_step(7).aggregate(outputs)


def test_empty_outputs_list_still_fails_closed():
    """An empty transport list is not an all-None round: it is still an
    invalid aggregation input."""
    with pytest.raises(RuntimeError, match="at least one"):
        _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).aggregate([])


def test_sampling_disabled_all_none_still_demands_receipts():
    """Without the sampling contract, the aggregator keeps its existing
    behavior: an all-None round still fails closed because receipts are
    demanded (no deferred tolerance exists)."""
    with pytest.raises(RuntimeError, match="exactly one OwnerReceiptBatch"):
        _aggregator(0, 1, 2).aggregate([None, None, None])


def test_empty_batches_terminal_round_merges():
    """Empty owner batches (no requests scheduled for any owner) still
    satisfy the exact-one-batch-per-slot contract and merge to an empty
    output."""
    outputs = [_output(rank, step=7) for rank in (0, 1, 2)]
    result = _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.req_ids == []
    assert result.sampled_token_ids == []
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]


def test_all_none_round_does_not_consume_aggregator_state():
    """The all-None round is stateless on the shared aggregator: the same
    aggregator still serves unrelated adapters and terminal rounds."""
    aggregator = _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2])
    assert aggregator.aggregate([None, None, None]) is None
    assert aggregator.for_step(5).aggregate([None, None, None]) is None
    outputs = [_output(rank, step=9) for rank in (0, 1, 2)]
    result = aggregator.for_step(9).aggregate(outputs)
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]
