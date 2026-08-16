# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for G3 owner-sampling envelope aggregation and all-worker merge.

CPU-only: exercises :class:`ModelRunnerOutputAggregator` with
:class:`OwnerSamplingBatch` envelopes, :class:`GlobalRowId` identities and
plain :class:`ModelRunnerOutput` values.  No GPU model runner and no NPU are
constructed.
"""

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.core.sched.owner_layout import GlobalRowId
from vllm.v1.core.sched.ownership import (
    OwnerLeaseKey,
    OwnerReceipt,
    OwnerReceiptBatch,
)
from vllm.v1.executor.output_aggregator import ModelRunnerOutputAggregator
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    ModelRunnerOutput,
    OwnerSamplingBatch,
    RoutedExpertsLists,
)


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


def _samp(
    owner_rank: int,
    req_ids: tuple[str, ...] = (),
    step: int = 7,
    position: int = 3,
) -> OwnerSamplingBatch:
    """Batch whose row_ids are aligned to req_ids at ``position``."""
    return OwnerSamplingBatch(
        owner_rank=owner_rank,
        emitted_step_seq=step,
        row_ids=tuple(_row(rid, position=position) for rid in req_ids),
    )


def _output(
    owner_rank: int,
    batches: list[OwnerSamplingBatch] | None,
    req_ids: tuple[str, ...] = (),
    sampled_token_ids: list[list[int]] | None = None,
    logprobs: LogprobsLists | None = None,
    num_nans: dict[str, int] | None = None,
    **fields,
) -> ModelRunnerOutput:
    req_ids = list(req_ids)
    if sampled_token_ids is None:
        sampled_token_ids = [[1] for _ in req_ids]
    else:
        sampled_token_ids = [list(t) for t in sampled_token_ids]
    output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: idx for idx, rid in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        num_nans_in_logits=num_nans,
        owner_sampling_batches=batches,
    )
    for name, value in fields.items():
        setattr(output, name, value)
    return output


def _aggregator(
    *ranks: int,
    kv_aggregator=None,
    sampling_ranks: list[int] | None = None,
) -> ModelRunnerOutputAggregator:
    return ModelRunnerOutputAggregator(
        list(ranks),
        kv_aggregator=kv_aggregator,
        expected_sampling_owner_ranks=sampling_ranks,
    )


def _three_owner_outputs(
    step: int = 7,
    sampling_ranks: list[int] | None = None,
    **fields,
) -> tuple[list[ModelRunnerOutput], list[OwnerSamplingBatch]]:
    """Ragged owners: rank 0 owns two requests, rank 1 none, rank 2 three."""
    req_ids_by_rank = {
        0: ("req-0a", "req-0b"),
        1: (),
        2: ("req-2a", "req-2b", "req-2c"),
    }
    outputs: list[ModelRunnerOutput] = []
    batches: list[OwnerSamplingBatch] = []
    for rank in sampling_ranks or [0, 1, 2]:
        req_ids = req_ids_by_rank[rank]
        batch = _samp(rank, req_ids, step=step)
        batches.append(batch)
        outputs.append(_output(rank, [batch], req_ids=req_ids, **fields))
    return outputs, batches


# ---------------------------------------------------------------------------
# Envelope and default-off behavior
# ---------------------------------------------------------------------------


def test_owner_sampling_batch_default_none_and_pickle():
    """ModelRunnerOutput keeps owner_sampling_batches=None by default and the
    envelope round-trips through pickle (the multiproc MQ wire format)."""
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert output.owner_sampling_batches is None
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_sampling_batches is None

    batch = _samp(1, ("req-a", "req-b"), step=9)
    output.owner_sampling_batches = [batch, _samp(2, (), step=9)]
    restored = pickle.loads(pickle.dumps(output))
    assert restored.owner_sampling_batches == output.owner_sampling_batches
    assert restored.owner_sampling_batches[0].row_ids == batch.row_ids
    assert restored.owner_sampling_batches[0].row_ids[0].request_uid == (
        OwnerLeaseKey(request_id="req-a", owner_epoch=1)
    )


def test_owner_sampling_batch_immutable():
    batch = _samp(0, ("req-0",))
    with pytest.raises(FrozenInstanceError):
        batch.owner_rank = 1
    with pytest.raises(FrozenInstanceError):
        batch.row_ids = ()


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"owner_rank": True}, TypeError),
        ({"owner_rank": -1}, ValueError),
        ({"emitted_step_seq": 0}, TypeError),
        ({"emitted_step_seq": True}, TypeError),
        ({"row_ids": [_row()]}, TypeError),
        ({"row_ids": ("not-a-row",)}, TypeError),
    ],
)
def test_owner_sampling_batch_validation(kwargs, exc):
    params = {"owner_rank": 0, "emitted_step_seq": 7}
    params.update(kwargs)
    with pytest.raises(exc):
        OwnerSamplingBatch(**params)


def test_default_off_unchanged():
    """Sampling-disabled aggregator with no batches anywhere is byte-for-byte
    the receipt/KV path: payload containers are shallow-shared with the
    selected output and owner_sampling_batches stays None."""
    outputs, _ = _three_owner_outputs(sampling_ranks=[0, 1, 2])
    for rank, output in enumerate(outputs):
        output.owner_sampling_batches = None
        output.owner_receipt_batches = [
            OwnerReceiptBatch(owner_rank=rank, emitted_step_seq=7, events=())
        ]
    selected = outputs[0]
    result = _aggregator(0, 1, 2).aggregate(outputs)
    assert result is not selected
    assert result.owner_sampling_batches is None
    assert result.req_ids is selected.req_ids
    assert result.req_id_to_index is selected.req_id_to_index
    assert result.sampled_token_ids is selected.sampled_token_ids
    assert result.logprobs is selected.logprobs
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_sampling_batches is None
    assert selected.owner_sampling_batches is None


def test_disabled_aggregator_rejects_sampling_batches():
    """Worker batches arriving at a sampling-disabled aggregator are never
    silently dropped or carried."""
    outputs, _ = _three_owner_outputs()
    with pytest.raises(RuntimeError, match="sampling aggregation is disabled"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_empty_sampling_ranks_contract_fails_closed():
    """An explicitly empty expected-sampling-ranks list is enabled-but-empty:
    any transport slot fails closed instead of silently disabling."""
    aggregator = ModelRunnerOutputAggregator(
        [], expected_sampling_owner_ranks=[]
    )
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    with pytest.raises(RuntimeError, match="expected 0"):
        aggregator.aggregate([output])


@pytest.mark.parametrize(
    "ranks, exc",
    [
        ([0, 1, 1], ValueError),
        ([True], TypeError),
        ([-1], TypeError),
    ],
)
def test_expected_sampling_owner_ranks_validation(ranks, exc):
    with pytest.raises(exc):
        ModelRunnerOutputAggregator(
            [], expected_sampling_owner_ranks=ranks
        )


# ---------------------------------------------------------------------------
# Merge correctness
# ---------------------------------------------------------------------------


def test_mixed_ragged_owners_merged_bijective():
    outputs, batches = _three_owner_outputs()
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.req_ids == ["req-0a", "req-0b", "req-2a", "req-2b", "req-2c"]
    assert result.req_id_to_index == {
        "req-0a": 0,
        "req-0b": 1,
        "req-2a": 2,
        "req-2b": 3,
        "req-2c": 4,
    }
    assert len(result.req_id_to_index) == len(result.req_ids) == 5
    assert result.sampled_token_ids == [[1]] * 5
    # Aggregated sampling batches preserved, sorted by numeric owner rank.
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]
    assert result.owner_sampling_batches[0] is batches[0]
    assert result.owner_sampling_batches[2] is batches[2]


def test_unsorted_expected_sampling_ranks_slot_mapping():
    """Expected ranks are normalized to the sorted contract exactly like the
    receipt path: transport slot i belongs to the i-th sorted rank."""
    outputs, _ = _three_owner_outputs(sampling_ranks=[0, 1, 2])
    result = _aggregator(sampling_ranks=[2, 0, 1]).aggregate(outputs)
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]
    # Slot 0 belongs to sorted rank 0; claiming rank 2 there fails.
    swapped = list(outputs)
    swapped[0], swapped[2] = swapped[2], swapped[0]
    with pytest.raises(RuntimeError, match="claims owner rank"):
        _aggregator(sampling_ranks=[2, 0, 1]).aggregate(swapped)


def test_chunked_prefill_one_sampling_identity():
    """A request with many scheduled prefill tokens still yields exactly one
    terminal sampling identity, not one per token."""
    req_id = "req-chunked"
    batch = OwnerSamplingBatch(
        owner_rank=0,
        emitted_step_seq=7,
        row_ids=(_row(req_id, position=199),),
    )
    output = _output(
        0,
        [batch],
        req_ids=(req_id,),
        sampled_token_ids=[[11]],
    )
    result = _aggregator(sampling_ranks=[0]).aggregate([output])
    assert len(batch.row_ids) == 1
    assert batch.row_ids[0].logical_token_position == 199
    assert result.req_ids == [req_id]
    assert result.sampled_token_ids == [[11]]
    assert result.owner_sampling_batches[0] is batch


def test_empty_zero_owner_batch():
    """A zero-owner still emits its (empty) envelope; the merged output has
    no requests from it but preserves the batch for scheduler validation."""
    outputs, batches = _three_owner_outputs()
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert batches[1].row_ids == ()
    assert result.owner_sampling_batches[1] is batches[1]
    assert result.owner_sampling_batches[1].row_ids == ()


def test_all_owners_empty():
    outputs = [
        _output(0, [_samp(0, (), step=7)], req_ids=()),
        _output(1, [_samp(1, (), step=7)], req_ids=()),
        _output(2, [_samp(2, (), step=7)], req_ids=()),
    ]
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.req_ids == []
    assert result.req_id_to_index == {}
    assert result.sampled_token_ids == []
    assert result.logprobs is None
    assert result.num_nans_in_logits is None
    assert [b.owner_rank for b in result.owner_sampling_batches] == [0, 1, 2]


def test_discarded_request_empty_token_list():
    """Discarded/no-token attempts stay aligned as [] in the merged output."""
    outputs, _ = _three_owner_outputs()
    outputs[0] = _output(
        0,
        [_samp(0, ("req-0a", "req-0b"))],
        req_ids=("req-0a", "req-0b"),
        sampled_token_ids=[[5], []],
    )
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.sampled_token_ids == [[5], [], [1], [1], [1]]


def test_spec_multi_token_sampled_ids_merge_without_logprobs():
    outputs = [
        _output(
            0,
            [_samp(0, ("req-0", "req-1"))],
            req_ids=("req-0", "req-1"),
            sampled_token_ids=[[1, 2, 3], [4]],
        ),
        _output(
            1,
            [_samp(1, ("req-2",))],
            req_ids=("req-2",),
            sampled_token_ids=[[5, 6]],
        ),
    ]
    result = _aggregator(sampling_ranks=[0, 1]).aggregate(outputs)
    assert result.req_ids == ["req-0", "req-1", "req-2"]
    assert result.sampled_token_ids == [[1, 2, 3], [4], [5, 6]]


def test_spec_multi_token_with_logprobs_fails_closed():
    output = _output(
        0,
        [_samp(0, ("req-0",))],
        req_ids=("req-0",),
        sampled_token_ids=[[1, 2]],
        logprobs=_logprobs(1),
    )
    with pytest.raises(RuntimeError, match="multi-token requests"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


def test_sampled_token_ids_alignment_fails_closed():
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        sampled_token_ids=[],
        owner_sampling_batches=[_samp(0, ("req-0",))],
    )
    with pytest.raises(RuntimeError, match="aligned 1:1"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


# ---------------------------------------------------------------------------
# Slot and step fencing
# ---------------------------------------------------------------------------


def test_exact_slot_owner_fencing():
    outputs, _ = _three_owner_outputs()
    outputs[1] = _output(1, [_samp(1, req_ids=(), step=7)], req_ids=())
    outputs[1].owner_sampling_batches = [_samp(0, (), step=7)]
    with pytest.raises(RuntimeError, match="claims owner rank"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_missing_batch_fails_closed():
    outputs, _ = _three_owner_outputs()
    outputs[1] = _output(1, None, req_ids=())
    with pytest.raises(RuntimeError, match="exactly one OwnerSamplingBatch"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_duplicate_batch_fails_closed():
    outputs, _ = _three_owner_outputs()
    outputs[1].owner_sampling_batches = [_samp(1, (), step=7), _samp(1, (), step=7)]
    with pytest.raises(RuntimeError, match="exactly one OwnerSamplingBatch"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_none_transport_slot_fails_closed():
    outputs, _ = _three_owner_outputs()
    outputs[1] = None
    with pytest.raises(RuntimeError, match="exactly one OwnerSamplingBatch"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_transport_slot_count_mismatch_fails_closed():
    outputs, _ = _three_owner_outputs()
    with pytest.raises(RuntimeError, match="expected 2"):
        _aggregator(sampling_ranks=[0, 1]).aggregate(outputs)


def test_mixed_step_seq_fails_closed():
    outputs, _ = _three_owner_outputs()
    outputs[1].owner_sampling_batches = [_samp(1, (), step=8)]
    with pytest.raises(RuntimeError, match="mixed emitted_step_seq"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_expected_step_seq_mismatch_fails_closed():
    outputs, _ = _three_owner_outputs(step=7)
    with pytest.raises(RuntimeError, match="does not match expected_step_seq"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(
            outputs, expected_step_seq=8
        )


def test_for_step_sampling_fence():
    outputs, _ = _three_owner_outputs(step=7)
    aggregator = _aggregator(sampling_ranks=[0, 1, 2])
    result = aggregator.for_step(7).aggregate(outputs)
    assert result.req_ids == ["req-0a", "req-0b", "req-2a", "req-2b", "req-2c"]
    with pytest.raises(RuntimeError, match="does not match expected_step_seq"):
        aggregator.for_step(8).aggregate(outputs)


# ---------------------------------------------------------------------------
# Duplicates and alignment
# ---------------------------------------------------------------------------


def test_duplicate_global_row_id_fails_closed():
    """The same terminal row identity claimed by two owners is fatal; checked
    before the (implied) duplicate request id."""
    same_row = _row("req-x", position=3)
    outputs = [
        _output(
            0,
            [OwnerSamplingBatch(0, 7, (same_row,))],
            req_ids=("req-x",),
        ),
        _output(
            1,
            [OwnerSamplingBatch(1, 7, (same_row,))],
            req_ids=("req-x",),
        ),
    ]
    with pytest.raises(RuntimeError, match="duplicate GlobalRowId"):
        _aggregator(sampling_ranks=[0, 1]).aggregate(outputs)


def test_duplicate_request_id_fails_closed():
    """Two owners executing the same request id (with different rows) is
    fatal."""
    outputs = [
        _output(0, [_samp(0, ("req-x",), position=3)], req_ids=("req-x",)),
        _output(1, [_samp(1, ("req-x",), position=4)], req_ids=("req-x",)),
    ]
    with pytest.raises(RuntimeError, match="duplicate request id"):
        _aggregator(sampling_ranks=[0, 1]).aggregate(outputs)


def test_row_request_length_misalignment_fails_closed():
    output = ModelRunnerOutput(
        req_ids=["req-0", "req-1"],
        req_id_to_index={"req-0": 0, "req-1": 1},
        sampled_token_ids=[[1], [1]],
        owner_sampling_batches=[
            OwnerSamplingBatch(0, 7, (_row("req-0"),)),
        ],
    )
    with pytest.raises(RuntimeError, match="1:1 alignment"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


def test_row_request_id_misalignment_fails_closed():
    output = _output(
        0,
        [OwnerSamplingBatch(0, 7, (_row("req-other"),))],
        req_ids=("req-0",),
    )
    with pytest.raises(RuntimeError, match="does not match partial output"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


def test_partial_req_id_to_index_inconsistent_fails_closed():
    output = ModelRunnerOutput(
        req_ids=["req-0", "req-1"],
        req_id_to_index={"req-0": 1, "req-1": 0},
        sampled_token_ids=[[1], [1]],
        owner_sampling_batches=[_samp(0, ("req-0", "req-1"))],
    )
    with pytest.raises(RuntimeError, match="bijective"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


# ---------------------------------------------------------------------------
# No mutation, receipt and KV composition
# ---------------------------------------------------------------------------


def test_no_input_or_singleton_mutation():
    outputs, batches = _three_owner_outputs()
    originals = [(list(o.req_ids), dict(o.req_id_to_index), list(o.sampled_token_ids))
                 for o in outputs]
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    for i, output in enumerate(outputs):
        assert output.req_ids == originals[i][0]
        assert output.req_id_to_index == originals[i][1]
        assert output.sampled_token_ids == originals[i][2]
        assert output.owner_sampling_batches is not None
        assert output.owner_sampling_batches[0] is batches[i]
    assert result not in outputs
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_sampling_batches is None
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None


def test_receipt_and_sampling_coexistence():
    receipt = OwnerReceipt(
        key=OwnerLeaseKey(request_id="req-0a", owner_epoch=1),
        owner_id=0,
        command_seq=1,
        accepted=True,
    )
    outputs, batches = _three_owner_outputs()
    for rank, output in enumerate(outputs):
        output.owner_receipt_batches = [
            OwnerReceiptBatch(
                owner_rank=rank,
                emitted_step_seq=7,
                events=(receipt,) if rank == 0 else (),
            )
        ]
    result = _aggregator(0, 1, 2, sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert result.owner_receipt_batches[0].events == (receipt,)
    assert result.owner_sampling_batches[0] is batches[0]
    assert result.req_ids == ["req-0a", "req-0b", "req-2a", "req-2b", "req-2c"]


def test_kv_composition_with_sampling():
    kv_aggregator = KVOutputAggregator(expected_finished_count=3)
    outputs, _ = _three_owner_outputs()
    for i, output in enumerate(outputs):
        output.kv_connector_output = KVConnectorOutput(
            finished_sending={"req_send"},
            invalid_block_ids={i},
        )
    result = _aggregator(
        kv_aggregator=kv_aggregator, sampling_ranks=[0, 1, 2]
    ).aggregate(outputs)
    assert result.kv_connector_output.finished_sending == {"req_send"}
    assert result.kv_connector_output.invalid_block_ids == {0, 1, 2}
    assert result.req_ids == ["req-0a", "req-0b", "req-2a", "req-2b", "req-2c"]
    for i, output in enumerate(outputs):
        assert output.kv_connector_output.invalid_block_ids == {i}
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


def test_kv_without_aggregator_fails_closed():
    output = _output(0, [_samp(0, ("req-0",))], req_ids=("req-0",))
    output.kv_connector_output = KVConnectorOutput(invalid_block_ids={1})
    with pytest.raises(RuntimeError, match="without a KV aggregator"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


# ---------------------------------------------------------------------------
# Logprobs and num_nans
# ---------------------------------------------------------------------------


def _logprobs(rows: int, max_num_logprobs: int = 3, base: int = 0):
    return LogprobsLists(
        np.arange(
            rows * max_num_logprobs, dtype=np.int64
        ).reshape(rows, max_num_logprobs)
        + base,
        np.full((rows, max_num_logprobs), 0.5, dtype=np.float32),
        np.arange(rows, dtype=np.int64) + base,
        None,
    )


def test_logprobs_merge_supported():
    outputs, _ = _three_owner_outputs()
    outputs[0].logprobs = _logprobs(2, base=0)
    outputs[2].logprobs = _logprobs(3, base=100)
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.logprobs is not None
    assert result.logprobs.cu_num_generated_tokens is None
    assert result.logprobs.logprob_token_ids.shape == (5, 3)
    assert result.logprobs.logprob_token_ids.tolist()[0][0] == 0
    assert result.logprobs.logprob_token_ids.tolist()[2][0] == 100
    assert result.logprobs.sampled_token_ranks.tolist() == [0, 1, 100, 101, 102]


def test_logprobs_with_zero_token_request_fails_closed():
    """Row cardinality is ambiguous once a discarded/no-token attempt exists
    (it may or may not contribute a logprobs row): fail closed instead of
    asserting rows == req_ids blindly."""
    outputs, _ = _three_owner_outputs()
    outputs[0] = _output(
        0,
        [_samp(0, ("req-0a", "req-0b"))],
        req_ids=("req-0a", "req-0b"),
        sampled_token_ids=[[5], []],
        logprobs=_logprobs(2, base=0),
    )
    outputs[2].logprobs = _logprobs(3, base=100)
    with pytest.raises(RuntimeError, match="zero-token/discarded"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_logprobs_partial_missing_fails_closed():
    outputs, _ = _three_owner_outputs()
    outputs[0].logprobs = _logprobs(2, base=0)
    # outputs[2] has requests but no logprobs while another partial has them.
    with pytest.raises(RuntimeError, match="every nonempty partial"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_logprobs_row_count_mismatch_fails_closed():
    output = _output(
        0,
        [_samp(0, ("req-0",))],
        req_ids=("req-0",),
        logprobs=_logprobs(2, base=0),
    )
    with pytest.raises(RuntimeError, match="row count"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


def test_spec_shaped_logprobs_fails_closed():
    logprobs = _logprobs(1, base=0)
    spec_logprobs = LogprobsLists(
        logprobs.logprob_token_ids,
        logprobs.logprobs,
        logprobs.sampled_token_ranks,
        [0],
    )
    output = _output(
        0,
        [_samp(0, ("req-0",))],
        req_ids=("req-0",),
        logprobs=spec_logprobs,
    )
    with pytest.raises(RuntimeError, match="cu_num_generated_tokens"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


def test_inconsistent_max_num_logprobs_fails_closed():
    outputs, _ = _three_owner_outputs()
    outputs[0].logprobs = _logprobs(2, max_num_logprobs=3, base=0)
    outputs[2].logprobs = _logprobs(3, max_num_logprobs=5, base=100)
    with pytest.raises(RuntimeError, match="max_num_logprobs"):
        _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)


def test_malformed_logprobs_fails_closed():
    logprobs = LogprobsLists(
        np.zeros((1, 3), dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((2,), dtype=np.int64),
        None,
    )
    output = _output(
        0,
        [_samp(0, ("req-0",))],
        req_ids=("req-0",),
        logprobs=logprobs,
    )
    with pytest.raises(RuntimeError, match="malformed logprobs"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


def test_num_nans_merge():
    outputs, _ = _three_owner_outputs()
    outputs[0].num_nans_in_logits = {"req-0a": 0, "req-0b": 2}
    outputs[2].num_nans_in_logits = {"req-2a": 1}
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.num_nans_in_logits == {"req-0a": 0, "req-0b": 2, "req-2a": 1}


def test_num_nans_all_none_stays_none():
    outputs, _ = _three_owner_outputs()
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.num_nans_in_logits is None


def test_num_nans_empty_dict_preserved():
    outputs, _ = _three_owner_outputs()
    outputs[0].num_nans_in_logits = {}
    result = _aggregator(sampling_ranks=[0, 1, 2]).aggregate(outputs)
    assert result.num_nans_in_logits == {}


def test_num_nans_foreign_key_fails_closed():
    output = _output(0, [_samp(0, ("req-0",))], req_ids=("req-0",))
    output.num_nans_in_logits = {"foreign-req": 1}
    with pytest.raises(RuntimeError, match="foreign key"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


def test_num_nans_invalid_value_fails_closed():
    output = _output(0, [_samp(0, ("req-0",))], req_ids=("req-0",))
    output.num_nans_in_logits = {"req-0": -1}
    with pytest.raises(RuntimeError, match="nonnegative"):
        _aggregator(sampling_ranks=[0]).aggregate([output])


# ---------------------------------------------------------------------------
# Unsupported payload fields fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("prompt_logprobs_dict", {"req-0": None}, "prompt_logprobs_dict"),
        ("pooler_output", [None], "pooler_output"),
        ("ec_connector_output", ECConnectorOutput(), "ec_connector_output"),
        ("cudagraph_stats", object(), "cudagraph_stats"),
        (
            "routed_experts",
            RoutedExpertsLists(np.zeros((1, 1, 1), dtype=np.int64), np.zeros(1)),
            "routed_experts",
        ),
    ],
)
def test_unsupported_payload_fields_fail_closed(field, value, match):
    output = _output(0, [_samp(0, ("req-0",))], req_ids=("req-0",))
    setattr(output, field, value)
    with pytest.raises(RuntimeError, match=match):
        _aggregator(sampling_ranks=[0]).aggregate([output])
