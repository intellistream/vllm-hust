# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for G0 request-owner receipt aggregation.

CPU-only: exercises :class:`ModelRunnerOutputAggregator` with protocol
dataclasses and plain :class:`ModelRunnerOutput` values.  No GPU model runner
and no NPU are constructed.
"""

import pickle
from functools import partial

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.core.sched.ownership import OwnerLeaseKey, OwnerReceipt, OwnerReceiptBatch
from vllm.v1.executor.output_aggregator import ModelRunnerOutputAggregator
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)


def _receipt(
    owner_rank: int,
    request_id: str = "req-0",
    owner_epoch: int = 1,
    command_seq: int = 1,
    accepted: bool = True,
    runnable_through: int = 10,
    released: bool = False,
    **kwargs,
) -> OwnerReceipt:
    return OwnerReceipt(
        key=OwnerLeaseKey(request_id=request_id, owner_epoch=owner_epoch),
        owner_id=owner_rank,
        command_seq=command_seq,
        accepted=accepted,
        runnable_through=runnable_through,
        released=released,
        **kwargs,
    )


def _batch(
    owner_rank: int,
    emitted_step_seq: int = 7,
    events: tuple[OwnerReceipt, ...] = (),
    **kwargs,
) -> OwnerReceiptBatch:
    return OwnerReceiptBatch(
        owner_rank=owner_rank,
        emitted_step_seq=emitted_step_seq,
        events=tuple(events),
        **kwargs,
    )


def _output(
    owner_rank: int,
    batches: list[OwnerReceiptBatch] | None,
    empty: bool = False,
) -> ModelRunnerOutput:
    if empty:
        return ModelRunnerOutput(
            req_ids=[], req_id_to_index={}, owner_receipt_batches=batches
        )
    req_id = f"req-{owner_rank}"
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[1]],
        owner_receipt_batches=batches,
    )


def _three_worker_outputs(
    step_seq: int = 7,
    events_by_rank: dict[int, tuple[OwnerReceipt, ...]] | None = None,
) -> list[ModelRunnerOutput]:
    events_by_rank = events_by_rank or {}
    return [
        _output(0, [_batch(0, step_seq, events_by_rank.get(0, ()))], empty=True),
        _output(1, [_batch(1, step_seq, events_by_rank.get(1, ()))], empty=True),
        _output(2, [_batch(2, step_seq, events_by_rank.get(2, ()))], empty=True),
    ]


def _aggregator(*ranks: int, kv_aggregator=None) -> ModelRunnerOutputAggregator:
    return ModelRunnerOutputAggregator(list(ranks), kv_aggregator=kv_aggregator)


def test_selected_rank0_empty_output_with_events_survives():
    """Selected output with req_ids=[] is valid and returns that output
    carrying all batches."""
    outputs = _three_worker_outputs(
        events_by_rank={
            1: (_receipt(1, request_id="req-a"),),
            2: (_receipt(2, request_id="req-b"),),
        }
    )
    result = _aggregator(0, 1, 2).aggregate(outputs)
    assert result.req_ids == []
    assert result.req_id_to_index == {}
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert result.owner_receipt_batches[1].events == (_receipt(1, request_id="req-a"),)
    assert result.owner_receipt_batches[2].events == (_receipt(2, request_id="req-b"),)


def test_all_rank_empty_events_survives():
    outputs = _three_worker_outputs()
    result = _aggregator(0, 1, 2).aggregate(outputs)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert all(b.events == () for b in result.owner_receipt_batches)


def test_authenticated_transport_slots_preserve_event_order():
    """Each slot carries its own rank's batch and event order is preserved."""
    ev_a = _receipt(1, request_id="req-a", command_seq=1)
    ev_b = _receipt(1, request_id="req-a", command_seq=2)
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(1, events=(ev_a, ev_b))], empty=True),
        _output(2, [_batch(2, events=(_receipt(2, request_id="req-z"),))], empty=True),
    ]
    result = _aggregator(0, 1, 2).aggregate(outputs)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert result.owner_receipt_batches[1].events == (ev_a, ev_b)


def test_transport_slot_cannot_impersonate_another_owner() -> None:
    outputs = [
        _output(0, [_batch(1)], empty=True),
        _output(1, [_batch(0)], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="transport slot 0"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_transport_slot_cannot_supply_multiple_owner_batches() -> None:
    outputs = [
        _output(0, [_batch(0), _batch(1)], empty=True),
        _output(1, None, empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="exactly one OwnerReceiptBatch"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_no_singleton_or_original_mutation():
    """The shared EMPTY_MODEL_RUNNER_OUTPUT singleton and the original worker
    outputs must never be mutated."""
    outputs = _three_worker_outputs()
    selected = outputs[0]
    result = _aggregator(0, 1, 2).aggregate(outputs)
    assert result is not selected
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None
    assert selected.owner_receipt_batches == [_batch(0)]
    assert selected.owner_receipt_batches is not result.owner_receipt_batches
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    # Shallow copy: non-mutated containers are shared with the selected output.
    assert result.sampled_token_ids is selected.sampled_token_ids
    assert result.req_ids == []


def test_missing_owner_rank_fails_closed():
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(1)], empty=True),
        _output(2, None, empty=True),
    ]
    with pytest.raises(RuntimeError, match="exactly one OwnerReceiptBatch"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_all_workers_disabled_fails_closed():
    """Enabled aggregator with no batches anywhere (feature disabled on every
    worker) is a missing-batch failure, never a silent no-op."""
    outputs = [
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    ]
    with pytest.raises(RuntimeError, match="exactly one OwnerReceiptBatch"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_duplicate_owner_rank_fails_closed():
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(1), _batch(1)], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="exactly one OwnerReceiptBatch"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_unexpected_owner_rank_fails_closed():
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(3)], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="claims owner rank"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_bool_owner_rank_fails_closed() -> None:
    outputs = [
        _output(0, [_batch(False)], empty=True),
        _output(1, [_batch(1)], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="owner_rank must be"):
        _aggregator(0, 1, 2).aggregate(outputs)


@pytest.mark.parametrize("step_seq", [0, -1, True, False, 7.0, "7"])
def test_invalid_emitted_step_seq_fails_closed(step_seq) -> None:
    outputs = [
        _output(0, [_batch(0, emitted_step_seq=step_seq)], empty=True),
        _output(1, [_batch(1)], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="positive non-bool"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_mixed_step_seq_fails_closed():
    outputs = [
        _output(0, [_batch(0, emitted_step_seq=7)], empty=True),
        _output(1, [_batch(1, emitted_step_seq=8)], empty=True),
        _output(2, [_batch(2, emitted_step_seq=7)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="mixed emitted_step_seq"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_expected_step_seq_mismatch_fails_closed():
    outputs = _three_worker_outputs(step_seq=7)
    aggregator = _aggregator(0, 1, 2)
    with pytest.raises(RuntimeError, match="expected_step_seq"):
        aggregator.aggregate(outputs, expected_step_seq=8)
    result = aggregator.aggregate(outputs, expected_step_seq=7)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]


@pytest.mark.parametrize("expected", [True, False, 0, -1, 7.0, "7"])
def test_invalid_expected_step_seq_fails_closed(expected) -> None:
    with pytest.raises(RuntimeError, match="expected_step_seq must be"):
        _aggregator(0, 1, 2).aggregate(
            _three_worker_outputs(step_seq=7), expected_step_seq=expected
        )


def test_for_step_correct_step_aggregates():
    """for_step(step_seq) binds the exact step: matching worker emissions
    aggregate normally through the per-step adapter."""
    outputs = _three_worker_outputs(step_seq=7)
    adapter = _aggregator(0, 1, 2).for_step(7)
    result = adapter.aggregate(outputs)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert all(b.emitted_step_seq == 7 for b in result.owner_receipt_batches)


def test_for_step_stale_and_future_mismatch_fails_closed():
    """Stale (older) and future (newer) worker emissions than the bound step
    both fail closed with the expected_step_seq error."""
    outputs = _three_worker_outputs(step_seq=7)
    stale = _aggregator(0, 1, 2).for_step(6)
    future = _aggregator(0, 1, 2).for_step(8)
    with pytest.raises(RuntimeError, match="expected_step_seq"):
        stale.aggregate(outputs)
    with pytest.raises(RuntimeError, match="expected_step_seq"):
        future.aggregate(outputs)


def test_for_step_wrappers_independent_and_shared_aggregator_stateless():
    """Adapters for different steps are independent and the shared
    aggregator stores no per-step state: creating or using one adapter never
    re-binds another adapter or the shared aggregator's default unbound
    behavior."""
    shared = _aggregator(0, 1, 2)
    step7 = shared.for_step(7)
    step8 = shared.for_step(8)
    assert step7.step_seq == 7
    assert step8.step_seq == 8

    outputs7 = _three_worker_outputs(step_seq=7)
    outputs8 = _three_worker_outputs(step_seq=8)
    assert [b.owner_rank for b in step7.aggregate(outputs7).owner_receipt_batches] == [
        0,
        1,
        2,
    ]
    assert [b.owner_rank for b in step8.aggregate(outputs8).owner_receipt_batches] == [
        0,
        1,
        2,
    ]
    # Using step8 did not re-bind step7...
    with pytest.raises(RuntimeError, match="expected_step_seq"):
        step8.aggregate(outputs7)
    assert [b.owner_rank for b in step7.aggregate(outputs7).owner_receipt_batches] == [
        0,
        1,
        2,
    ]
    # ...nor the shared aggregator: unbound aggregation still accepts any
    # single consistent emission step.
    assert [b.owner_rank for b in shared.aggregate(outputs7).owner_receipt_batches] == [
        0,
        1,
        2,
    ]
    assert [b.owner_rank for b in shared.aggregate(outputs8).owner_receipt_batches] == [
        0,
        1,
        2,
    ]


def test_shared_aggregator_holds_no_step_state():
    """The shared aggregator never carries the per-step binding; only the
    immutable adapter does."""
    shared = _aggregator(0, 1, 2)
    adapter = shared.for_step(7)
    assert not hasattr(shared, "expected_step_seq")
    assert adapter.step_seq == 7
    # The adapter is stateless beyond its bound step: repeated and interleaved
    # calls keep the same binding.
    outputs = _three_worker_outputs(step_seq=7)
    assert [b.owner_rank for b in adapter.aggregate(outputs).owner_receipt_batches] == [
        0,
        1,
        2,
    ]


def test_for_step_adapter_delattr_rejected():
    """Attribute deletion is unconditionally rejected: deleting _frozen would
    unlock mutation and deleting _step_seq would break the binding, so both
    (and any other attribute) fail closed and the binding stays unchanged."""
    shared = _aggregator(0, 1, 2)
    adapter = shared.for_step(7)

    with pytest.raises(AttributeError, match="immutable"):
        del adapter._frozen
    with pytest.raises(AttributeError, match="immutable"):
        del adapter._step_seq
    with pytest.raises(AttributeError, match="immutable"):
        del adapter._aggregator

    # The adapter is still frozen and still bound to the original step.
    assert adapter.step_seq == 7
    with pytest.raises(AttributeError, match="immutable"):
        adapter._step_seq = 99
    outputs = _three_worker_outputs(step_seq=7)
    assert [b.owner_rank for b in adapter.aggregate(outputs).owner_receipt_batches] == [
        0,
        1,
        2,
    ]
    # The shared aggregator is untouched.
    assert not hasattr(shared, "expected_step_seq")


def test_for_step_rejects_non_int_step_seq():
    """for_step fails closed on non-int step values."""
    aggregator = _aggregator(0, 1, 2)
    for bad in ("7", 7.0, None, [7]):
        with pytest.raises(TypeError, match="step_seq"):
            aggregator.for_step(bad)


def test_for_step_rejects_bool_step_seq():
    """bool is an int subclass but never a valid step sequence."""
    aggregator = _aggregator(0, 1, 2)
    for bad in (True, False):
        with pytest.raises(TypeError, match="step_seq"):
            aggregator.for_step(bad)


def test_for_step_rejects_nonpositive_step_seq():
    aggregator = _aggregator(0, 1, 2)
    for bad in (0, -1, -7):
        with pytest.raises(ValueError, match="positive"):
            aggregator.for_step(bad)


def test_expected_owner_ranks_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        ModelRunnerOutputAggregator([0, 1, 1])
    for bad in ([False, 1], [-1, 1], [0, "1"]):
        with pytest.raises(TypeError, match="nonnegative non-bool"):
            ModelRunnerOutputAggregator(bad)


def test_for_step_adapter_frozen_after_construction():
    """The adapter is immutable after construction: the bound step and the
    aggregator cannot be reassigned, the class-level step_seq property cannot
    be overwritten, no new attributes can be added, and the adapter cannot be
    unfrozen.  The binding survives all failed mutation attempts and the
    shared aggregator stays untouched."""
    shared = _aggregator(0, 1, 2)
    adapter = shared.for_step(7)

    with pytest.raises(AttributeError, match="immutable"):
        adapter._step_seq = 99
    with pytest.raises(AttributeError, match="immutable"):
        adapter._aggregator = None
    with pytest.raises(AttributeError, match="immutable"):
        adapter.step_seq = 99
    with pytest.raises(AttributeError, match="immutable"):
        adapter.extra_attr = 1
    with pytest.raises(AttributeError, match="immutable"):
        adapter._frozen = False

    assert adapter.step_seq == 7
    outputs = _three_worker_outputs(step_seq=7)
    assert [b.owner_rank for b in adapter.aggregate(outputs).owner_receipt_batches] == [
        0,
        1,
        2,
    ]
    # The shared aggregator carries no per-step state after the attempts.
    assert not hasattr(shared, "expected_step_seq")


def test_for_step_adapter_sync_and_future_duck_typing():
    """The adapter matches the KVOutputAggregator duck-typed surface used by
    sync and FutureWrapper paths: direct ``aggregate(outputs,
    output_rank=...)`` and an output_rank-pre-bound partial."""
    outputs = _three_worker_outputs(
        step_seq=7,
        events_by_rank={0: (_receipt(0, request_id="req-0"),)},
    )
    adapter = _aggregator(0, 1, 2).for_step(7)

    # Sync path: direct call; output_rank selects the output carrier.
    result = adapter.aggregate(outputs, output_rank=2)
    assert result.req_ids == []
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]

    # FutureWrapper path: aggregate pre-bound to output_rank the same way
    # multiproc collective_rpc builds its partial.
    bound = partial(adapter.aggregate, output_rank=0)
    assert [b.owner_rank for b in bound(outputs).owner_receipt_batches] == [0, 1, 2]


def test_for_step_adapter_kv_composition_unchanged():
    """Existing KV connector composition stays identical through the per-step
    adapter."""
    kv_aggregator = KVOutputAggregator(expected_finished_count=3)
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(1, events=(_receipt(1, request_id="req-a"),))], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    for i, output in enumerate(outputs):
        output.kv_connector_output = KVConnectorOutput(
            finished_sending={"req_send"},
            invalid_block_ids={i},
        )

    adapter = _aggregator(0, 1, 2, kv_aggregator=kv_aggregator).for_step(7)
    result = adapter.aggregate(outputs)

    assert result.kv_connector_output.finished_sending == {"req_send"}
    assert result.kv_connector_output.invalid_block_ids == {0, 1, 2}
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    for i, output in enumerate(outputs):
        assert output.kv_connector_output.finished_sending == {"req_send"}
        assert output.kv_connector_output.invalid_block_ids == {i}
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


def test_exact_duplicate_event_deduped():
    """Exact payload replay is idempotent: equal payloads (even as distinct
    objects) with the same identity collapse to one event."""
    ev = _receipt(1, request_id="req-a", command_seq=3)
    duplicate = _receipt(1, request_id="req-a", command_seq=3)
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(1, events=(ev, duplicate, ev))], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    result = _aggregator(0, 1, 2).aggregate(outputs)
    assert result.owner_receipt_batches[1].events == (ev,)


def test_event_owner_id_must_match_batch_owner_rank():
    """A worker cannot spoof or misroute another owner's receipt: an event
    whose owner_id differs from its enclosing batch owner_rank fails closed."""
    spoofed = _receipt(2, request_id="req-a")
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(1, events=(spoofed,))], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="does not match enclosing batch"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_bool_event_identity_cannot_alias_integer_identity() -> None:
    valid = _receipt(1, owner_epoch=1, command_seq=1)
    bool_owner = OwnerReceipt(
        key=OwnerLeaseKey("req-bool-owner", 1),
        owner_id=True,
        command_seq=2,
        accepted=False,
    )
    outputs = _three_worker_outputs(events_by_rank={1: (valid, bool_owner)})
    with pytest.raises(RuntimeError, match="does not match enclosing batch"):
        _aggregator(0, 1, 2).aggregate(outputs)

    # OwnerLeaseKey now rejects a bool epoch before it can hash/equal-alias 1.
    with pytest.raises(TypeError, match="owner_epoch"):
        OwnerLeaseKey("req", True)

    bool_seq = OwnerReceipt(
        key=OwnerLeaseKey("req-bool-seq", 1),
        owner_id=1,
        command_seq=True,
        accepted=False,
    )
    outputs = _three_worker_outputs(events_by_rank={1: (bool_seq,)})
    with pytest.raises(RuntimeError, match="command_seq"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_conflicting_duplicate_event_fatal():
    """Same identity with a conflicting payload raises instead of silently
    dropping or keeping one variant."""
    ev = _receipt(1, request_id="req-a", command_seq=3, runnable_through=10)
    conflicting = _receipt(1, request_id="req-a", command_seq=3, runnable_through=99)
    outputs = [
        _output(0, [_batch(0)], empty=True),
        _output(1, [_batch(1, events=(ev, conflicting))], empty=True),
        _output(2, [_batch(2)], empty=True),
    ]
    with pytest.raises(RuntimeError, match="conflicting duplicate OwnerReceipt"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_kv_and_owner_composition():
    """KV connector composition and owner receipt aggregation compose on one
    outputs list without mutating originals or the singleton."""
    kv_aggregator = KVOutputAggregator(expected_finished_count=3)
    outputs = [
        _output(
            0,
            [_batch(0)],
            empty=True,
        ),
        _output(
            1,
            [_batch(1, events=(_receipt(1, request_id="req-a"),))],
            empty=True,
        ),
        _output(
            2,
            [_batch(2)],
            empty=True,
        ),
    ]
    for i, output in enumerate(outputs):
        output.kv_connector_output = KVConnectorOutput(
            finished_sending={"req_send"},
            invalid_block_ids={i},
        )

    result = _aggregator(0, 1, 2, kv_aggregator=kv_aggregator).aggregate(outputs)

    # KV composition visible on the returned (copied) selected output.
    assert result.kv_connector_output.finished_sending == {"req_send"}
    assert result.kv_connector_output.invalid_block_ids == {0, 1, 2}
    # Owner batches carried alongside.
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]
    assert result.owner_receipt_batches[1].events == (_receipt(1, request_id="req-a"),)
    # Originals untouched.
    for i, output in enumerate(outputs):
        assert output.kv_connector_output.finished_sending == {"req_send"}
        assert output.kv_connector_output.invalid_block_ids == {i}
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


def test_kv_composition_keeps_singleton_and_originals_untouched():
    outputs = _three_worker_outputs()
    kv_aggregator = KVOutputAggregator(expected_finished_count=3)
    for i, output in enumerate(outputs):
        output.kv_connector_output = KVConnectorOutput(invalid_block_ids={i})

    result = _aggregator(0, 1, 2, kv_aggregator=kv_aggregator).aggregate(outputs)
    assert result.kv_connector_output is not None
    assert result.kv_connector_output.invalid_block_ids == {0, 1, 2}
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]


def test_none_transport_slot_fails_explicitly():
    outputs = _three_worker_outputs()
    outputs[0] = None
    with pytest.raises(RuntimeError, match="exactly one OwnerReceiptBatch"):
        _aggregator(0, 1, 2).aggregate(outputs)


def test_no_worker_outputs_fails_closed():
    with pytest.raises(RuntimeError, match="at least one worker output"):
        _aggregator(0, 1, 2).aggregate([])


def test_empty_expected_owner_ranks():
    aggregator = ModelRunnerOutputAggregator([])
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    result = aggregator.aggregate([output])
    assert result.owner_receipt_batches == []
    assert result is not output


def test_output_rank_selection():
    outputs = _three_worker_outputs(
        events_by_rank={0: (_receipt(0, request_id="req-0"),)}
    )
    result = _aggregator(0, 1, 2).aggregate(outputs, output_rank=2)
    assert result.req_ids == []
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]


def test_model_runner_output_default_and_pickle():
    """ModelRunnerOutput keeps owner_receipt_batches=None by default and
    round-trips batches through pickle (the wire format used by the
    multiproc MQs)."""
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert output.owner_receipt_batches is None
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None

    output.owner_receipt_batches = [
        _batch(1, events=(_receipt(1, request_id="req-a"),)),
        _batch(2),
    ]
    restored = pickle.loads(pickle.dumps(output))
    assert restored.owner_receipt_batches == output.owner_receipt_batches
    assert restored.owner_receipt_batches[0].events == (
        _receipt(1, request_id="req-a"),
    )
