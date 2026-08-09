# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU tests for the G1 owner-layout host/reference codec.

Exercises :class:`GlobalRowId`, :class:`ExecutionRow`, and
:class:`OwnerRowLayout` with epoch-distinct request uids, noncontiguous
group ranks, every owner-count distribution, the exact permutation
invariants, list and CPU torch round trips, padded-buffer logical-prefix
handling, and every fail-closed case.  No GPU or model runner is built.
"""

import pytest
import torch

from vllm.v1.core.sched.owner_layout import (
    ExecutionRow,
    GlobalRowId,
    OwnerCollectivePlan,
    OwnerLayoutError,
    OwnerRowLayout,
)
from vllm.v1.core.sched.ownership import OwnerLeaseKey


def _key(request_id: str = "req-0", epoch: int = 0) -> OwnerLeaseKey:
    return OwnerLeaseKey(request_id=request_id, owner_epoch=epoch)


def _rid(
    request_id: str = "req-0", position: int = 0, lane: int = 0, epoch: int = 0
) -> GlobalRowId:
    return GlobalRowId(_key(request_id, epoch), position, lane)


def _rows(
    request_ids: list[str],
    positions: list[int],
    lanes: list[int] | None = None,
    epochs: list[int] | None = None,
) -> list[GlobalRowId]:
    if lanes is None:
        lanes = [0] * len(request_ids)
    if epochs is None:
        epochs = [0] * len(request_ids)
    return [
        _rid(rid, position, lane, epoch)
        for rid, position, lane, epoch in zip(request_ids, positions, lanes, epochs)
    ]


def _owners(
    request_ids: list[str], mapping: dict[str, int]
) -> dict[OwnerLeaseKey, int]:
    return {_key(rid): owner for rid, owner in mapping.items() if rid in request_ids}


# -- identity types ----------------------------------------------------------


def test_global_row_id_rejects_negative_and_wrong_typed_fields() -> None:
    with pytest.raises(OwnerLayoutError):
        GlobalRowId(_key(), -1)
    with pytest.raises(OwnerLayoutError):
        GlobalRowId(_key(), 0, logical_lane=-2)
    with pytest.raises(OwnerLayoutError):
        GlobalRowId(_key(), "0")
    with pytest.raises(OwnerLayoutError):
        GlobalRowId("req-0", 0)
    # Nonnegative fields are accepted, lanes default to zero.
    assert GlobalRowId(_key(), 7) == GlobalRowId(_key(), 7, 0)
    assert GlobalRowId(_key(), 7, 0).logical_lane == 0


def test_global_row_id_is_frozen() -> None:
    row_id = _rid()
    with pytest.raises(AttributeError):
        row_id.logical_token_position = 1  # type: ignore[misc]


def test_execution_row_validates_fence_and_row() -> None:
    with pytest.raises(OwnerLayoutError):
        ExecutionRow(-1, _rid())
    with pytest.raises(OwnerLayoutError):
        ExecutionRow(3, _key())  # type: ignore[arg-type]
    row = ExecutionRow(3, _rid("req-0", 2))
    assert row.step_seq == 3
    assert row.row_id == _rid("req-0", 2)


def test_epoch_distinct_request_uids_are_distinct_rows() -> None:
    old_epoch = GlobalRowId(_key("req-0", 0), 4)
    new_epoch = GlobalRowId(_key("req-0", 1), 4)
    assert old_epoch != new_epoch
    # Same request id from a new epoch is a different row, not a duplicate.
    layout = OwnerRowLayout.build(
        0,
        [old_epoch, new_epoch],
        {old_epoch.request_uid: 7, new_epoch.request_uid: 2},
        (7, 2, 11),
    )
    assert layout.logical_len == 2


# -- basic layout and rank mapping -------------------------------------------


def test_basic_layout_with_nonsorted_noncontiguous_group() -> None:
    rows = _rows(["a", "a", "b", "b", "c"], [0, 1, 0, 1, 0])
    owner = _owners(["a", "b", "c"], {"a": 11, "b": 2, "c": 7})
    layout = OwnerRowLayout.build(5, rows, owner, (7, 2, 11))
    assert layout.step_seq == 5
    assert layout.group_ranks == (7, 2, 11)
    assert layout.world_size == 3
    assert layout.global_to_local == {7: 0, 2: 1, 11: 2}
    assert layout.local_rank_of_global(11) == 2
    assert layout.local_rank_of_global(7) == 0
    assert layout.global_rank_of_local(1) == 2
    assert layout.global_rank_of_local(0) == 7
    # Buckets: rank 7 -> [c0], rank 2 -> [b0, b1], rank 11 -> [a0, a1].
    assert layout.owner_counts == (1, 2, 2)
    assert layout.owner_offsets == (0, 1, 3, 5)
    assert layout.forward_permutation == (4, 2, 3, 0, 1)
    assert layout.inverse_permutation == (3, 4, 1, 2, 0)
    for index, owner_row in enumerate(layout.owner_rows):
        assert owner_row.step_seq == 5


def test_owner_major_order_differs_from_canonical_and_is_stable() -> None:
    # Canonical: a0 a1 a2 b0 b1 c0 c1 c2; owner-major: c0 c1 c2 b0 b1 a0 a1 a2.
    rows = _rows(["a"] * 3 + ["b"] * 2 + ["c"] * 3, [0, 1, 2, 0, 1, 0, 1, 2])
    owner = _owners(["a", "b", "c"], {"a": 11, "b": 2, "c": 7})
    layout = OwnerRowLayout.build(1, rows, owner, (7, 2, 11))
    assert layout.global_rows == tuple(ExecutionRow(1, row) for row in rows)
    assert [row.row_id for row in layout.owner_rows] == [
        _rid("c", 0),
        _rid("c", 1),
        _rid("c", 2),
        _rid("b", 0),
        _rid("b", 1),
        _rid("a", 0),
        _rid("a", 1),
        _rid("a", 2),
    ]
    assert layout.owner_counts == (3, 2, 3)
    assert layout.owner_offsets == (0, 3, 5, 8)
    # Stable canonical order inside every owner bucket.
    for local in range(layout.world_size):
        bucket = layout.forward_permutation[layout.owner_slice_for_local(local)]
        assert list(bucket) == sorted(bucket)


def test_owner_slices_match_counts_and_rows() -> None:
    rows = _rows(["a"] * 4 + ["b"] * 2, [0, 1, 2, 3, 0, 1])
    owner = _owners(["a", "b"], {"a": 7, "b": 2})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    for local in range(layout.world_size):
        owner_slice = layout.owner_slice_for_local(local)
        assert owner_slice.stop - owner_slice.start == layout.owner_counts[local]
        assert (
            layout.owner_rows[owner_slice]
            == layout.owner_rows[
                layout.owner_offsets[local] : layout.owner_offsets[local + 1]
            ]
        )


def test_unequal_multi_request_mixed_lanes() -> None:
    rows = _rows(
        ["a", "a", "a", "b", "b", "c"], [0, 1, 2, 0, 1, 0], lanes=[0, 1, 2, 0, 1, 0]
    )
    owner = _owners(["a", "b", "c"], {"a": 2, "b": 11, "c": 7})
    layout = OwnerRowLayout.build(2, rows, owner, (7, 2, 11))
    assert layout.logical_len == 6
    assert layout.owner_counts == (1, 3, 2)
    assert [row.row_id.logical_lane for row in layout.owner_rows] == [0, 0, 1, 2, 0, 1]


# -- count distributions -----------------------------------------------------


@pytest.mark.parametrize(
    "distribution",
    [
        [0, 6],
        [6, 0],
        [1, 5],
        [2, 40],
    ],
)
def test_two_owner_count_distributions(distribution: list[int]) -> None:
    first, second = distribution
    rows = _rows(
        ["a"] * first + ["b"] * second, list(range(first)) + list(range(second))
    )
    owner = _owners(["a", "b"], {"a": 3, "b": 9})
    layout = OwnerRowLayout.build(0, rows, owner, (3, 9))
    assert layout.owner_counts == (first, second)
    assert layout.owner_offsets == (0, first, first + second)
    assert layout.forward_permutation == tuple(
        list(range(first)) + list(range(first, first + second))
    )
    assert layout.inverse_permutation == tuple(range(first + second))
    _assert_permutation_invariants(layout)


def test_all_zero_counts_for_empty_logical_work() -> None:
    layout = OwnerRowLayout.build(0, [], {}, (3, 9))
    assert layout.owner_counts == (0, 0)
    assert layout.owner_offsets == (0, 0, 0)
    assert layout.forward_permutation == ()
    assert layout.inverse_permutation == ()
    assert layout.global_rows == ()
    assert layout.owner_rows == ()
    assert layout.forward([]) == []
    assert layout.restore([]) == []
    _assert_permutation_invariants(layout)


def test_severe_skew_over_three_owners() -> None:
    rows = _rows(["a"] * 100, list(range(100)))
    owner = _owners(["a"], {"a": 11})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    assert layout.owner_counts == (0, 0, 100)
    assert layout.owner_offsets == (0, 0, 0, 100)
    assert layout.forward_permutation == tuple(range(100))
    assert layout.inverse_permutation == tuple(range(100))
    _assert_permutation_invariants(layout)


# -- exact permutation invariants --------------------------------------------


def _assert_permutation_invariants(layout: OwnerRowLayout) -> None:
    forward = layout.forward_permutation
    inverse = layout.inverse_permutation
    assert len(forward) == len(inverse) == layout.logical_len
    for owner_index, canonical_index in enumerate(forward):
        assert layout.owner_rows[owner_index] == layout.global_rows[canonical_index]
        assert inverse[canonical_index] == owner_index
    for canonical_index, owner_index in enumerate(inverse):
        assert layout.global_rows[canonical_index] == layout.owner_rows[owner_index]
        assert forward[owner_index] == canonical_index


def test_exact_permutation_compositions() -> None:
    rows = _rows(
        ["a"] * 4 + ["b"] * 3 + ["c"] * 5 + ["d"] * 2,
        [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1],
    )
    owner = _owners(["a", "b", "c", "d"], {"a": 2, "b": 11, "c": 7, "d": 2})
    layout = OwnerRowLayout.build(9, rows, owner, (7, 2, 11))
    _assert_permutation_invariants(layout)
    # restore(forward(payload)) == payload for the row payload itself.
    assert layout.restore(layout.owner_rows) == list(layout.global_rows)
    assert layout.forward(layout.global_rows) == list(layout.owner_rows)


# -- list helpers ------------------------------------------------------------


def test_list_forward_restore_round_trip() -> None:
    rows = _rows(["a", "a", "b", "b", "c"], [0, 1, 0, 1, 0])
    owner = _owners(["a", "b", "c"], {"a": 11, "b": 2, "c": 7})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    payload = [f"row-{index}" for index in range(5)]
    expected_owner_major = [payload[index] for index in layout.forward_permutation]
    assert layout.forward(payload) == expected_owner_major
    assert layout.restore(layout.forward(payload)) == payload
    # Exact-length enforcement.
    with pytest.raises(OwnerLayoutError):
        layout.forward(payload[:-1])
    with pytest.raises(OwnerLayoutError):
        layout.forward(payload + ["extra"])
    with pytest.raises(OwnerLayoutError):
        layout.restore(payload[:-1])


def test_empty_list_round_trip() -> None:
    layout = OwnerRowLayout.build(0, [], {}, (7, 2, 11))
    assert layout.forward([]) == []
    assert layout.restore([]) == []


# -- tensor helpers (CPU torch only) -----------------------------------------


def test_tensor_1d_forward_restore_round_trip() -> None:
    rows = _rows(["a", "a", "b", "b", "c"], [0, 1, 0, 1, 0])
    owner = _owners(["a", "b", "c"], {"a": 11, "b": 2, "c": 7})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    tensor = torch.arange(5, dtype=torch.int64)
    expected = tensor[list(layout.forward_permutation)]
    assert torch.equal(layout.forward_tensor(tensor), expected)
    assert torch.equal(layout.restore_tensor(expected), tensor)
    assert layout.forward_tensor(tensor).dtype == tensor.dtype


def test_tensor_2d_round_trip_preserves_trailing_shape_and_dtype() -> None:
    rows = _rows(["a", "a", "b", "b", "c"], [0, 1, 0, 1, 0])
    owner = _owners(["a", "b", "c"], {"a": 11, "b": 2, "c": 7})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    tensor = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    forward = layout.forward_tensor(tensor)
    assert forward.shape == tensor.shape
    assert forward.dtype == tensor.dtype
    assert torch.equal(forward, tensor[list(layout.forward_permutation)])
    assert torch.equal(layout.restore_tensor(forward), tensor)


def test_tensor_round_trip_matches_list_helper() -> None:
    rows = _rows(["a"] * 3 + ["b"] * 2, [0, 1, 2, 0, 1])
    owner = _owners(["a", "b"], {"a": 2, "b": 7})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    tensor = torch.arange(5)
    payload = [int(value) for value in tensor.tolist()]
    assert layout.forward_tensor(tensor).tolist() == layout.forward(payload)
    assert layout.restore_tensor(tensor).tolist() == layout.restore(payload)


def test_padded_buffer_logical_prefix_handling() -> None:
    rows = _rows(["a", "a", "b", "b", "c"], [0, 1, 0, 1, 0])
    owner = _owners(["a", "b", "c"], {"a": 11, "b": 2, "c": 7})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    logical_len = layout.logical_len
    pad_len = 3
    tensor = torch.arange(logical_len + pad_len, dtype=torch.int64)
    forward = layout.forward_tensor(tensor, logical_len=logical_len)
    # The logical prefix is permuted; physical padding rows stay in place.
    assert torch.equal(forward[:logical_len], tensor[list(layout.forward_permutation)])
    assert torch.equal(forward[logical_len:], tensor[logical_len:])
    restored = layout.restore_tensor(forward, logical_len=logical_len)
    assert torch.equal(restored, tensor)
    # 2-D padded buffers keep dtype and trailing shape.
    padded_2d = torch.arange((logical_len + pad_len) * 3).reshape(
        logical_len + pad_len, 3
    )
    forward_2d = layout.forward_tensor(padded_2d, logical_len=logical_len)
    assert forward_2d.shape == padded_2d.shape
    assert torch.equal(
        layout.restore_tensor(forward_2d, logical_len=logical_len), padded_2d
    )
    # Fail closed: padded buffer without the declared logical prefix length.
    with pytest.raises(OwnerLayoutError):
        layout.forward_tensor(tensor)
    with pytest.raises(OwnerLayoutError):
        layout.forward_tensor(tensor, logical_len=logical_len + 1)
    with pytest.raises(OwnerLayoutError):
        layout.restore_tensor(tensor)
    with pytest.raises(OwnerLayoutError):
        layout.restore_tensor(tensor, logical_len=logical_len - 1)


def test_tensor_helpers_fail_closed_on_wrong_leading_dimension() -> None:
    rows = _rows(["a", "a", "b", "b", "c"], [0, 1, 0, 1, 0])
    owner = _owners(["a", "b", "c"], {"a": 11, "b": 2, "c": 7})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    with pytest.raises(OwnerLayoutError):
        layout.forward_tensor(torch.arange(4))
    with pytest.raises(OwnerLayoutError):
        layout.restore_tensor(torch.arange(6))
    with pytest.raises(OwnerLayoutError):
        layout.forward_tensor(torch.arange(5), logical_len=-1)
    with pytest.raises(OwnerLayoutError):
        layout.restore_tensor(torch.arange(3), logical_len=5)


# -- fail-closed construction ------------------------------------------------


def test_rejects_empty_group_ranks() -> None:
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, [], {}, [])
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, [_rid()], {_key(): 0}, [])


def test_rejects_duplicate_and_invalid_group_ranks() -> None:
    rows = [_rid()]
    owner = {_key(): 7}
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, rows, owner, (7, 7, 2))
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, rows, owner, (-1, 2))
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, rows, owner, (7, 2, "11"))  # type: ignore[list-item]


def test_rejects_unknown_owner_and_missing_mapping() -> None:
    rows = [_rid("a", 0), _rid("b", 0)]
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, rows, {_key("a"): 5, _key("b"): 2}, (7, 2, 11))
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, rows, {_key("a"): 7}, (7, 2, 11))
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, rows, {_key("a"): 7, _key("b"): -1}, (7, 2, 11))


def test_rejects_duplicate_global_row_id() -> None:
    rows = [_rid("a", 0), _rid("a", 0)]
    owner = {_key("a"): 7}
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    # Same request, same position, distinct lanes are different rows.
    lanes = [_rid("a", 0, 0), _rid("a", 0, 1)]
    layout = OwnerRowLayout.build(0, lanes, owner, (7, 2, 11))
    assert layout.logical_len == 2


def test_rejects_negative_step_seq_and_bad_row_type() -> None:
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(-1, [_rid()], {_key(): 7}, (7, 2))
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, [_key()], {_key(): 7}, (7, 2))  # type: ignore[list-item]


def test_rejects_invalid_local_and_global_rank_queries() -> None:
    layout = OwnerRowLayout.build(0, [], {}, (7, 2, 11))
    with pytest.raises(OwnerLayoutError):
        layout.owner_slice_for_local(-1)
    with pytest.raises(OwnerLayoutError):
        layout.owner_slice_for_local(3)
    with pytest.raises(OwnerLayoutError):
        layout.global_rank_of_local(3)
    with pytest.raises(OwnerLayoutError):
        layout.local_rank_of_global(999)
    with pytest.raises(OwnerLayoutError):
        layout.local_rank_of_global(-1)


# -- bool fail-closed hardening ----------------------------------------------


def test_rejects_bool_in_nonnegative_fields() -> None:
    with pytest.raises(OwnerLayoutError):
        GlobalRowId(_key(), True)
    with pytest.raises(OwnerLayoutError):
        GlobalRowId(_key(), 0, logical_lane=False)
    with pytest.raises(OwnerLayoutError):
        ExecutionRow(True, _rid())
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(False, [_rid()], {_key(): 7}, (7, 2))
    layout = OwnerRowLayout.build(0, [], {}, (7, 2, 11))
    with pytest.raises(OwnerLayoutError):
        layout.forward_tensor(torch.arange(3), logical_len=True)


def test_rejects_bool_group_ranks_and_owner_rank_values() -> None:
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, [_rid()], {_key(): 1}, (True, 2))
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, [_rid()], {_key(): False}, (1, 2))
    # Regression: True previously matched global rank 1 by hash equality.
    with pytest.raises(OwnerLayoutError):
        OwnerRowLayout.build(0, [_rid()], {_key(): True}, (1, 2))


def test_rejects_bool_rank_query_indices() -> None:
    layout = OwnerRowLayout.build(0, [], {}, (1, 2, 11))
    # Regression: True/False previously matched ranks 1 and 0 by hash.
    with pytest.raises(OwnerLayoutError):
        layout.local_rank_of_global(True)
    with pytest.raises(OwnerLayoutError):
        layout.global_rank_of_local(False)
    with pytest.raises(OwnerLayoutError):
        layout.owner_slice_for_local(True)
    with pytest.raises(OwnerLayoutError):
        layout.owner_slice_for_local(False)


def test_owner_row_layout_is_immutable_after_construction() -> None:
    rows = _rows(["a", "a", "b"], [0, 1, 0])
    owner = _owners(["a", "b"], {"a": 2, "b": 7})
    layout = OwnerRowLayout.build(0, rows, owner, (7, 2, 11))
    with pytest.raises(AttributeError):
        layout._step_seq = 9
    with pytest.raises(AttributeError):
        layout._forward = (2, 0, 1)
    with pytest.raises(AttributeError):
        layout._inverse = (0, 1, 2)
    with pytest.raises(AttributeError):
        layout._owner_rows = ()
    with pytest.raises(AttributeError):
        layout._counts = (0, 0, 0)
    with pytest.raises(AttributeError):
        layout._offsets = (0, 0, 0, 0)
    with pytest.raises(AttributeError):
        layout._group_ranks = (7, 2)
    with pytest.raises(AttributeError):
        layout._global_to_local = {}
    with pytest.raises(AttributeError):
        del layout._offsets
    # Public state-corrupting assignments fail as well.
    with pytest.raises(AttributeError):
        layout.forward_permutation = (0, 1, 2)
    with pytest.raises(AttributeError):
        layout.owner_counts = (0, 0, 0)
    with pytest.raises(AttributeError):
        layout.group_ranks = (7, 2)
    # Failed mutations leave the layout unchanged.
    assert layout.step_seq == 0
    assert layout.forward_permutation == (2, 0, 1)
    assert layout.inverse_permutation == (1, 2, 0)
    assert layout.owner_counts == (1, 2, 0)
    assert layout.owner_offsets == (0, 1, 3, 3)
    assert layout.group_ranks == (7, 2, 11)
    assert layout.global_to_local == {7: 0, 2: 1, 11: 2}


# -- OwnerCollectivePlan ------------------------------------------------------


def _plan_group(layout: OwnerRowLayout) -> list[OwnerCollectivePlan]:
    """One plan per group member, built from its process-global rank."""
    return [
        OwnerCollectivePlan(layout, layout.group_ranks[local])
        for local in range(layout.world_size)
    ]


def _simulate_fanout(
    layout: OwnerRowLayout, plans: list[OwnerCollectivePlan]
) -> list[tuple[ExecutionRow, ...]]:
    """Deterministic whole-group all_to_all_single simulation for fanout.

    Every rank's send buffer is its local owner rows tiled once per
    destination; only the plan's split vectors and the layout are used to
    route rows.  Returns each rank's receive buffer in source-major order.
    """
    world_size = layout.world_size
    sends: list[tuple[ExecutionRow, ...]] = []
    for local, plan in enumerate(plans):
        send = (
            tuple(layout.owner_rows[layout.owner_slice_for_local(local)]) * world_size
        )
        assert (
            plan.owner_to_all_input_splits == (layout.owner_counts[local],) * world_size
        )
        assert sum(plan.owner_to_all_input_splits) == len(send)
        assert plan.fanout_send_rows == send
        sends.append(send)
    receives: list[tuple[ExecutionRow, ...]] = []
    for local, plan in enumerate(plans):
        parts: list[ExecutionRow] = []
        for src in range(world_size):
            input_splits = plans[src].owner_to_all_input_splits
            start = sum(input_splits[:local])
            part = sends[src][start : start + input_splits[local]]
            assert part == tuple(layout.owner_rows[layout.owner_slice_for_local(src)])
            assert len(part) == plan.owner_to_all_output_splits[src]
            parts.extend(part)
        receive = tuple(parts)
        assert sum(plan.owner_to_all_output_splits) == len(receive)
        receives.append(receive)
    return receives


def _simulate_fanin(
    layout: OwnerRowLayout, plans: list[OwnerCollectivePlan]
) -> list[tuple[ExecutionRow, ...]]:
    """Deterministic whole-group all_to_all_single simulation for fanin.

    Every rank's send buffer is the full owner-major rows array; only the
    plan's split vectors and the layout are used to route rows.  Returns
    each rank's receive buffer in source-major order.
    """
    world_size = layout.world_size
    sends: list[tuple[ExecutionRow, ...]] = []
    for plan in plans:
        send = tuple(layout.owner_rows)
        assert plan.all_to_owner_input_splits == layout.owner_counts
        assert sum(plan.all_to_owner_input_splits) == len(send)
        assert plan.fanin_send_rows == send
        sends.append(send)
    receives: list[tuple[ExecutionRow, ...]] = []
    for local, plan in enumerate(plans):
        parts: list[ExecutionRow] = []
        for src in range(world_size):
            input_splits = plans[src].all_to_owner_input_splits
            start = sum(input_splits[:local])
            part = sends[src][start : start + input_splits[local]]
            assert part == tuple(layout.owner_rows[layout.owner_slice_for_local(local)])
            assert len(part) == plan.all_to_owner_output_splits[src]
            parts.extend(part)
        receive = tuple(parts)
        assert sum(plan.all_to_owner_output_splits) == len(receive)
        receives.append(receive)
    return receives


def _ragged_zero_owner_layout() -> OwnerRowLayout:
    # Noncontiguous, unsorted group (7, 2, 11, 5); rank 7 owns zero rows,
    # ranks 2/11/5 own 3/2/1 rows respectively.
    rows = _rows(["b", "b", "b", "c", "c", "d"], [0, 1, 2, 0, 1, 0])
    owner = _owners(["b", "c", "d"], {"b": 2, "c": 11, "d": 5})
    layout = OwnerRowLayout.build(3, rows, owner, (7, 2, 11, 5))
    assert layout.owner_counts == (0, 3, 2, 1)
    assert layout.owner_offsets == (0, 0, 3, 5, 6)
    return layout


def test_plan_binding_exact_splits_and_rank_mapping() -> None:
    layout = _ragged_zero_owner_layout()
    plans = _plan_group(layout)
    for local, plan in enumerate(plans):
        global_rank = layout.group_ranks[local]
        assert plan.layout is layout
        assert plan.owner_global_rank == global_rank
        assert plan.local_rank == local
        assert plan.local_rank == layout.local_rank_of_global(global_rank)
        assert plan.world_size == layout.world_size == 4
        count = layout.owner_counts[local]
        assert plan.local_owner_count == count
        assert plan.local_owner_slice == layout.owner_slice_for_local(local)
        assert plan.local_owner_rows == tuple(
            layout.owner_rows[layout.owner_slice_for_local(local)]
        )
        # Fanout: input splits tile the local owner count, output splits
        # are the owner counts; fanin is the exact mirror.
        assert plan.owner_to_all_input_splits == (count,) * 4
        assert plan.owner_to_all_output_splits == layout.owner_counts
        assert plan.all_to_owner_input_splits == layout.owner_counts
        assert plan.all_to_owner_output_splits == (count,) * 4
        # Expected identities.
        assert plan.fanout_receive_rows == tuple(layout.owner_rows)
        assert plan.fanin_send_rows == tuple(layout.owner_rows)
        assert plan.fanin_receive_rows == plan.local_owner_rows * 4
        assert plan.fanout_send_rows == plan.local_owner_rows * 4
        # Split sums match the simulated buffers.
        assert sum(plan.owner_to_all_input_splits) == len(plan.fanout_send_rows)
        assert sum(plan.owner_to_all_output_splits) == len(plan.fanout_receive_rows)
        assert sum(plan.all_to_owner_input_splits) == len(plan.fanin_send_rows)
        assert sum(plan.all_to_owner_output_splits) == len(plan.fanin_receive_rows)


def test_fanout_simulation_every_rank_receives_owner_rows() -> None:
    layout = _ragged_zero_owner_layout()
    plans = _plan_group(layout)
    fanout_receives = _simulate_fanout(layout, plans)
    for local, plan in enumerate(plans):
        assert fanout_receives[local] == tuple(layout.owner_rows)
        assert fanout_receives[local] == plan.fanout_receive_rows
    # The zero owner participates: it sends empty tiles and receives the
    # same full owner-major array as everyone else.
    assert plans[0].local_owner_count == 0
    assert plans[0].local_owner_rows == ()
    assert fanout_receives[0] == tuple(layout.owner_rows)


def test_fanin_simulation_every_owner_receives_source_major_rows() -> None:
    layout = _ragged_zero_owner_layout()
    plans = _plan_group(layout)
    fanin_receives = _simulate_fanin(layout, plans)
    for local, plan in enumerate(plans):
        expected = (
            tuple(layout.owner_rows[layout.owner_slice_for_local(local)])
            * layout.world_size
        )
        assert fanin_receives[local] == expected
        assert fanin_receives[local] == plan.fanin_receive_rows
    # The zero owner receives nothing.
    assert plans[0].fanin_receive_rows == ()


def test_q_fragment_transpose_and_o_partial_exact_sums() -> None:
    layout = _ragged_zero_owner_layout()
    world_size = layout.world_size
    # Q fanin arrives source-major: (source, row).  Transpose it to
    # row-major source shards, i.e. every source's copy of one row.
    plan = OwnerCollectivePlan(layout, 11)  # local rank 2, count 2
    count = plan.local_owner_count
    assert count == 2
    fragment = plan.fanin_receive_rows
    assert fragment == (plan.local_owner_rows * world_size)
    for index in range(count):
        row = plan.local_owner_rows[index]
        shard = fragment[index::count]
        assert shard == (row,) * world_size
    # O partials: every source contributes (src + 1) * position per row;
    # summing the source-major buffer per row must be exact.
    partials = tuple(
        (src + 1) * row.row_id.logical_token_position
        for src in range(world_size)
        for row in plan.local_owner_rows
    )
    assert len(partials) == count * world_size
    for index in range(count):
        position = plan.local_owner_rows[index].row_id.logical_token_position
        assert sum(partials[index::count]) == (
            sum(src + 1 for src in range(world_size)) * position
        )
    # The zero owner's fragment and partials are both empty.
    zero_plan = OwnerCollectivePlan(layout, 7)
    assert zero_plan.fanin_receive_rows == ()
    assert (
        tuple(
            (src + 1) * row.row_id.logical_token_position
            for src in range(world_size)
            for row in zero_plan.local_owner_rows
        )
        == ()
    )


def test_all_zero_rows_collective_plan_on_every_rank() -> None:
    group = (7, 2, 11)
    layout = OwnerRowLayout.build(0, [], {}, group)
    zero = (0, 0, 0)
    for global_rank in group:
        plan = OwnerCollectivePlan(layout, global_rank)
        assert plan.local_owner_count == 0
        assert plan.local_owner_rows == ()
        assert plan.owner_to_all_input_splits == zero
        assert plan.owner_to_all_output_splits == zero
        assert plan.all_to_owner_input_splits == zero
        assert plan.all_to_owner_output_splits == zero
        assert plan.fanout_send_rows == ()
        assert plan.fanout_receive_rows == ()
        assert plan.fanin_send_rows == ()
        assert plan.fanin_receive_rows == ()


def test_plan_rejects_bool_nonint_and_unknown_ranks() -> None:
    layout = OwnerRowLayout.build(0, [], {}, (7, 2, 11))
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan(layout, True)
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan(layout, False)
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan(layout, "11")  # type: ignore[arg-type]
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan(layout, 2.0)  # type: ignore[arg-type]
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan(layout, -1)
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan(layout, 999)
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan(layout, 5)  # not a group member
    with pytest.raises(OwnerLayoutError):
        OwnerCollectivePlan("layout", 7)  # type: ignore[arg-type]
    # A valid group member is accepted.
    plan = OwnerCollectivePlan(layout, 2)
    assert plan.local_rank == 1
    assert plan.owner_global_rank == 2


def test_owner_collective_plan_is_immutable_after_construction() -> None:
    layout = _ragged_zero_owner_layout()
    plan = OwnerCollectivePlan(layout, 11)
    with pytest.raises(AttributeError):
        plan._local_owner_count = 7
    with pytest.raises(AttributeError):
        plan._layout = layout
    with pytest.raises(AttributeError):
        plan._owner_to_all_input_splits = (1, 1, 1, 1)
    with pytest.raises(AttributeError):
        plan._fanin_receive_rows = ()
    with pytest.raises(AttributeError):
        del plan._fanout_receive_rows
    # Public state-corrupting assignments and deletions fail as well.
    with pytest.raises(AttributeError):
        plan.local_owner_count = 0
    with pytest.raises(AttributeError):
        plan.owner_to_all_input_splits = (1, 1, 1, 1)
    with pytest.raises(AttributeError):
        del plan.fanin_receive_rows
    # Failed mutations leave the plan unchanged.
    assert plan.local_owner_count == 2
    assert plan.local_rank == 2
    assert plan.owner_to_all_input_splits == (2, 2, 2, 2)
    assert plan.fanin_receive_rows == plan.local_owner_rows * 4
