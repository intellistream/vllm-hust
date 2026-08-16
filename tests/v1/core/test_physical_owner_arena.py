# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU semantic seal for fixed physical owner seats."""

from dataclasses import replace

import pytest

from vllm.v1.core.sched.owner_layout import (
    GlobalRowId,
    OwnerRowLayout,
    RequestOwnedGraphSignature,
    balanced_decode_graph_signature,
)
from vllm.v1.core.sched.ownership import OwnerLeaseKey
from vllm.v1.core.sched.physical_owner_arena import (
    PhysicalOwnerArenaError,
    PhysicalOwnerArenaPlan,
    build_physical_owner_arena_plan,
    select_request_owned_decode_graph_envelope,
)

_GROUP_RANKS = (7, 2, 11, 5, 13, 3, 17, 19)


def _layout(counts: tuple[int, ...], *, step_seq: int = 23) -> OwnerRowLayout:
    assert len(counts) == len(_GROUP_RANKS)
    rows: list[GlobalRowId] = []
    owners: dict[OwnerLeaseKey, int] = {}
    # Deliberately interleave owners in reverse group order so nontrivial
    # cohorts exercise a real canonical/owner-major permutation.
    for seat in range(max(counts, default=0)):
        for owner_index in reversed(range(len(counts))):
            if seat >= counts[owner_index]:
                continue
            key = OwnerLeaseKey(
                request_id=f"owner-{owner_index}-seat-{seat}",
                owner_epoch=step_seq,
            )
            rows.append(GlobalRowId(key, logical_token_position=step_seq + seat))
            owners[key] = _GROUP_RANKS[owner_index]
    return OwnerRowLayout.build(step_seq, rows, owners, _GROUP_RANKS)


_COUNT_PATTERNS = (
    (4, 4, 4, 4, 4, 4, 4, 4),  # 32
    (4, 4, 4, 4, 4, 4, 4, 3),  # 31
    (4, 4, 4, 4, 3, 3, 3, 3),  # 28
    (4, 0, 4, 0, 4, 0, 4, 0),  # 16 with local-zero owners
    (0, 0, 0, 0, 0, 0, 0, 4),  # 4, maximally skewed
)


@pytest.mark.parametrize("counts", _COUNT_PATTERNS)
def test_plan_is_a_bijective_fixed_physical_envelope(
    counts: tuple[int, ...],
) -> None:
    layout = _layout(counts)
    plan = build_physical_owner_arena_plan(
        layout,
        rows_per_owner=4,
        num_reqs=sum(counts),
        num_tokens=sum(counts),
        uniform_decode=True,
    )
    assert isinstance(plan, PhysicalOwnerArenaPlan)
    assert plan.step_seq == layout.step_seq
    assert plan.group_ranks == layout.group_ranks
    assert plan.logical_owner_counts == counts
    assert plan.physical_owner_counts == (4,) * 8
    assert plan.capacity == 32
    assert plan.logical_len == sum(counts)
    assert plan.graph_signature == RequestOwnedGraphSignature(
        owner_counts=(4,) * 8,
        canonical_to_owner=tuple(range(32)),
    )

    assert sorted(plan.forward_indices) == list(range(32))
    assert sorted(plan.inverse_indices) == list(range(32))
    assert all(
        plan.inverse_indices[canonical] == physical
        for physical, canonical in enumerate(plan.forward_indices)
    )
    assert sorted(value for value in plan.physical_to_logical if value >= 0) == list(
        range(sum(counts))
    )
    assert sorted(
        plan.forward_indices[physical]
        for physical, valid in enumerate(plan.valid_mask)
        if not valid
    ) == list(range(sum(counts), 32))

    for owner_index, count in enumerate(counts):
        physical = plan.owner_physical_slice(owner_index)
        active = plan.owner_active_slice(owner_index)
        assert physical == slice(owner_index * 4, owner_index * 4 + 4)
        assert active == slice(owner_index * 4, owner_index * 4 + count)
        assert plan.valid_mask[physical] == (True,) * count + (False,) * (4 - count)
        owner_begin = layout.owner_offsets[owner_index]
        expected_logical = layout.forward_permutation[owner_begin : owner_begin + count]
        assert plan.physical_to_logical[active] == expected_logical

    canonical = tuple(f"input-{index}" for index in range(32))
    physical = tuple(canonical[index] for index in plan.forward_indices)
    restored = tuple(physical[index] for index in plan.inverse_indices)
    assert restored == canonical
    assert (
        tuple(
            physical[plan.logical_to_physical[logical]]
            for logical in range(plan.logical_len)
        )
        == canonical[: plan.logical_len]
    )


def test_full_plan_matches_the_existing_balanced_graph_key() -> None:
    layout = _layout((4,) * 8)
    plan = build_physical_owner_arena_plan(
        layout,
        rows_per_owner=4,
        num_reqs=32,
        num_tokens=32,
        uniform_decode=True,
    )
    assert plan is not None
    assert plan.graph_signature == balanced_decode_graph_signature(
        layout,
        num_reqs=32,
        num_tokens=32,
        uniform_decode=True,
    )


@pytest.mark.parametrize(
    ("counts", "expects_physical"),
    (
        ((4,) * 8, False),
        ((2,) * 8, True),  # balanced logical partial occupancy
        ((4, 0) * 4, True),
    ),
)
def test_graph_envelope_never_shrinks_an_ordinary_partial_batch(
    counts: tuple[int, ...], expects_physical: bool
) -> None:
    signature, plan = select_request_owned_decode_graph_envelope(
        _layout(counts),
        rows_per_owner=4,
        num_reqs=sum(counts),
        num_tokens=sum(counts),
        uniform_decode=True,
    )
    assert signature == RequestOwnedGraphSignature(
        owner_counts=(4,) * 8,
        canonical_to_owner=tuple(range(32)),
    )
    assert isinstance(plan, PhysicalOwnerArenaPlan) is expects_physical


def test_graph_envelope_preserves_the_balanced_multi_token_lane() -> None:
    layout = _layout((4,) * 8)
    signature, plan = select_request_owned_decode_graph_envelope(
        layout,
        rows_per_owner=4,
        num_reqs=16,
        num_tokens=32,
        uniform_decode=True,
    )
    assert signature == balanced_decode_graph_signature(
        layout,
        num_reqs=16,
        num_tokens=32,
        uniform_decode=True,
    )
    assert plan is None


@pytest.mark.parametrize(
    ("counts", "num_reqs", "num_tokens", "uniform_decode"),
    (
        ((4,) * 8, 32, 32, False),
        ((4,) * 8, 31, 32, True),
        ((4,) * 8, 16, 32, True),  # partial speculative K+1
        ((4,) * 8, 32, 31, True),
        ((5, 0, 0, 0, 0, 0, 0, 0), 5, 5, True),
        ((0,) * 8, 0, 0, True),  # global zero-work never replays FULL
    ),
)
def test_ineligible_batches_fail_closed_to_no_plan(
    counts: tuple[int, ...],
    num_reqs: int,
    num_tokens: int,
    uniform_decode: bool,
) -> None:
    assert (
        build_physical_owner_arena_plan(
            _layout(counts),
            rows_per_owner=4,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            uniform_decode=uniform_decode,
        )
        is None
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"rows_per_owner": 0}, "rows_per_owner must be positive"),
        ({"rows_per_owner": True}, "rows_per_owner must be a nonnegative"),
        ({"num_reqs": -1}, "num_reqs must be a nonnegative"),
        ({"num_tokens": True}, "num_tokens must be a nonnegative"),
        ({"uniform_decode": 1}, "uniform_decode must be a bool"),
    ),
)
def test_malformed_builder_arguments_raise(
    kwargs: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {
        "rows_per_owner": 4,
        "num_reqs": 4,
        "num_tokens": 4,
        "uniform_decode": True,
    }
    arguments.update(kwargs)
    with pytest.raises(PhysicalOwnerArenaError, match=match):
        build_physical_owner_arena_plan(
            _layout((4,) + (0,) * 7),
            **arguments,  # type: ignore[arg-type]
        )


def test_plan_rejects_non_prefix_or_non_bijective_manual_state() -> None:
    plan = build_physical_owner_arena_plan(
        _layout((2, 0, 1, 0, 0, 0, 0, 0)),
        rows_per_owner=4,
        num_reqs=3,
        num_tokens=3,
        uniform_decode=True,
    )
    assert plan is not None

    mask = list(plan.valid_mask)
    mask[0], mask[2] = False, True
    with pytest.raises(PhysicalOwnerArenaError):
        replace(plan, valid_mask=tuple(mask))

    duplicate = list(plan.forward_indices)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(PhysicalOwnerArenaError, match="exact permutation"):
        replace(plan, forward_indices=tuple(duplicate))

    with pytest.raises(PhysicalOwnerArenaError, match="strict bools"):
        replace(plan, valid_mask=tuple(1 if value else 0 for value in plan.valid_mask))


def test_plan_is_frozen_and_does_not_modify_the_logical_layout() -> None:
    layout = _layout((1, 0, 1, 0, 1, 0, 1, 0))
    before = (
        layout.global_rows,
        layout.owner_rows,
        layout.owner_counts,
        layout.forward_permutation,
        layout.inverse_permutation,
    )
    plan = build_physical_owner_arena_plan(
        layout,
        rows_per_owner=4,
        num_reqs=4,
        num_tokens=4,
        uniform_decode=True,
    )
    assert plan is not None
    with pytest.raises(AttributeError):
        plan.rows_per_owner = 8  # type: ignore[misc]
    assert before == (
        layout.global_rows,
        layout.owner_rows,
        layout.owner_counts,
        layout.forward_permutation,
        layout.inverse_permutation,
    )
    assert all(row.row_id.request_uid.request_id for row in layout.global_rows)
    assert not any(isinstance(value, GlobalRowId) for value in plan.physical_to_logical)
