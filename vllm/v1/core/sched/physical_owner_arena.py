# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure host plan for fixed physical owner seats over real logical rows.

The scheduler-owned :class:`~vllm.v1.core.sched.owner_layout.OwnerRowLayout`
contains only real rows and identities.  A reusable full graph instead needs a
fixed number of token seats per owner.  This module bridges those contracts
without manufacturing a ``GlobalRowId`` or lease for padding.

R4-v1 deliberately uses owner-prefix packing: each owner's real single-token
decode rows occupy the leading seats of its physical block.  Invalid seats map
to unique elements of the canonical padded input tail, so the physical
forward/inverse tensors remain exact fixed-size permutations while the
logical/physical maps use ``-1`` only as a host-side invalid sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.sched.owner_layout import (
    OwnerLayoutError,
    OwnerRowLayout,
    RequestOwnedGraphSignature,
)


class PhysicalOwnerArenaError(OwnerLayoutError):
    """A physical owner plan violates its fixed-seat contract."""


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhysicalOwnerArenaError(
            f"{name} must be a nonnegative non-bool int, got {value!r}."
        )
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise PhysicalOwnerArenaError(f"{name} must be positive, got 0.")
    return result


def _exact_permutation(values: tuple[int, ...], size: int, name: str) -> None:
    if (
        len(values) != size
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        )
        or tuple(sorted(values)) != tuple(range(size))
    ):
        raise PhysicalOwnerArenaError(
            f"{name} must be an exact permutation of range({size}), got {values!r}."
        )


@dataclass(frozen=True, slots=True)
class PhysicalOwnerArenaPlan:
    """Immutable fixed-seat execution plan for one real logical layout.

    ``physical_to_logical`` stores canonical logical indices for valid seats
    and ``-1`` for invalid seats.  ``forward_indices`` instead maps every
    physical seat to a unique canonical *input* position, including the
    padded tail, and is therefore an exact permutation of ``range(capacity)``.
    """

    step_seq: int
    group_ranks: tuple[int, ...]
    rows_per_owner: int
    logical_owner_counts: tuple[int, ...]
    valid_mask: tuple[bool, ...]
    physical_to_logical: tuple[int, ...]
    logical_to_physical: tuple[int, ...]
    forward_indices: tuple[int, ...]
    inverse_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        step_seq = _nonnegative_int(self.step_seq, "step_seq")
        rows_per_owner = _positive_int(self.rows_per_owner, "rows_per_owner")
        group_ranks = tuple(self.group_ranks)
        counts = tuple(self.logical_owner_counts)
        valid_mask = tuple(self.valid_mask)
        physical_to_logical = tuple(self.physical_to_logical)
        logical_to_physical = tuple(self.logical_to_physical)
        forward = tuple(self.forward_indices)
        inverse = tuple(self.inverse_indices)

        if not group_ranks:
            raise PhysicalOwnerArenaError("group_ranks must not be empty.")
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in group_ranks
        ) or len(set(group_ranks)) != len(group_ranks):
            raise PhysicalOwnerArenaError(
                f"group_ranks must be distinct nonnegative ints, got {group_ranks!r}."
            )
        if len(counts) != len(group_ranks):
            raise PhysicalOwnerArenaError(
                "logical_owner_counts must cover every group rank: "
                f"{len(counts)} != {len(group_ranks)}."
            )
        for owner_index, count in enumerate(counts):
            count = _nonnegative_int(count, f"logical_owner_counts[{owner_index}]")
            if count > rows_per_owner:
                raise PhysicalOwnerArenaError(
                    "logical owner count exceeds physical capacity: "
                    f"owner={owner_index} count={count} capacity={rows_per_owner}."
                )

        capacity = len(group_ranks) * rows_per_owner
        logical_len = sum(counts)
        if len(valid_mask) != capacity or any(
            not isinstance(value, bool) for value in valid_mask
        ):
            raise PhysicalOwnerArenaError(
                f"valid_mask must contain {capacity} strict bools."
            )
        if len(physical_to_logical) != capacity:
            raise PhysicalOwnerArenaError(
                f"physical_to_logical must contain {capacity} entries."
            )
        if len(logical_to_physical) != logical_len:
            raise PhysicalOwnerArenaError(
                "logical_to_physical length must equal the logical row count: "
                f"{len(logical_to_physical)} != {logical_len}."
            )
        _exact_permutation(forward, capacity, "forward_indices")
        _exact_permutation(inverse, capacity, "inverse_indices")
        if any(
            inverse[canonical] != physical for physical, canonical in enumerate(forward)
        ):
            raise PhysicalOwnerArenaError(
                "forward_indices and inverse_indices must be exact inverses."
            )

        expected_logical = tuple(range(logical_len))
        seen_logical: list[int] = []
        seen_padding: list[int] = []
        for physical, (valid, logical, canonical_input) in enumerate(
            zip(valid_mask, physical_to_logical, forward, strict=True)
        ):
            if valid:
                if (
                    isinstance(logical, bool)
                    or not isinstance(logical, int)
                    or not 0 <= logical < logical_len
                ):
                    raise PhysicalOwnerArenaError(
                        "valid physical seat "
                        f"{physical} has invalid logical index {logical!r}."
                    )
                if canonical_input != logical:
                    raise PhysicalOwnerArenaError(
                        "a valid seat's forward input must be its canonical "
                        "logical row: "
                        f"seat={physical} logical={logical} input={canonical_input}."
                    )
                seen_logical.append(logical)
            else:
                if logical != -1:
                    raise PhysicalOwnerArenaError(
                        "invalid physical seat "
                        f"{physical} must map to -1, got {logical!r}."
                    )
                if canonical_input < logical_len:
                    raise PhysicalOwnerArenaError(
                        "an invalid seat must map to the canonical padded tail: "
                        f"seat={physical} input={canonical_input} "
                        f"logical_len={logical_len}."
                    )
                seen_padding.append(canonical_input)

        if tuple(sorted(seen_logical)) != expected_logical:
            raise PhysicalOwnerArenaError(
                "valid physical seats must cover every logical row exactly once."
            )
        if tuple(sorted(seen_padding)) != tuple(range(logical_len, capacity)):
            raise PhysicalOwnerArenaError(
                "invalid physical seats must cover every padded input exactly once."
            )
        for logical, physical in enumerate(logical_to_physical):
            if (
                isinstance(physical, bool)
                or not isinstance(physical, int)
                or not 0 <= physical < capacity
                or physical_to_logical[physical] != logical
            ):
                raise PhysicalOwnerArenaError(
                    "logical_to_physical must invert physical_to_logical: "
                    f"logical={logical} physical={physical!r}."
                )

        for owner_index, count in enumerate(counts):
            begin = owner_index * rows_per_owner
            expected = (True,) * count + (False,) * (rows_per_owner - count)
            actual = valid_mask[begin : begin + rows_per_owner]
            if actual != expected:
                raise PhysicalOwnerArenaError(
                    "R4-v1 requires owner-prefix packing: "
                    f"owner={owner_index} mask={actual!r} expected={expected!r}."
                )

        object.__setattr__(self, "step_seq", step_seq)
        object.__setattr__(self, "group_ranks", group_ranks)
        object.__setattr__(self, "rows_per_owner", rows_per_owner)
        object.__setattr__(self, "logical_owner_counts", counts)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "physical_to_logical", physical_to_logical)
        object.__setattr__(self, "logical_to_physical", logical_to_physical)
        object.__setattr__(self, "forward_indices", forward)
        object.__setattr__(self, "inverse_indices", inverse)

    @property
    def world_size(self) -> int:
        return len(self.group_ranks)

    @property
    def capacity(self) -> int:
        return self.world_size * self.rows_per_owner

    @property
    def logical_len(self) -> int:
        return sum(self.logical_owner_counts)

    @property
    def physical_owner_counts(self) -> tuple[int, ...]:
        return (self.rows_per_owner,) * self.world_size

    @property
    def graph_signature(self) -> RequestOwnedGraphSignature:
        return RequestOwnedGraphSignature(
            owner_counts=self.physical_owner_counts,
            canonical_to_owner=tuple(range(self.capacity)),
        )

    def owner_physical_slice(self, local_rank: int) -> slice:
        rank = _nonnegative_int(local_rank, "local_rank")
        if rank >= self.world_size:
            raise PhysicalOwnerArenaError(
                f"local_rank {rank} is outside world_size {self.world_size}."
            )
        begin = rank * self.rows_per_owner
        return slice(begin, begin + self.rows_per_owner)

    def owner_active_slice(self, local_rank: int) -> slice:
        physical = self.owner_physical_slice(local_rank)
        return slice(
            physical.start, physical.start + self.logical_owner_counts[local_rank]
        )


def build_physical_owner_arena_plan(
    layout: OwnerRowLayout,
    *,
    rows_per_owner: int,
    num_reqs: int,
    num_tokens: int,
    uniform_decode: bool,
) -> PhysicalOwnerArenaPlan | None:
    """Return the R4-v1 fixed-seat ordinary-decode plan, or ``None``.

    ``None`` means that FULL is ineligible and the caller must fail closed to
    an existing non-FULL mode.  Malformed arguments raise instead of being
    reinterpreted as an ordinary ineligible batch.
    """

    if not isinstance(layout, OwnerRowLayout):
        raise PhysicalOwnerArenaError(
            f"layout must be an OwnerRowLayout, got {layout!r}."
        )
    physical_rows = _positive_int(rows_per_owner, "rows_per_owner")
    reqs = _nonnegative_int(num_reqs, "num_reqs")
    tokens = _nonnegative_int(num_tokens, "num_tokens")
    if not isinstance(uniform_decode, bool):
        raise PhysicalOwnerArenaError(
            f"uniform_decode must be a bool, got {uniform_decode!r}."
        )

    # R4-v1 is exactly one ordinary decode token per real request. Global
    # zero-work is not replayed; partial speculative/prefill/ragged batches
    # remain on the existing fail-closed path.
    if (
        not uniform_decode
        or reqs == 0
        or tokens != reqs
        or layout.logical_len != tokens
        or any(count > physical_rows for count in layout.owner_counts)
    ):
        return None

    logical_len = layout.logical_len
    capacity = layout.world_size * physical_rows
    forward: list[int] = []
    physical_to_logical: list[int] = []
    valid_mask: list[bool] = []
    next_padding = logical_len

    for owner_index, count in enumerate(layout.owner_counts):
        owner_begin = layout.owner_offsets[owner_index]
        owner_rows = layout.forward_permutation[owner_begin : owner_begin + count]
        forward.extend(owner_rows)
        physical_to_logical.extend(owner_rows)
        valid_mask.extend((True,) * count)
        invalid_count = physical_rows - count
        padding = tuple(range(next_padding, next_padding + invalid_count))
        next_padding += invalid_count
        forward.extend(padding)
        physical_to_logical.extend((-1,) * invalid_count)
        valid_mask.extend((False,) * invalid_count)

    if next_padding != capacity:
        raise PhysicalOwnerArenaError(
            "internal padded-tail construction ended at "
            f"{next_padding}, expected {capacity}."
        )
    inverse = [0] * capacity
    for physical, canonical_input in enumerate(forward):
        inverse[canonical_input] = physical
    logical_to_physical = tuple(inverse[:logical_len])

    return PhysicalOwnerArenaPlan(
        step_seq=layout.step_seq,
        group_ranks=layout.group_ranks,
        rows_per_owner=physical_rows,
        logical_owner_counts=layout.owner_counts,
        valid_mask=tuple(valid_mask),
        physical_to_logical=tuple(physical_to_logical),
        logical_to_physical=logical_to_physical,
        forward_indices=tuple(forward),
        inverse_indices=tuple(inverse),
    )
