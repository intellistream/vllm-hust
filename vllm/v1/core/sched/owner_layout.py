# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-neutral G1 GlobalRowId owner-layout host/reference codec.

This module defines the persistent identity of one logical row in the
request-owned global KV grid (:class:`GlobalRowId`, fenced by
:class:`OwnerLeaseKey`) and the immutable owner-major layout
(:class:`OwnerRowLayout`) that maps canonical execution rows onto
per-owner row buffers for the request-owned attention collectives.

The companion :class:`OwnerCollectivePlan` binds such a layout to one
process-global rank of the owning group and derives the exact split
vectors and row identities of the two unconditional raw
``all_to_all_single`` exchanges of the request-owned collectives:
owner_to_all / fanout (every rank sends its local owner rows to every
rank and receives the full owner-major rows array) and all_to_owner /
fanin (every rank sends the full owner-major rows array and each owner
receives its stable rows once per source, source-rank-major).  The plan
is dependency-neutral: no torch or distributed import, no Q/KV-specific
API; rank lookup always goes through
:meth:`OwnerRowLayout.local_rank_of_global`.

The layout is group-local: ``group_ranks`` lists the process-global ranks
of the owning group (local index -> global rank) and may be noncontiguous
and unsorted.  ``owner_counts``, ``owner_offsets``, and ``owner_rows`` are
in group-local rank order; ``owner_offsets`` has length ``world_size + 1``
so owners with zero rows simply repeat their offset.  The canonical
execution order is the input ``row_ids`` order, and rows inside each owner
bucket keep that canonical order (stable buckets).

Exact invariants (verified at build time and directly by the tests):

* ``owner_rows[j] == global_rows[forward[j]]``
* ``global_rows[i] == owner_rows[inverse[i]]``
* ``forward[inverse[i]] == i``
* ``inverse[forward[j]] == j``

Everything here is pure Python: the standard library plus the ownership
types from ``vllm.v1.core.sched.ownership``.  Tensor helpers only use
duck-typed first-axis indexing (no torch import), so dtype and trailing
shapes are preserved.  This module never fabricates a :class:`GlobalRowId`
for physical padding: the helpers operate on the exact logical prefix and
fail closed on any mismatch.
"""

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from vllm.v1.core.sched.ownership import OwnerLeaseKey, OwnerLeaseToken


class OwnerLayoutError(Exception):
    """Raised when an owner-row layout or payload violates a contract."""


_MISSING = object()


@dataclass(frozen=True)
class RequestOwnedGraphSignature:
    """Step-invariant owner layout identity for a full decode graph.

    The execution fence and logical row identities deliberately do not belong
    in this key: both change every step while the captured data-plane envelope
    remains the same. Counts and the canonical-to-owner permutation do belong
    in the key because either can change collective geometry or row meaning.
    """

    owner_counts: tuple[int, ...]
    canonical_to_owner: tuple[int, ...]

    def __post_init__(self) -> None:
        counts = tuple(self.owner_counts)
        permutation = tuple(self.canonical_to_owner)
        object.__setattr__(self, "owner_counts", counts)
        object.__setattr__(self, "canonical_to_owner", permutation)
        if not counts:
            raise OwnerLayoutError("graph signature owner_counts must not be empty")
        for count in counts:
            _require_nonneg_int(count, "graph signature owner count")
        if sum(counts) != len(permutation):
            raise OwnerLayoutError(
                "graph signature owner counts do not cover its permutation"
            )
        expected = tuple(range(len(permutation)))
        if tuple(sorted(permutation)) != expected:
            raise OwnerLayoutError(
                "graph signature canonical_to_owner must be a permutation"
            )


def _require_nonneg_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OwnerLayoutError(f"{name} must be a nonnegative integer, got {value!r}")
    return value


def _validate_group_ranks(group_ranks: Sequence[int]) -> tuple[int, ...]:
    ranks = tuple(group_ranks)
    if not ranks:
        raise OwnerLayoutError("group_ranks must not be empty")
    seen: set[int] = set()
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise OwnerLayoutError(f"invalid global rank {rank!r}")
        if rank in seen:
            raise OwnerLayoutError(f"duplicate global rank {rank!r}")
        seen.add(rank)
    return ranks


def _verify_permutations(forward: Sequence[int], inverse: Sequence[int]) -> None:
    """Fail closed if the permutation pair is not exactly inverse."""
    if len(forward) != len(inverse):
        raise OwnerLayoutError("forward and inverse permutations differ in length")
    for index, canonical_index in enumerate(inverse):
        if forward[canonical_index] != index:
            raise OwnerLayoutError("inconsistent forward/inverse permutation")
    for index, owner_index in enumerate(forward):
        if inverse[owner_index] != index:
            raise OwnerLayoutError("inconsistent forward/inverse permutation")


def _first_axis_length(payload: object) -> int:
    shape = getattr(payload, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except (IndexError, TypeError):
            pass
    try:
        return len(payload)  # type: ignore[arg-type]
    except TypeError as exc:
        raise OwnerLayoutError(f"payload has no first axis: {payload!r}") from exc


@dataclass(frozen=True)
class GlobalRowId:
    """Generation-fenced identity of one logical row in the global grid.

    ``request_uid`` is the :class:`OwnerLeaseKey` (request id fenced by the
    reuse epoch), ``logical_token_position`` the per-request token offset,
    and ``logical_lane`` the lane of that request/token.  Positions and
    lanes are nonnegative; physical padding never receives a
    :class:`GlobalRowId`.
    """

    request_uid: OwnerLeaseKey
    logical_token_position: int
    logical_lane: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request_uid, OwnerLeaseKey):
            raise OwnerLayoutError(
                f"request_uid must be an OwnerLeaseKey, got {self.request_uid!r}"
            )
        _require_nonneg_int(self.logical_token_position, "logical_token_position")
        _require_nonneg_int(self.logical_lane, "logical_lane")


@dataclass(frozen=True)
class ExecutionRow:
    """A logical row bound to one execution step.

    ``step_seq`` is the execution fence: it rides on the row but is not
    part of the persistent row identity (:class:`GlobalRowId`), so the
    same logical row in a later execution carries a new fence.
    """

    step_seq: int
    row_id: GlobalRowId

    def __post_init__(self) -> None:
        _require_nonneg_int(self.step_seq, "step_seq")
        if not isinstance(self.row_id, GlobalRowId):
            raise OwnerLayoutError(f"row_id must be a GlobalRowId, got {self.row_id!r}")


class OwnerRowLayout:
    """Immutable owner-major permutation of canonical execution rows.

    Args:
        step_seq: execution fence shared by all rows of this layout.
        row_ids: logical rows in canonical execution order.
        owner_by_request_uid: maps each row's request uid to its
            process-global owner rank.
        group_ranks: process-global ranks of the owning group in
            group-local index order (may be noncontiguous and unsorted).

    Raises:
        OwnerLayoutError: on any contract violation; see the module
            docstring for the exact invariants.
    """

    __slots__ = (
        "_step_seq",
        "_global_rows",
        "_owner_rows",
        "_forward",
        "_inverse",
        "_counts",
        "_offsets",
        "_group_ranks",
        "_global_to_local",
        "_frozen",
    )

    def __init__(
        self,
        step_seq: int,
        row_ids: Sequence[GlobalRowId],
        owner_by_request_uid: Mapping[OwnerLeaseKey, int],
        group_ranks: Sequence[int],
    ) -> None:
        step = _require_nonneg_int(step_seq, "step_seq")
        if not isinstance(owner_by_request_uid, Mapping):
            raise OwnerLayoutError("owner_by_request_uid must be a Mapping")
        group = _validate_group_ranks(group_ranks)
        global_to_local = {rank: local for local, rank in enumerate(group)}
        rows = []
        for row_id in row_ids:
            if not isinstance(row_id, GlobalRowId):
                raise OwnerLayoutError(
                    f"row_ids must contain GlobalRowId, got {row_id!r}"
                )
            rows.append(ExecutionRow(step, row_id))
        seen: set[GlobalRowId] = set()
        for row in rows:
            if row.row_id in seen:
                raise OwnerLayoutError(f"duplicate GlobalRowId {row.row_id!r}")
            seen.add(row.row_id)
        owners = []
        for row in rows:
            owner = owner_by_request_uid.get(row.row_id.request_uid, _MISSING)
            if owner is _MISSING:
                raise OwnerLayoutError(
                    f"no owner mapping for {row.row_id.request_uid!r}"
                )
            if (
                isinstance(owner, bool)
                or not isinstance(owner, int)
                or owner not in global_to_local
            ):
                raise OwnerLayoutError(
                    f"unknown owner {owner!r} for {row.row_id.request_uid!r}"
                )
            owners.append(owner)
        buckets: list[list[int]] = [[] for _ in group]
        for canonical_index, owner in enumerate(owners):
            buckets[global_to_local[owner]].append(canonical_index)
        forward = [index for bucket in buckets for index in bucket]
        inverse = [0] * len(forward)
        for owner_index, canonical_index in enumerate(forward):
            inverse[canonical_index] = owner_index
        _verify_permutations(forward, inverse)
        counts = tuple(len(bucket) for bucket in buckets)
        offsets = [0] * (len(group) + 1)
        for local in range(len(group)):
            offsets[local + 1] = offsets[local] + counts[local]
        self._step_seq = step
        self._group_ranks = group
        self._global_to_local = global_to_local
        self._global_rows = tuple(rows)
        self._owner_rows = tuple(rows[index] for index in forward)
        self._forward = tuple(forward)
        self._inverse = tuple(inverse)
        self._counts = counts
        self._offsets = tuple(offsets)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(f"OwnerRowLayout is immutable: cannot set {name!r}")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"OwnerRowLayout is immutable: cannot delete {name!r}")

    @classmethod
    def build(
        cls,
        step_seq: int,
        row_ids: Sequence[GlobalRowId],
        owner_by_request_uid: Mapping[OwnerLeaseKey, int],
        group_ranks: Sequence[int],
    ) -> "OwnerRowLayout":
        """Construct and validate an immutable owner row layout."""
        return cls(step_seq, row_ids, owner_by_request_uid, group_ranks)

    # -- exposed layout state ------------------------------------------------

    @property
    def step_seq(self) -> int:
        """Execution fence shared by all rows of this layout."""
        return self._step_seq

    @property
    def group_ranks(self) -> tuple[int, ...]:
        """Process-global ranks in group-local index order."""
        return self._group_ranks

    @property
    def world_size(self) -> int:
        """Number of owning ranks (``len(group_ranks)``)."""
        return len(self._group_ranks)

    @property
    def logical_len(self) -> int:
        """Number of logical rows; padding never adds rows here."""
        return len(self._global_rows)

    @property
    def global_to_local(self) -> Mapping[int, int]:
        """Process-global rank -> group-local index mapping."""
        return MappingProxyType(self._global_to_local)

    @property
    def global_rows(self) -> tuple[ExecutionRow, ...]:
        """Canonical execution rows (input order)."""
        return self._global_rows

    @property
    def owner_rows(self) -> tuple[ExecutionRow, ...]:
        """Execution rows in owner-major order (stable within buckets)."""
        return self._owner_rows

    @property
    def owner_counts(self) -> tuple[int, ...]:
        """Rows per owner in group-local rank order."""
        return self._counts

    @property
    def owner_offsets(self) -> tuple[int, ...]:
        """Prefix offsets into owner-major arrays; length world_size + 1."""
        return self._offsets

    @property
    def forward_permutation(self) -> tuple[int, ...]:
        """Owner-major index -> canonical index."""
        return self._forward

    @property
    def inverse_permutation(self) -> tuple[int, ...]:
        """Canonical index -> owner-major index."""
        return self._inverse

    # -- rank mapping --------------------------------------------------------

    def local_rank_of_global(self, global_rank: int) -> int:
        """Group-local index of a process-global rank (unknown -> error)."""
        if (
            isinstance(global_rank, bool)
            or not isinstance(global_rank, int)
            or global_rank not in self._global_to_local
        ):
            raise OwnerLayoutError(f"unknown global rank {global_rank!r}")
        return self._global_to_local[global_rank]

    def global_rank_of_local(self, local_index: int) -> int:
        """Process-global rank of a group-local index."""
        if (
            isinstance(local_index, bool)
            or not isinstance(local_index, int)
            or not 0 <= local_index < self.world_size
        ):
            raise OwnerLayoutError(f"invalid local rank {local_index!r}")
        return self._group_ranks[local_index]

    def owner_slice_for_local(self, local_index: int) -> slice:
        """Slice of owner-major arrays owned by the given local rank."""
        if (
            isinstance(local_index, bool)
            or not isinstance(local_index, int)
            or not 0 <= local_index < self.world_size
        ):
            raise OwnerLayoutError(f"invalid local rank {local_index!r}")
        return slice(self._offsets[local_index], self._offsets[local_index + 1])

    # -- payload helpers -----------------------------------------------------

    def forward(self, payload: Sequence[object]) -> list[object]:
        """Reorder an exact-length sequence into owner-major order."""
        self._require_length(_first_axis_length(payload), "payload")
        return [payload[index] for index in self._forward]

    def restore(self, payload: Sequence[object]) -> list[object]:
        """Restore an exact-length owner-major sequence to canonical order."""
        self._require_length(_first_axis_length(payload), "payload")
        return [payload[index] for index in self._inverse]

    def forward_tensor(self, tensor: object, logical_len: int | None = None) -> object:
        """Owner-major reorder of the first tensor axis.

        With ``logical_len`` unset the leading axis must exactly equal the
        logical length.  With ``logical_len`` the caller declares the exact
        logical prefix length (it must equal the layout's logical length);
        rows beyond the prefix are physical padding and are left in place.
        Duck-typed first-axis indexing preserves dtype and trailing shape.
        """
        index = self._tensor_index(tensor, logical_len, self._forward)
        return tensor[index]

    def restore_tensor(self, tensor: object, logical_len: int | None = None) -> object:
        """Restore the first tensor axis to canonical order; see
        :meth:`forward_tensor` for the ``logical_len`` padding contract."""
        index = self._tensor_index(tensor, logical_len, self._inverse)
        return tensor[index]

    # -- internals -----------------------------------------------------------

    def _require_length(self, length: int, name: str) -> None:
        if length != self.logical_len:
            raise OwnerLayoutError(
                f"{name} length {length} != logical length {self.logical_len}"
            )

    def _tensor_index(
        self, tensor: object, logical_len: int | None, permutation: Sequence[int]
    ) -> list[int]:
        leading = _first_axis_length(tensor)
        if logical_len is None:
            self._require_length(leading, "tensor leading axis")
            return list(permutation)
        _require_nonneg_int(logical_len, "logical_len")
        if logical_len != self.logical_len:
            raise OwnerLayoutError(
                f"logical_len {logical_len} != logical length {self.logical_len}"
            )
        if leading < logical_len:
            raise OwnerLayoutError(
                f"tensor leading axis {leading} < logical length {logical_len}"
            )
        return list(permutation) + list(range(logical_len, leading))


def balanced_decode_graph_signature(
    layout: OwnerRowLayout,
    *,
    num_reqs: int,
    num_tokens: int,
    uniform_decode: bool,
) -> RequestOwnedGraphSignature | None:
    """Return the fixed request-owned uniform-decode signature, or ``None``.

    The graphable envelope remains intentionally narrow: the same positive
    number of requests per owner and the same positive number of token rows
    per request. The per-owner count is the product of those two dimensions.
    The graph key is normalized to identity because the runner-owned arena
    stages the live canonical/owner permutation into fixed-address tensors
    before replay; request churn therefore changes values, not graph shape.
    A caller must treat ``None`` as "FULL forbidden" and fall back to
    PIECEWISE; it must never reinterpret it as the baseline (non-owner) key.
    """

    if not isinstance(layout, OwnerRowLayout):
        raise OwnerLayoutError(f"layout must be an OwnerRowLayout, got {layout!r}")
    reqs = _require_nonneg_int(num_reqs, "num_reqs")
    tokens = _require_nonneg_int(num_tokens, "num_tokens")
    if not isinstance(uniform_decode, bool):
        raise OwnerLayoutError(f"uniform_decode must be a bool, got {uniform_decode!r}")
    if not uniform_decode or reqs == 0 or tokens == 0 or tokens % reqs != 0:
        return None
    if layout.logical_len != tokens or reqs % layout.world_size != 0:
        return None
    token_rows_per_owner = tokens // layout.world_size
    expected_identity = tuple(range(tokens))
    if layout.owner_counts != (token_rows_per_owner,) * layout.world_size:
        return None
    return RequestOwnedGraphSignature(
        owner_counts=layout.owner_counts,
        canonical_to_owner=expected_identity,
    )


class OwnerCollectivePlan:
    """Immutable all_to_all_single plan for one owner of a row layout.

    Binds an :class:`OwnerRowLayout` to one process-global rank of the
    owning group (the local process) and derives the exact split vectors
    and expected row identities of the two unconditional raw
    ``all_to_all_single`` exchanges:

    * owner_to_all / fanout: every rank sends its local owner rows to
      every rank, so after the exchange every rank holds the full
      owner-major rows array.  The send buffer is the local owner rows
      tiled ``world_size`` times (one tile per destination): the input
      split vector is ``(local_owner_row_count,) * world_size`` and the
      output split vector is ``layout.owner_counts`` (the segment
      received from source ``j`` is that source's owner bucket).
    * all_to_owner / fanin: every rank sends the full owner-major rows
      array back to the owners and destination ``j`` keeps only its own
      bucket.  The input split vector is ``layout.owner_counts`` and the
      output split vector is ``(local_owner_row_count,) * world_size``:
      the local owner receives its stable rows once per source, in
      source-rank-major order.

    Everything here is pure Python and group-local.  The process-global
    rank is resolved through :meth:`OwnerRowLayout.local_rank_of_global`,
    never by assuming a rank range or ordering, so the group may be
    noncontiguous and unsorted.

    Args:
        layout: the owner-major row layout of this execution step.
        owner_global_rank: process-global rank of the local process; it
            must be an ``int`` member of ``layout.group_ranks``.

    Raises:
        OwnerLayoutError: on a non-``OwnerRowLayout`` binding, a
            bool/non-int rank, or a rank that is not a group member.
    """

    __slots__ = (
        "_layout",
        "_owner_global_rank",
        "_local_rank",
        "_local_owner_count",
        "_local_owner_slice",
        "_local_owner_rows",
        "_owner_to_all_input_splits",
        "_owner_to_all_output_splits",
        "_all_to_owner_input_splits",
        "_all_to_owner_output_splits",
        "_fanout_send_rows",
        "_fanout_receive_rows",
        "_fanin_send_rows",
        "_fanin_receive_rows",
        "_frozen",
    )

    def __init__(self, layout: OwnerRowLayout, owner_global_rank: int) -> None:
        if not isinstance(layout, OwnerRowLayout):
            raise OwnerLayoutError(f"layout must be an OwnerRowLayout, got {layout!r}")
        if isinstance(owner_global_rank, bool) or not isinstance(
            owner_global_rank, int
        ):
            raise OwnerLayoutError(
                f"owner_global_rank must be an int, got {owner_global_rank!r}"
            )
        local_rank = layout.local_rank_of_global(owner_global_rank)
        world_size = layout.world_size
        count = layout.owner_counts[local_rank]
        owner_slice = layout.owner_slice_for_local(local_rank)
        local_owner_rows = tuple(layout.owner_rows[owner_slice])
        tile = (count,) * world_size
        owner_counts = tuple(layout.owner_counts)
        owner_rows = tuple(layout.owner_rows)
        self._layout = layout
        self._owner_global_rank = owner_global_rank
        self._local_rank = local_rank
        self._local_owner_count = count
        self._local_owner_slice = owner_slice
        self._local_owner_rows = local_owner_rows
        self._owner_to_all_input_splits = tile
        self._owner_to_all_output_splits = owner_counts
        self._all_to_owner_input_splits = owner_counts
        self._all_to_owner_output_splits = tile
        self._fanout_send_rows = local_owner_rows * world_size
        self._fanout_receive_rows = owner_rows
        self._fanin_send_rows = owner_rows
        self._fanin_receive_rows = local_owner_rows * world_size
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"OwnerCollectivePlan is immutable: cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"OwnerCollectivePlan is immutable: cannot delete {name!r}"
        )

    # -- binding and rank mapping --------------------------------------------

    @property
    def layout(self) -> OwnerRowLayout:
        """The owner-major row layout this plan is bound to."""
        return self._layout

    @property
    def owner_global_rank(self) -> int:
        """Process-global rank of the local process."""
        return self._owner_global_rank

    @property
    def local_rank(self) -> int:
        """Group-local index of the local process."""
        return self._local_rank

    @property
    def world_size(self) -> int:
        """Number of owning ranks (``layout.world_size``)."""
        return self._layout.world_size

    # -- local owner state ---------------------------------------------------

    @property
    def local_owner_count(self) -> int:
        """Number of rows owned by the local process."""
        return self._local_owner_count

    @property
    def local_owner_slice(self) -> slice:
        """Slice of owner-major arrays holding the local owner's rows."""
        return self._local_owner_slice

    @property
    def local_owner_rows(self) -> tuple[ExecutionRow, ...]:
        """The local owner's stable rows in owner-major order."""
        return self._local_owner_rows

    # -- owner_to_all / fanout -----------------------------------------------

    @property
    def owner_to_all_input_splits(self) -> tuple[int, ...]:
        """Fanout input split vector: ``(local_owner_count,) * world_size``."""
        return self._owner_to_all_input_splits

    @property
    def owner_to_all_output_splits(self) -> tuple[int, ...]:
        """Fanout output split vector: ``layout.owner_counts``."""
        return self._owner_to_all_output_splits

    @property
    def fanout_send_rows(self) -> tuple[ExecutionRow, ...]:
        """Fanout send identities: local owner rows tiled once per rank."""
        return self._fanout_send_rows

    @property
    def fanout_receive_rows(self) -> tuple[ExecutionRow, ...]:
        """Fanout receive identities: the full owner-major rows array."""
        return self._fanout_receive_rows

    # -- all_to_owner / fanin ------------------------------------------------

    @property
    def all_to_owner_input_splits(self) -> tuple[int, ...]:
        """Fanin input split vector: ``layout.owner_counts``."""
        return self._all_to_owner_input_splits

    @property
    def all_to_owner_output_splits(self) -> tuple[int, ...]:
        """Fanin output split vector: ``(local_owner_count,) * world_size``."""
        return self._all_to_owner_output_splits

    @property
    def fanin_send_rows(self) -> tuple[ExecutionRow, ...]:
        """Fanin send identities: the full owner-major rows array."""
        return self._fanin_send_rows

    @property
    def fanin_receive_rows(self) -> tuple[ExecutionRow, ...]:
        """Fanin receive identities: the local owner's stable rows repeated
        once per source rank, in source-rank-major order."""
        return self._fanin_receive_rows


# ---------------------------------------------------------------------------
# Step builder from scheduler lease tokens
# ---------------------------------------------------------------------------


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OwnerLayoutError(f"{name} must be a nonempty string, got {value!r}")
    return value


def _require_token_position(value: object, name: str) -> int:
    """Coerce a non-bool integer-like token position to a plain ``int``.

    Accepts anything implementing ``__index__`` (so numpy scalars work without
    a numpy import) and rejects ``bool`` aliases explicitly.
    """
    if isinstance(value, bool):
        raise OwnerLayoutError(f"{name} must be a nonnegative integer, got {value!r}")
    try:
        position = operator.index(value)
    except TypeError as exc:
        raise OwnerLayoutError(
            f"{name} must be a nonnegative integer, got {value!r}"
        ) from exc
    if position < 0:
        raise OwnerLayoutError(f"{name} must be a nonnegative integer, got {value!r}")
    return position


def build_owner_row_layout(
    step_seq: int,
    request_ids: Sequence[str],
    token_positions: Sequence[int],
    leases: Sequence[OwnerLeaseToken],
    group_ranks: Sequence[int],
) -> OwnerRowLayout:
    """Build the immutable :class:`OwnerRowLayout` for one execution step.

    The worker constructs this only after the flattened execution order is
    known: ``request_ids`` and ``token_positions`` are the per-row flattened
    request ids and absolute logical token positions in actual execution-row
    order (one entry per row, equal lengths).  ``leases`` are the scheduler's
    published :class:`OwnerLeaseToken` sequence for this step and
    ``group_ranks`` the process-global ranks of the owning group in
    group-local index order.

    Contracts (all fail closed via :class:`OwnerLayoutError`):

    * ``request_ids`` and ``token_positions`` have exactly equal lengths and
      every request id is a nonempty string.
    * Every token position is a nonnegative non-bool integer and is
      strictly below the matching lease's ``runnable_num_tokens``
      (exclusive 0-based upper bound: ``position < runnable_num_tokens``;
      the boundary itself is never a legal position).
    * There is exactly one lease per distinct scheduled request: no missing,
      extra, or duplicate leases; every lease's ``step_seq`` exactly equals
      ``step_seq``; the lease key's request id matches the row request id and
      supplies the request epoch; the owner is a member of ``group_ranks``.
    * Every row uses logical lane 0.

    Empty row input requires empty ``leases`` and produces a valid zero-row
    layout (``group_ranks`` must still be nonempty).  The returned layout is
    immutable; no tensor or device state is staged here.
    """
    step = _require_nonneg_int(step_seq, "step_seq")
    rows_request_ids = [
        _require_nonempty_string(request_id, "request_ids[i]")
        for request_id in request_ids
    ]
    positions = [
        _require_token_position(position, "token_positions[i]")
        for position in token_positions
    ]
    if len(rows_request_ids) != len(positions):
        raise OwnerLayoutError(
            "request_ids and token_positions must have exactly equal lengths, "
            f"got {len(rows_request_ids)} and {len(positions)}"
        )
    group = _validate_group_ranks(group_ranks)
    group_local = {rank: local for local, rank in enumerate(group)}

    lease_by_request_id: dict[str, OwnerLeaseToken] = {}
    for lease in leases:
        if not isinstance(lease, OwnerLeaseToken):
            raise OwnerLayoutError(
                f"leases must contain OwnerLeaseToken, got {lease!r}"
            )
        lease_request_id = _require_nonempty_string(
            lease.key.request_id, "lease key request_id"
        )
        if isinstance(lease.step_seq, bool) or not isinstance(lease.step_seq, int):
            raise OwnerLayoutError(
                "lease step_seq must be a non-bool int for request "
                f"{lease_request_id!r}, got {lease.step_seq!r}"
            )
        if lease.step_seq != step:
            raise OwnerLayoutError(
                "lease step_seq must exactly equal the layout step_seq; got "
                f"{lease.step_seq} for request {lease_request_id!r}, "
                f"expected {step}"
            )
        if lease_request_id in lease_by_request_id:
            raise OwnerLayoutError(
                f"duplicate lease for scheduled request {lease_request_id!r}"
            )
        lease_by_request_id[lease_request_id] = lease

    row_request_id_set = set(rows_request_ids)
    missing = row_request_id_set - lease_by_request_id.keys()
    if missing:
        raise OwnerLayoutError(f"no lease for scheduled request {sorted(missing)[0]!r}")
    extra = lease_by_request_id.keys() - row_request_id_set
    if extra:
        raise OwnerLayoutError(f"lease for unscheduled request {sorted(extra)[0]!r}")

    row_ids: list[GlobalRowId] = []
    owner_by_request_uid: dict[OwnerLeaseKey, int] = {}
    for request_id, position in zip(rows_request_ids, positions):
        lease = lease_by_request_id[request_id]
        owner = lease.owner_id
        if (
            isinstance(owner, bool)
            or not isinstance(owner, int)
            or owner not in group_local
        ):
            raise OwnerLayoutError(
                f"lease owner {owner!r} for request {request_id!r} is not in "
                f"group_ranks {group!r}"
            )
        if isinstance(lease.runnable_num_tokens, bool) or not isinstance(
            lease.runnable_num_tokens, int
        ):
            raise OwnerLayoutError(
                f"lease runnable_num_tokens must be an int for request "
                f"{request_id!r}, got {lease.runnable_num_tokens!r}"
            )
        if position >= lease.runnable_num_tokens:
            raise OwnerLayoutError(
                f"token position {position} for request {request_id!r} is at or "
                f"beyond the exclusive bound runnable_num_tokens "
                f"{lease.runnable_num_tokens}"
            )
        key = OwnerLeaseKey(request_id=request_id, owner_epoch=lease.key.owner_epoch)
        row_ids.append(GlobalRowId(request_uid=key, logical_token_position=position))
        owner_by_request_uid[key] = owner
    return OwnerRowLayout(step, row_ids, owner_by_request_uid, group)
