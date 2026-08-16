# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advisory owner-prefix directory with bounded affinity spill.

The directory records only immutable content hashes and owner ids.  It is a
placement hint, never cache or lifecycle authority: the selected worker still
performs the exact hybrid lookup and reports the actual hit in its RESERVE
receipt.  Stale hints can therefore cost locality but cannot create a false
hit or over-admit physical capacity.
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.request_owned_prefix import OwnerPrefixDescriptor
from vllm.v1.core.sched.ownership import OwnerCommandKind, OwnerReceipt
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class OwnerPrefixPlacement:
    owner_id: int
    matched_tokens: int


class RequestOwnedPrefixDirectory:
    """Compact scheduler-local prefix hints for one owner world."""

    def __init__(
        self,
        world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_hashes_per_owner: int = 16_384,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if scheduler_block_size <= 0 or hash_block_size <= 0:
            raise ValueError("prefix directory block sizes must be positive")
        if scheduler_block_size % hash_block_size:
            raise ValueError(
                "scheduler_block_size must be divisible by hash_block_size"
            )
        if max_hashes_per_owner <= 0:
            raise ValueError("max_hashes_per_owner must be positive")
        self.world_size = world_size
        self.scheduler_block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        self.max_hashes_per_owner = max_hashes_per_owner
        self._hash_stride = scheduler_block_size // hash_block_size
        self._hashes_by_owner: tuple[OrderedDict[bytes, None], ...] = tuple(
            OrderedDict() for _ in range(world_size)
        )

    def observe_computed_prefix(
        self,
        owner_id: int,
        block_hashes: Sequence[BlockHash],
        num_computed_tokens: int,
        num_prompt_tokens: int,
    ) -> None:
        """Remember scheduler-aligned immutable prompt boundaries.

        Observation happens only at the next synchronous schedule call after
        a successful worker output. Generated suffix hashes are intentionally
        excluded until their transport/commit protocol is explicit.
        """

        self._validate_owner(owner_id)
        committed = min(
            num_computed_tokens,
            num_prompt_tokens,
            len(block_hashes) * self.hash_block_size,
        )
        num_boundaries = committed // self.scheduler_block_size
        owner_hashes = self._hashes_by_owner[owner_id]
        first_new_boundary = 1
        # A chained boundary already in the LRU proves every earlier content
        # boundary was observed in the same chain. Find the newest retained
        # one so steady decode observation is O(1), not O(prompt blocks).
        for boundary in range(num_boundaries, 0, -1):
            block_hash = bytes(
                block_hashes[boundary * self._hash_stride - 1]
            )
            if block_hash in owner_hashes:
                owner_hashes.move_to_end(block_hash)
                first_new_boundary = boundary + 1
                break
        for boundary in range(first_new_boundary, num_boundaries + 1):
            block_hash = bytes(
                block_hashes[boundary * self._hash_stride - 1]
            )
            owner_hashes[block_hash] = None
            owner_hashes.move_to_end(block_hash)
        while len(owner_hashes) > self.max_hashes_per_owner:
            owner_hashes.popitem(last=False)

    def longest_match(
        self,
        owner_id: int,
        block_hashes: Sequence[BlockHash],
        num_prompt_tokens: int,
    ) -> int:
        """Return the longest contiguous scheduler-aligned hinted prefix."""

        self._validate_owner(owner_id)
        max_tokens = min(
            max(num_prompt_tokens - 1, 0),
            len(block_hashes) * self.hash_block_size,
        )
        max_boundaries = max_tokens // self.scheduler_block_size
        owner_hashes = self._hashes_by_owner[owner_id]
        # Hashes are chained over all preceding blocks, so one matching
        # terminal boundary identifies its complete content prefix. Search
        # longest-first; the worker remains the exact physical authority.
        for boundary in range(max_boundaries, 0, -1):
            block_hash = bytes(
                block_hashes[boundary * self._hash_stride - 1]
            )
            if block_hash in owner_hashes:
                owner_hashes.move_to_end(block_hash)
                return boundary * self.scheduler_block_size
        return 0

    def select_with_bounded_affinity(
        self,
        block_hashes: Sequence[BlockHash],
        num_prompt_tokens: int,
        projected_free: Mapping[int, int],
        fresh_demand: Mapping[int, int],
        live_leases: Mapping[int, int],
    ) -> OwnerPrefixPlacement | None:
        """Prefer a prefix hit without spending more than one request of skew.

        All mappings cover the already capacity-feasible owners. The owner
        with greatest projected free capacity is the load baseline. A cached
        owner remains affinity-eligible only while its deficit from that
        baseline is no greater than its own conservative fresh-request
        demand. Once the deficit grows larger, selection spills to a colder
        owner; computing there naturally creates another immutable replica.
        """

        owners = tuple(sorted(projected_free))
        if not owners:
            return None
        if set(fresh_demand) != set(owners) or set(live_leases) != set(owners):
            raise ValueError("prefix placement mappings must cover the same owners")
        for owner_id in owners:
            self._validate_owner(owner_id)

        best_free = max(projected_free.values())
        affinity_eligible = tuple(
            owner_id
            for owner_id in owners
            if best_free - projected_free[owner_id]
            <= max(fresh_demand[owner_id], 1)
        )
        matches = {
            owner_id: self.longest_match(
                owner_id, block_hashes, num_prompt_tokens
            )
            for owner_id in affinity_eligible
        }
        if max(matches.values(), default=0) == 0:
            candidates = owners
            matches = {owner_id: 0 for owner_id in owners}
        else:
            candidates = affinity_eligible

        owner = min(
            candidates,
            key=lambda owner_id: (
                -matches[owner_id],
                -projected_free[owner_id],
                live_leases[owner_id],
                owner_id,
            ),
        )
        return OwnerPrefixPlacement(owner, matches[owner])

    def _validate_owner(self, owner_id: int) -> None:
        if (
            isinstance(owner_id, bool)
            or not isinstance(owner_id, int)
            or not 0 <= owner_id < self.world_size
        ):
            raise ValueError(f"owner_id {owner_id!r} is outside the owner world")

    def reset(self) -> None:
        """Discard advisory placement state; physical workers reset separately."""

        for owner_hashes in self._hashes_by_owner:
            owner_hashes.clear()


class RequestOwnedPrefixScheduler:
    """Narrow scheduler adapter around the advisory directory and wire seal."""

    def __init__(
        self,
        enabled: bool,
        world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
    ) -> None:
        self.enabled = enabled
        self.scheduler_block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        self.directory = (
            RequestOwnedPrefixDirectory(
                world_size, scheduler_block_size, hash_block_size
            )
            if enabled
            else None
        )

    def observe_running(self, requests: Iterable[Request]) -> None:
        directory = self.directory
        if directory is None:
            return
        for request in requests:
            owner_id = request.attention_owner
            if owner_id is not None and not request.skip_reading_prefix_cache:
                directory.observe_computed_prefix(
                    owner_id,
                    request.block_hashes,
                    request.num_computed_tokens,
                    request.num_prompt_tokens,
                )

    def observe_scheduled(
        self,
        requests: Mapping[str, Request],
        request_ids: Iterable[str],
    ) -> None:
        """Observe terminally successful work before finished rows disappear."""

        self.observe_running(
            request
            for request_id in request_ids
            if (request := requests.get(request_id)) is not None
        )

    def descriptor(self, request: Request) -> OwnerPrefixDescriptor | None:
        if not self.enabled or request.skip_reading_prefix_cache:
            return None
        # ``Request.block_hashes`` grows with generated output.  This protocol
        # deliberately publishes only immutable prompt prefixes; transporting
        # suffix hashes on a resumed request would otherwise make worker-side
        # publication silently outrun the scheduler directory's prompt-only
        # authority boundary.
        prompt_hashes = request.num_prompt_tokens // self.hash_block_size
        return OwnerPrefixDescriptor(tuple(request.block_hashes[:prompt_hashes]))

    def select_owner(
        self,
        request: Request,
        projected_free: Mapping[int, int],
        fresh_demand: Mapping[int, int],
        live_leases: Mapping[int, int],
    ) -> int | None:
        directory = self.directory
        if directory is None or request.skip_reading_prefix_cache:
            return None
        placement = directory.select_with_bounded_affinity(
            request.block_hashes,
            request.num_prompt_tokens,
            projected_free,
            fresh_demand,
            live_leases,
        )
        return placement.owner_id if placement is not None else None

    def validate_reserve_receipt(
        self,
        event: OwnerReceipt,
        pending: tuple[int, OwnerCommandKind] | None,
        request: Request | None,
    ) -> int | None:
        """Validate and return an exact hit before scheduler mutation."""

        matching_reserve = pending == (
            event.command_seq,
            OwnerCommandKind.RESERVE,
        )
        if not event.accepted or not matching_reserve:
            return None
        expects_hit = bool(
            self.enabled
            and request is not None
            and not request.skip_reading_prefix_cache
        )
        hit = event.prefix_cache_hit_tokens
        if expects_hit and hit is None:
            raise RuntimeError(
                "accepted request-owned RESERVE is missing its exact "
                f"owner-local prefix hit for {event.key}."
            )
        if not expects_hit and hit is not None:
            raise RuntimeError(
                "request-owned RESERVE published a prefix hit when lookup "
                f"was disabled for {event.key}."
            )
        if hit is None:
            return None
        assert request is not None
        max_hit = max(request.num_prompt_tokens - 1, 0)
        hash_coverage = len(request.block_hashes) * self.hash_block_size
        if (
            hit > min(max_hit, hash_coverage)
            or hit % self.scheduler_block_size
        ):
            raise RuntimeError(
                "request-owned prefix hit violates the scheduler boundary "
                f"for {event.key}: hit={hit}, max={max_hit}, "
                f"hash_coverage={hash_coverage}, "
                f"alignment={self.scheduler_block_size}."
            )
        if hit < request.num_computed_tokens:
            raise RuntimeError(
                "request-owned prefix hit regresses computed progress for "
                f"{event.key}: {hit} < {request.num_computed_tokens}."
            )
        return hit

    def record_lookup(
        self,
        request: Request,
        hit_tokens: int,
        stats: PrefixCacheStats | None,
    ) -> None:
        """Expose worker-exact owner hits through the standard cache metrics."""

        if stats is None or request.num_computed_tokens != 0:
            return
        stats.record(
            num_tokens=request.num_prompt_tokens,
            num_hits=hit_tokens,
            preempted=request.num_preemptions > 0,
        )
        stats.record_block_lookup(
            num_queries=len(request.block_hashes),
            num_hits=hit_tokens // self.hash_block_size,
        )

    def reset(self) -> None:
        directory = self.directory
        if directory is not None:
            directory.reset()
