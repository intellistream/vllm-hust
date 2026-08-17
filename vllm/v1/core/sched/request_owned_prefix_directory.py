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
from typing import Literal

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.request_owned_prefix import OwnerPrefixDescriptor
from vllm.v1.core.sched.ownership import OwnerCommandKind, OwnerReceipt
from vllm.v1.core.sched.request_owned_prefix_observability import (
    OwnerPrefixAdmissionObservation,
    OwnerPrefixCandidateObservation,
    OwnerPrefixObservation,
    OwnerPrefixObservationSink,
    OwnerPrefixPublicationObservation,
    OwnerPrefixReserveObservation,
    make_prefix_observation_identity,
)
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

        return self._longest_match(
            owner_id,
            block_hashes,
            num_prompt_tokens,
            touch=True,
        )

    def peek_longest_match(
        self,
        owner_id: int,
        block_hashes: Sequence[BlockHash],
        num_prompt_tokens: int,
    ) -> int:
        """Inspect one advisory match without changing its LRU recency."""

        return self._longest_match(
            owner_id,
            block_hashes,
            num_prompt_tokens,
            touch=False,
        )

    def _longest_match(
        self,
        owner_id: int,
        block_hashes: Sequence[BlockHash],
        num_prompt_tokens: int,
        *,
        touch: bool,
    ) -> int:
        """Return one match, optionally retaining the existing LRU touch."""

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
                if touch:
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

    _MAX_OBSERVED_REQUESTS = 16_384

    def __init__(
        self,
        enabled: bool,
        world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        observation_sink: OwnerPrefixObservationSink | None = None,
    ) -> None:
        self.enabled = enabled
        self.scheduler_block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        self._observation_sink = observation_sink
        self._pending_admissions: dict[str, OwnerPrefixAdmissionObservation] = {}
        self._reserve_hits: dict[str, int] = {}
        self._published_tokens: OrderedDict[str, int] = OrderedDict()
        self.directory = (
            RequestOwnedPrefixDirectory(
                world_size, scheduler_block_size, hash_block_size
            )
            if enabled
            else None
        )

    def set_observation_sink(
        self,
        observation_sink: OwnerPrefixObservationSink | None,
    ) -> None:
        """Replace the optional exact-evidence sink at a clean boundary."""

        self._observation_sink = observation_sink
        self._pending_admissions.clear()
        self._reserve_hits.clear()
        self._published_tokens.clear()

    def observe_running(self, requests: Iterable[Request]) -> None:
        directory = self.directory
        if directory is None:
            return
        for request in requests:
            owner_id = request.attention_owner
            if owner_id is not None and not request.skip_reading_prefix_cache:
                previous_hint = directory.peek_longest_match(
                    owner_id,
                    request.block_hashes,
                    request.num_prompt_tokens,
                )
                directory.observe_computed_prefix(
                    owner_id,
                    request.block_hashes,
                    request.num_computed_tokens,
                    request.num_prompt_tokens,
                )
                self._observe_publication(request, owner_id, previous_hint)

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
        if placement is not None and self._observation_sink is not None:
            owners = tuple(sorted(projected_free))
            best_free = max(projected_free.values())
            candidates = tuple(
                OwnerPrefixCandidateObservation(
                    owner_id=owner_id,
                    projected_free=projected_free[owner_id],
                    fresh_demand=fresh_demand[owner_id],
                    live_leases=live_leases[owner_id],
                    hinted_tokens=directory.peek_longest_match(
                        owner_id,
                        request.block_hashes,
                        request.num_prompt_tokens,
                    ),
                    affinity_eligible=(
                        best_free - projected_free[owner_id]
                        <= max(fresh_demand[owner_id], 1)
                    ),
                )
                for owner_id in owners
            )
            prefix_digest, prefix_tokens = make_prefix_observation_identity(
                request.block_hashes,
                request.num_prompt_tokens,
                self.scheduler_block_size,
                self.hash_block_size,
            )
            reason: Literal["affinity_hit", "bounded_spill", "cold_balance"]
            if placement.matched_tokens > 0:
                reason = "affinity_hit"
            elif any(candidate.hinted_tokens > 0 for candidate in candidates):
                reason = "bounded_spill"
            else:
                reason = "cold_balance"
            observation = OwnerPrefixAdmissionObservation(
                request_id=request.request_id,
                prefix_digest=prefix_digest,
                prefix_tokens=prefix_tokens,
                selected_owner=placement.owner_id,
                selected_hinted_tokens=placement.matched_tokens,
                reason=reason,
                candidates=candidates,
            )
            self._reserve_hits.pop(request.request_id, None)
            self._published_tokens.pop(request.request_id, None)
            self._pending_admissions[request.request_id] = observation
            self._emit(observation)
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
        if not matching_reserve:
            return None
        admission = self._pending_admissions.pop(event.key.request_id, None)
        if not event.accepted:
            self._emit_reserve_observation(
                event,
                request,
                admission,
                exact_hit_tokens=None,
                outcome="rejected_reserve",
            )
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
        hinted_tokens = admission.selected_hinted_tokens if admission is not None else 0
        outcome: Literal["exact_hit", "cold_miss", "stale_hint_miss"]
        if hit > 0:
            outcome = "exact_hit"
        elif hinted_tokens > 0:
            outcome = "stale_hint_miss"
        else:
            outcome = "cold_miss"
        self._emit_reserve_observation(
            event,
            request,
            admission,
            exact_hit_tokens=hit,
            outcome=outcome,
        )
        if self._observation_sink is not None:
            self._reserve_hits[event.key.request_id] = hit
            self._record_published_tokens(event.key.request_id, hit)
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
        self._pending_admissions.clear()
        self._reserve_hits.clear()
        self._published_tokens.clear()

    def _observe_publication(
        self,
        request: Request,
        owner_id: int,
        previous_hint: int,
    ) -> None:
        if self._observation_sink is None:
            return
        prefix_digest, prefix_tokens = make_prefix_observation_identity(
            request.block_hashes,
            request.num_prompt_tokens,
            self.scheduler_block_size,
            self.hash_block_size,
        )
        if prefix_digest is None:
            return
        previous_tokens = self._published_tokens.get(
            request.request_id,
            self._reserve_hits.get(request.request_id, 0),
        )
        published_tokens = min(request.num_computed_tokens, prefix_tokens)
        published_tokens = (
            published_tokens // self.scheduler_block_size * self.scheduler_block_size
        )
        if published_tokens > previous_tokens:
            self._emit(
                OwnerPrefixPublicationObservation(
                    request_id=request.request_id,
                    owner_id=owner_id,
                    prefix_digest=prefix_digest,
                    previous_tokens=previous_tokens,
                    published_tokens=published_tokens,
                    exact_reserve_hit_tokens=self._reserve_hits.get(request.request_id),
                    directory_hint_was_new=previous_hint < published_tokens,
                )
            )
            self._record_published_tokens(
                request.request_id,
                published_tokens,
            )
        if published_tokens >= prefix_tokens:
            self._reserve_hits.pop(request.request_id, None)

    def _emit_reserve_observation(
        self,
        event: OwnerReceipt,
        request: Request | None,
        admission: OwnerPrefixAdmissionObservation | None,
        *,
        exact_hit_tokens: int | None,
        outcome: Literal[
            "exact_hit",
            "cold_miss",
            "stale_hint_miss",
            "rejected_reserve",
        ],
    ) -> None:
        if self._observation_sink is None:
            return
        if admission is not None:
            prefix_digest = admission.prefix_digest
            prefix_tokens = admission.prefix_tokens
            hinted_tokens = admission.selected_hinted_tokens
        elif request is not None:
            prefix_digest, prefix_tokens = make_prefix_observation_identity(
                request.block_hashes,
                request.num_prompt_tokens,
                self.scheduler_block_size,
                self.hash_block_size,
            )
            directory = self.directory
            hinted_tokens = (
                directory.peek_longest_match(
                    event.owner_id,
                    request.block_hashes,
                    request.num_prompt_tokens,
                )
                if directory is not None
                else 0
            )
        else:
            prefix_digest = None
            prefix_tokens = 0
            hinted_tokens = 0
        self._emit(
            OwnerPrefixReserveObservation(
                request_id=event.key.request_id,
                owner_epoch=event.key.owner_epoch,
                owner_id=event.owner_id,
                prefix_digest=prefix_digest,
                prefix_tokens=prefix_tokens,
                hinted_tokens=hinted_tokens,
                accepted=event.accepted,
                exact_hit_tokens=exact_hit_tokens,
                outcome=outcome,
            )
        )

    def _emit(self, observation: OwnerPrefixObservation) -> None:
        if self._observation_sink is not None:
            self._observation_sink(observation)

    def _record_published_tokens(
        self,
        request_id: str,
        published_tokens: int,
    ) -> None:
        self._published_tokens[request_id] = published_tokens
        self._published_tokens.move_to_end(request_id)
        while len(self._published_tokens) > self._MAX_OBSERVED_REQUESTS:
            self._published_tokens.popitem(last=False)
