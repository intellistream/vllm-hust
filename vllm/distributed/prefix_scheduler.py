# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-aware routing helpers for distributed vLLM deployments.

The global scheduler stores prefix-cache block hashes reported by each vLLM
node and routes a new request to the node with the longest cached prefix.
"""

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import count
from typing import TypeAlias

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    maybe_convert_block_hash,
)

PrefixBlockHash: TypeAlias = ExternalBlockHash


@dataclass
class PrefixRouteDecision:
    """Routing decision returned by ``GlobalPrefixScheduler``."""

    node_id: str
    matched_tokens: int
    data_parallel_rank: int | None = None
    view_epoch: int = 0
    worker_incarnation: str | None = None


@dataclass
class PrefixCacheSnapshot:
    """Full prefix-cache view exported by one vLLM node.

    ``group_hashes`` stores one hash set per KV-cache group. Standard full
    attention models usually have one group. Hybrid attention models may have
    multiple groups with different block sizes.
    """

    node_id: str
    hash_block_size: int
    group_block_sizes: dict[int, int]
    group_hashes: dict[int, set[PrefixBlockHash]]
    data_parallel_rank: int | None = None


@dataclass
class NodePrefixCacheState:
    """Prefix-cache index for one vLLM node."""

    node_id: str
    hash_block_size: int
    data_parallel_rank: int | None = None
    group_block_sizes: dict[int, int] = field(default_factory=dict)
    group_hashes: dict[int, set[PrefixBlockHash]] = field(
        default_factory=lambda: defaultdict(set)
    )
    worker_incarnation: str | None = None
    view_epoch: int = 0
    last_receipt_at: float | None = None
    expires_at: float | None = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PrefixCacheSnapshot,
        *,
        worker_incarnation: str | None = None,
    ) -> "NodePrefixCacheState":
        return cls(
            node_id=snapshot.node_id,
            data_parallel_rank=snapshot.data_parallel_rank,
            hash_block_size=snapshot.hash_block_size,
            worker_incarnation=worker_incarnation,
            group_block_sizes=dict(snapshot.group_block_sizes),
            group_hashes=defaultdict(
                set,
                {
                    group_id: set(hashes)
                    for group_id, hashes in snapshot.group_hashes.items()
                },
            ),
        )

    def apply_snapshot(
        self,
        snapshot: PrefixCacheSnapshot,
        *,
        now: float | None = None,
        view_ttl_seconds: float | None = None,
        worker_incarnation: str | None = None,
        cache_generation: int | None = None,
    ) -> None:
        if snapshot.node_id != self.node_id:
            raise ValueError(
                f"snapshot for node {snapshot.node_id!r} cannot update "
                f"state for node {self.node_id!r}"
            )
        incarnation_changed = self._set_worker_incarnation(worker_incarnation)
        if incarnation_changed and cache_generation is not None:
            self.view_epoch = 0
        if cache_generation is not None and cache_generation < self.view_epoch:
            return
        self.data_parallel_rank = snapshot.data_parallel_rank
        self.hash_block_size = snapshot.hash_block_size
        self.group_block_sizes = dict(snapshot.group_block_sizes)
        self.group_hashes = defaultdict(
            set,
            {
                group_id: set(hashes)
                for group_id, hashes in snapshot.group_hashes.items()
            },
        )
        if cache_generation is None:
            self.view_epoch += 1
        else:
            self.view_epoch = cache_generation
        self._refresh_receipt(now, view_ttl_seconds)

    def apply_events(
        self,
        events: Iterable[KVCacheEvent],
        *,
        now: float | None = None,
        view_ttl_seconds: float | None = None,
        worker_incarnation: str | None = None,
        cache_generation: int | None = None,
        is_snapshot: bool = False,
    ) -> None:
        """Apply prefix-cache deltas emitted by a vLLM node."""
        incarnation_changed = self._set_worker_incarnation(worker_incarnation)
        if incarnation_changed and cache_generation is not None:
            self.view_epoch = 0
        if cache_generation is not None:
            if cache_generation < self.view_epoch:
                return
            if cache_generation == self.view_epoch and not is_snapshot:
                self._refresh_receipt(now, view_ttl_seconds)
                return
            if cache_generation > self.view_epoch + 1 and not is_snapshot:
                self.group_hashes.clear()
                self.group_block_sizes.clear()
        changed = incarnation_changed
        for event in events:
            if isinstance(event, BlockStored):
                group_idx = 0 if event.group_idx is None else event.group_idx
                self.group_block_sizes[group_idx] = event.block_size
                self.group_hashes[group_idx].update(event.block_hashes)
                changed = True
            elif isinstance(event, BlockRemoved):
                group_idx = 0 if event.group_idx is None else event.group_idx
                hashes = self.group_hashes.get(group_idx)
                if hashes is not None:
                    hashes.difference_update(event.block_hashes)
                changed = True
            elif isinstance(event, AllBlocksCleared):
                self.group_hashes.clear()
                changed = True
        if cache_generation is not None:
            self.view_epoch = cache_generation
        elif changed:
            self.view_epoch += 1
        self._refresh_receipt(now, view_ttl_seconds)

    def record_worker_receipt(
        self,
        *,
        now: float | None = None,
        view_ttl_seconds: float | None = None,
        worker_incarnation: str | None = None,
        cache_generation: int | None = None,
    ) -> None:
        """Refresh worker liveness without publishing new cache blocks."""
        if self._set_worker_incarnation(worker_incarnation):
            self.view_epoch += 1
        if cache_generation is not None:
            if cache_generation < self.view_epoch:
                return
            self.view_epoch = cache_generation
        self._refresh_receipt(now, view_ttl_seconds)

    def invalidate(
        self,
        *,
        worker_incarnation: str | None = None,
        now: float | None = None,
        view_ttl_seconds: float | None = None,
        cache_generation: int | None = None,
    ) -> None:
        self.group_hashes.clear()
        self.group_block_sizes.clear()
        if worker_incarnation is not None:
            self.worker_incarnation = worker_incarnation
        if cache_generation is None:
            self.view_epoch += 1
        elif cache_generation >= self.view_epoch:
            self.view_epoch = cache_generation
        self._refresh_receipt(now, view_ttl_seconds)

    def expire_if_stale(self, now: float) -> bool:
        if self.expires_at is None or now <= self.expires_at:
            return False
        self.group_hashes.clear()
        self.group_block_sizes.clear()
        self.last_receipt_at = None
        self.expires_at = None
        self.view_epoch += 1
        return True

    def _set_worker_incarnation(self, worker_incarnation: str | None) -> bool:
        if worker_incarnation is None:
            return False
        if self.worker_incarnation is None:
            self.worker_incarnation = worker_incarnation
            if not self.group_hashes:
                return False
            self.group_hashes.clear()
            self.group_block_sizes.clear()
            return True
        if self.worker_incarnation == worker_incarnation:
            return False
        self.worker_incarnation = worker_incarnation
        self.group_hashes.clear()
        self.group_block_sizes.clear()
        return True

    def _refresh_receipt(
        self, now: float | None, view_ttl_seconds: float | None
    ) -> None:
        if now is None:
            now = time.monotonic()
        self.last_receipt_at = now
        self.expires_at = None if view_ttl_seconds is None else now + view_ttl_seconds

    def longest_prefix_match(
        self,
        block_hashes: Sequence[BlockHash],
        prompt_num_tokens: int,
        max_cache_hit_length: int | None = None,
    ) -> int:
        """Return the longest cached prefix length in tokens for this node."""
        if not block_hashes or not self.group_hashes:
            return 0

        max_length = prompt_num_tokens - 1
        if max_cache_hit_length is not None:
            max_length = min(max_length, max_cache_hit_length)
        if max_length <= 0:
            return 0

        group_block_sizes = self.group_block_sizes or {
            group_id: self.hash_block_size for group_id in self.group_hashes
        }
        group_hits = [
            self._longest_group_match(
                block_hashes=block_hashes,
                hashes=self.group_hashes.get(group_id, set()),
                block_size=block_size,
                max_cache_hit_length=max_length,
            )
            for group_id, block_size in group_block_sizes.items()
        ]
        return min(group_hits, default=0)

    def _longest_group_match(
        self,
        block_hashes: Sequence[BlockHash],
        hashes: set[PrefixBlockHash],
        block_size: int,
        max_cache_hit_length: int,
    ) -> int:
        if block_size <= 0 or block_size % self.hash_block_size != 0:
            return 0

        scale = block_size // self.hash_block_size
        max_blocks = min(
            max_cache_hit_length // block_size,
            len(block_hashes) // scale,
        )

        matched_blocks = 0
        for block_idx in range(max_blocks):
            # Request block hashes form a prefix chain. The final hash-block
            # boundary inside this group block already fingerprints the full
            # prefix covered by the group block.
            hash_idx = (block_idx + 1) * scale - 1
            cache_key = maybe_convert_block_hash(block_hashes[hash_idx])
            if cache_key not in hashes:
                break
            matched_blocks += 1
        return matched_blocks * block_size


class GlobalPrefixScheduler:
    """In-memory longest-prefix-first router for vLLM nodes."""

    def __init__(
        self,
        *,
        view_ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        strict_cache_generation: bool = True,
    ) -> None:
        if view_ttl_seconds is not None and view_ttl_seconds <= 0:
            raise ValueError("view_ttl_seconds must be positive")
        self._view_ttl_seconds = view_ttl_seconds
        self._clock = clock
        self._strict_cache_generation = strict_cache_generation
        self._nodes: dict[tuple[str, int | None], NodePrefixCacheState] = {}
        self._node_defaults: dict[str, tuple[int, int | None, dict[int, int]]] = {}
        self._tie_breaker = count()

    def register_node(
        self,
        node_id: str,
        *,
        hash_block_size: int,
        data_parallel_rank: int | None = None,
        group_block_sizes: Mapping[int, int] | None = None,
    ) -> NodePrefixCacheState:
        group_block_sizes = dict(group_block_sizes or {})
        state = NodePrefixCacheState(
            node_id=node_id,
            hash_block_size=hash_block_size,
            data_parallel_rank=data_parallel_rank,
            group_block_sizes=group_block_sizes,
        )
        self._node_defaults[node_id] = (
            hash_block_size,
            data_parallel_rank,
            group_block_sizes,
        )
        self._nodes[node_id, data_parallel_rank] = state
        return state

    def update_snapshot(
        self,
        snapshot: PrefixCacheSnapshot,
        *,
        worker_incarnation: str | None = None,
        now: float | None = None,
        cache_generation: int | None = None,
    ) -> None:
        defaults = self._node_defaults.get(snapshot.node_id)
        if defaults is not None:
            _, configured_rank, _ = defaults
            if (
                snapshot.data_parallel_rank is not None
                and configured_rank is not None
                and snapshot.data_parallel_rank != configured_rank
            ):
                raise ValueError(
                    f"node {snapshot.node_id!r} is configured for data-parallel "
                    f"rank {configured_rank}, but reported rank "
                    f"{snapshot.data_parallel_rank}"
                )
            if snapshot.data_parallel_rank is None and configured_rank is not None:
                snapshot = replace(snapshot, data_parallel_rank=configured_rank)
        self._node_defaults.setdefault(
            snapshot.node_id,
            (
                snapshot.hash_block_size,
                None,
                dict(snapshot.group_block_sizes),
            ),
        )
        key = (snapshot.node_id, snapshot.data_parallel_rank)
        strict_generation = (
            cache_generation if self._strict_cache_generation else None
        )
        state = self._nodes.get(key)
        if state is None:
            self._discard_unranked_placeholder(snapshot.node_id)
            state = NodePrefixCacheState.from_snapshot(
                snapshot,
                worker_incarnation=worker_incarnation,
            )
            state.record_worker_receipt(
                now=self._resolve_now(now),
                view_ttl_seconds=self._view_ttl_seconds,
                worker_incarnation=worker_incarnation,
                cache_generation=strict_generation,
            )
            if strict_generation is None:
                state.view_epoch += 1
            self._nodes[key] = state
        else:
            state.apply_snapshot(
                snapshot,
                now=self._resolve_now(now),
                view_ttl_seconds=self._view_ttl_seconds,
                worker_incarnation=worker_incarnation,
                cache_generation=strict_generation,
            )

    def apply_events(
        self,
        node_id: str,
        events: Iterable[KVCacheEvent],
        data_parallel_rank: int | None = None,
        *,
        worker_incarnation: str | None = None,
        now: float | None = None,
        cache_generation: int | None = None,
        is_snapshot: bool = False,
    ) -> None:
        self._state_for_events(node_id, data_parallel_rank).apply_events(
            events,
            now=self._resolve_now(now),
            view_ttl_seconds=self._view_ttl_seconds,
            worker_incarnation=worker_incarnation,
            cache_generation=(
                cache_generation if self._strict_cache_generation else None
            ),
            is_snapshot=is_snapshot,
        )

    def apply_event_batch(
        self,
        node_id: str,
        batch: KVEventBatch,
        *,
        worker_incarnation: str | None = None,
        now: float | None = None,
    ) -> None:
        self.apply_events(
            node_id,
            batch.events,
            batch.data_parallel_rank,
            worker_incarnation=worker_incarnation,
            now=now,
            cache_generation=batch.cache_generation,
            is_snapshot=batch.is_snapshot,
        )

    def record_worker_receipt(
        self,
        node_id: str,
        data_parallel_rank: int | None = None,
        *,
        worker_incarnation: str | None = None,
        now: float | None = None,
        cache_generation: int | None = None,
    ) -> None:
        self._state_for_events(node_id, data_parallel_rank).record_worker_receipt(
            now=self._resolve_now(now),
            view_ttl_seconds=self._view_ttl_seconds,
            worker_incarnation=worker_incarnation,
            cache_generation=(
                cache_generation if self._strict_cache_generation else None
            ),
        )

    def invalidate_node(
        self,
        node_id: str,
        data_parallel_rank: int | None = None,
        *,
        worker_incarnation: str | None = None,
        now: float | None = None,
        cache_generation: int | None = None,
    ) -> None:
        self._state_for_events(node_id, data_parallel_rank).invalidate(
            worker_incarnation=worker_incarnation,
            now=self._resolve_now(now),
            view_ttl_seconds=self._view_ttl_seconds,
            cache_generation=(
                cache_generation if self._strict_cache_generation else None
            ),
        )

    def remove_node(self, node_id: str) -> None:
        self._node_defaults.pop(node_id, None)
        for key in [key for key in self._nodes if key[0] == node_id]:
            del self._nodes[key]

    def _state_for_events(
        self, node_id: str, data_parallel_rank: int | None
    ) -> NodePrefixCacheState:
        try:
            hash_block_size, configured_rank, group_block_sizes = self._node_defaults[
                node_id
            ]
        except KeyError as exc:
            raise KeyError(f"unknown prefix routing node {node_id!r}") from exc

        if data_parallel_rank is None:
            data_parallel_rank = configured_rank
        elif configured_rank is not None and data_parallel_rank != configured_rank:
            raise ValueError(
                f"node {node_id!r} is configured for data-parallel rank "
                f"{configured_rank}, but reported rank {data_parallel_rank}"
            )

        key = (node_id, data_parallel_rank)
        state = self._nodes.get(key)
        if state is None:
            self._discard_unranked_placeholder(node_id)
            state = NodePrefixCacheState(
                node_id=node_id,
                hash_block_size=hash_block_size,
                data_parallel_rank=data_parallel_rank,
                group_block_sizes=dict(group_block_sizes),
            )
            self._nodes[key] = state
        return state

    def _discard_unranked_placeholder(self, node_id: str) -> None:
        placeholder = self._nodes.get((node_id, None))
        if placeholder is not None and not placeholder.group_hashes:
            del self._nodes[node_id, None]

    def choose_node(
        self,
        block_hashes: Sequence[BlockHash],
        prompt_num_tokens: int,
        *,
        candidate_node_ids: Iterable[str] | None = None,
        max_cache_hit_length: int | None = None,
    ) -> PrefixRouteDecision | None:
        """Choose the node with the longest prefix-cache hit.

        Ties are broken round-robin so equally good nodes still receive traffic.
        """
        node_ids = list(candidate_node_ids) if candidate_node_ids is not None else None
        states = (
            [
                state
                for (node_id, _), state in self._nodes.items()
                if node_id in node_ids
            ]
            if node_ids is not None
            else list(self._nodes.values())
        )
        if not states:
            return None

        now = self._clock()
        states = [
            state
            for state in states
            if not state.expire_if_stale(now)
            and (self._view_ttl_seconds is None or state.last_receipt_at is not None)
        ]
        if not states:
            return None

        scored: list[tuple[int, NodePrefixCacheState]] = []
        for state in states:
            matched_tokens = state.longest_prefix_match(
                block_hashes=block_hashes,
                prompt_num_tokens=prompt_num_tokens,
                max_cache_hit_length=max_cache_hit_length,
            )
            scored.append((matched_tokens, state))

        best_match = max(matched_tokens for matched_tokens, _ in scored)
        tied_states = [
            state for matched_tokens, state in scored if matched_tokens == best_match
        ]
        state = tied_states[next(self._tie_breaker) % len(tied_states)]
        return PrefixRouteDecision(
            node_id=state.node_id,
            data_parallel_rank=state.data_parallel_rank,
            matched_tokens=best_match,
            view_epoch=state.view_epoch,
            worker_incarnation=state.worker_incarnation,
        )

    def is_decision_current(self, decision: PrefixRouteDecision) -> bool:
        state = self._nodes.get((decision.node_id, decision.data_parallel_rank))
        if state is None:
            return False
        if state.expire_if_stale(self._clock()):
            return False
        return (
            state.view_epoch == decision.view_epoch
            and state.worker_incarnation == decision.worker_incarnation
        )

    def get_node_route_generation(
        self, node_id: str, data_parallel_rank: int | None = None
    ) -> tuple[int, str | None] | None:
        state = self._nodes.get((node_id, data_parallel_rank))
        if state is None and data_parallel_rank is None:
            defaults = self._node_defaults.get(node_id)
            if defaults is not None:
                _, configured_rank, _ = defaults
                state = self._nodes.get((node_id, configured_rank))
        if state is None:
            return None
        if state.expire_if_stale(self._clock()):
            return None
        return state.view_epoch, state.worker_incarnation

    def _resolve_now(self, now: float | None) -> float:
        return self._clock() if now is None else now
