"""Body block registry for segment reuse.

The BodyBlockRegistry maintains a mapping from body hashes to pinned
KV block IDs. When a request's body is registered (seed), the blocks
are pinned and protected from eviction by the normal block pool LRU.
Subsequent requests with the same body hash can borrow those blocks
instead of recomputing the body's KV cache.

This registry operates alongside vLLM's BlockPool:
- BlockPool manages the standard prefix-cache block lifecycle
- BodyBlockRegistry pins specific blocks for body reuse
- When a body is no longer needed (all borrowers released), the
  blocks are unpinned and returned to the BlockPool's free list
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from vllm.v1.segment_reuse.types import BodyBlockEntry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BodyBlockRegistry:
    """Thread-safe registry mapping body hashes to pinned KV blocks.

    The registry is instantiated once per scheduler and lives for the
    lifetime of the serving process. All operations are thread-safe
    because the scheduler may process requests concurrently in the
    v1 async architecture.

    Attributes:
        _entries: Mapping from body_hash to BodyBlockEntry.
        _lock: Protects concurrent access to the registry.
    """

    def __init__(self) -> None:
        self._entries: dict[bytes, BodyBlockEntry] = {}
        self._lock = threading.Lock()
        self._total_lookups = 0
        self._total_hits = 0
        self._total_registrations = 0

    def register(
        self,
        body_hash: bytes,
        block_ids: list[int],
        token_count: int,
        block_size: int,
        model_key: str = "default",
    ) -> BodyBlockEntry:
        """Register a body's KV blocks in the registry.

        Called after a seed request completes its prefill. The blocks
        are "pinned" — they will not be evicted by the normal block
        pool until explicitly unregistered.

        Args:
            body_hash: SHA-256 hash of the body token sequence.
            block_ids: Physical KV block IDs to pin.
            token_count: Number of tokens the body occupies.
            block_size: The KV cache block size in tokens.
            model_key: Model identifier for cross-model isolation.

        Returns:
            The newly created BodyBlockEntry.

        Raises:
            ValueError: If body_hash is already registered.
        """
        with self._lock:
            if body_hash in self._entries:
                raise ValueError(
                    f"Body hash {body_hash.hex()[:16]} already registered"
                )
            entry = BodyBlockEntry(
                body_hash=body_hash,
                block_ids=list(block_ids),
                token_count=token_count,
                block_size=block_size,
                ref_count=0,
                model_key=model_key,
                created_at=time.monotonic(),
            )
            self._entries[body_hash] = entry
            self._total_registrations += 1
            logger.info(
                "segment_reuse: registered body hash=%s blocks=%d tokens=%d",
                body_hash.hex()[:16],
                len(block_ids),
                token_count,
            )
            return entry

    def lookup(self, body_hash: bytes) -> BodyBlockEntry | None:
        """Look up a body by its hash.

        Args:
            body_hash: SHA-256 hash to look up.

        Returns:
            The BodyBlockEntry if found, None otherwise.
        """
        with self._lock:
            self._total_lookups += 1
            entry = self._entries.get(body_hash)
            if entry is not None:
                self._total_hits += 1
            return entry

    def acquire(self, body_hash: bytes) -> bool:
        """Acquire a borrower reference on a body.

        Called when a stitch request starts using a body's KV blocks.
        Increments the ref_count.

        Args:
            body_hash: Hash of the body to acquire.

        Returns:
            True if the body exists and was acquired, False otherwise.
        """
        with self._lock:
            entry = self._entries.get(body_hash)
            if entry is None:
                return False
            entry.ref_count += 1
            return True

    def release(self, body_hash: bytes) -> None:
        """Release a borrower reference on a body.

        Called when a stitch request finishes or is preempted.
        Decrements the ref_count.

        Args:
            body_hash: Hash of the body to release.
        """
        with self._lock:
            entry = self._entries.get(body_hash)
            if entry is not None:
                entry.ref_count = max(0, entry.ref_count - 1)

    def unregister(self, body_hash: bytes) -> list[int]:
        """Remove a body from the registry and return its block IDs.

        Called when a body should be evicted (e.g., memory pressure).
        The returned block IDs should be returned to the BlockPool.

        Args:
            body_hash: Hash of the body to unregister.

        Returns:
            List of block IDs that were pinned for this body.

        Raises:
            KeyError: If body_hash is not registered.
            RuntimeError: If the body has active borrowers.
        """
        with self._lock:
            if body_hash not in self._entries:
                raise KeyError(f"Body hash {body_hash.hex()[:16]} not registered")
            entry = self._entries[body_hash]
            if entry.ref_count > 0:
                raise RuntimeError(
                    f"Cannot unregister body {body_hash.hex()[:16]}: "
                    f"{entry.ref_count} active borrowers"
                )
            del self._entries[body_hash]
            logger.info(
                "segment_reuse: unregistered body hash=%s blocks=%d",
                body_hash.hex()[:16],
                len(entry.block_ids),
            )
            return entry.block_ids

    def stats(self) -> dict[str, int | float]:
        """Return registry statistics.

        Returns:
            Dictionary with current registry state and counters.
        """
        with self._lock:
            total_pinned_blocks = sum(
                len(e.block_ids) for e in self._entries.values()
            )
            active_borrowers = sum(
                e.ref_count for e in self._entries.values()
            )
            return {
                "num_registered_bodies": len(self._entries),
                "total_pinned_blocks": total_pinned_blocks,
                "active_borrowers": active_borrowers,
                "total_lookups": self._total_lookups,
                "total_hits": self._total_hits,
                "total_registrations": self._total_registrations,
                "hit_rate": (
                    self._total_hits / self._total_lookups
                    if self._total_lookups > 0
                    else 0.0
                ),
            }

    def clear(self) -> list[int]:
        """Clear all entries and return all pinned block IDs.

        Used during shutdown or full reset.

        Returns:
            All block IDs that were pinned.
        """
        with self._lock:
            all_block_ids: list[int] = []
            for entry in self._entries.values():
                all_block_ids.extend(entry.block_ids)
            self._entries.clear()
            return all_block_ids

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, body_hash: bytes) -> bool:
        with self._lock:
            return body_hash in self._entries
