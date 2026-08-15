# SPDX-License-Identifier: Apache-2.0
"""Host-testable lifecycle controller for a pinned output arena.

This module deliberately has no torch dependency. Device-specific storage and
events are attached by the model runner after this state machine passes its host
correctness gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Lock


class ArenaLifecycleError(RuntimeError):
    """Raised when a lease violates slot ownership."""


class ArenaConsumer(str, Enum):
    OUTPUT = "output"
    INPUT_BATCH = "input_batch"


class SlotState(str, Enum):
    FREE = "free"
    COPYING = "copying"
    READY = "ready"
    LEASED = "leased"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class ArenaLease:
    arena_id: str
    slot_id: int
    generation: int
    shape: tuple[int, ...]
    dtype: str
    request_fingerprint: str
    consumers: frozenset[ArenaConsumer]


@dataclass
class ArenaCounters:
    eligible_outputs: int = 0
    arena_activations: int = 0
    arena_hits: int = 0
    arena_misses: int = 0
    native_fallbacks: int = 0
    stale_generations: int = 0
    request_mismatches: int = 0
    double_consumes: int = 0
    cancel_cleanups: int = 0
    exception_cleanups: int = 0
    quarantined_slots: int = 0


@dataclass
class _ArenaSlot:
    slot_id: int
    generation: int = 0
    state: SlotState = SlotState.FREE
    shape: tuple[int, ...] = ()
    dtype: str = ""
    request_fingerprint: str = ""
    pending_consumers: set[ArenaConsumer] = field(default_factory=set)


class PinnedOutputArena:
    """Own slot generations and fail closed on lifecycle violations."""

    def __init__(
        self,
        *,
        arena_id: str,
        capacity: int = 2,
        enabled: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.arena_id = arena_id
        self.enabled = enabled
        self._slots = [_ArenaSlot(slot_id=i) for i in range(capacity)]
        self._counters = ArenaCounters()
        self._lock = Lock()

    def acquire(
        self,
        *,
        shape: tuple[int, ...],
        dtype: str,
        request_fingerprint: str,
        consumers: frozenset[ArenaConsumer],
        eligible: bool = True,
    ) -> ArenaLease | None:
        """Acquire a free generation, or return ``None`` for native fallback."""
        if not consumers or ArenaConsumer.OUTPUT not in consumers:
            raise ValueError("every output lease requires the output consumer")
        with self._lock:
            if eligible:
                self._counters.eligible_outputs += 1
            if not self.enabled or not eligible:
                return None
            slot = next(
                (slot for slot in self._slots if slot.state is SlotState.FREE),
                None,
            )
            if slot is None:
                self._counters.arena_misses += 1
                return None
            slot.generation += 1
            slot.state = SlotState.COPYING
            slot.shape = shape
            slot.dtype = dtype
            slot.request_fingerprint = request_fingerprint
            slot.pending_consumers = set(consumers)
            self._counters.arena_activations += 1
            self._counters.arena_hits += 1
            return ArenaLease(
                arena_id=self.arena_id,
                slot_id=slot.slot_id,
                generation=slot.generation,
                shape=shape,
                dtype=dtype,
                request_fingerprint=request_fingerprint,
                consumers=consumers,
            )

    def mark_ready(self, lease: ArenaLease) -> None:
        with self._lock:
            slot = self._validated_slot(lease)
            if slot.state is not SlotState.COPYING:
                self._double_consume(slot, "copy completion recorded twice")
            slot.state = SlotState.READY

    def consume(
        self,
        lease: ArenaLease,
        consumer: ArenaConsumer,
        *,
        request_fingerprint: str | None = None,
    ) -> None:
        with self._lock:
            slot = self._validated_slot(lease)
            if request_fingerprint is not None and (
                request_fingerprint != slot.request_fingerprint
            ):
                self._counters.request_mismatches += 1
                self._quarantine(slot)
                raise ArenaLifecycleError("request fingerprint mismatch")
            if slot.state not in (SlotState.READY, SlotState.LEASED):
                self._double_consume(slot, "consumer ran before copy-ready")
            if consumer not in slot.pending_consumers:
                self._double_consume(slot, f"duplicate consumer: {consumer.value}")
            slot.pending_consumers.remove(consumer)
            if slot.pending_consumers:
                slot.state = SlotState.LEASED
            else:
                self._release(slot)

    def cancel(self, lease: ArenaLease, *, copy_complete: bool) -> None:
        with self._lock:
            self._counters.cancel_cleanups += 1
            slot = self._validated_slot(lease)
            if slot.state is SlotState.FREE:
                self._double_consume(slot, "lease cancelled after release")
            if copy_complete:
                self._release(slot)
            else:
                self._quarantine(slot)

    def fail(self, lease: ArenaLease, *, copy_complete: bool) -> None:
        with self._lock:
            self._counters.exception_cleanups += 1
            slot = self._validated_slot(lease)
            if slot.state is SlotState.FREE:
                self._double_consume(slot, "lease failed after release")
            if copy_complete:
                self._release(slot)
            else:
                self._quarantine(slot)

    def record_native_fallback(self) -> None:
        with self._lock:
            self._counters.native_fallbacks += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "arena_id": self.arena_id,
                "enabled": self.enabled,
                "counters": asdict(self._counters),
                "slots": [
                    {
                        "slot_id": slot.slot_id,
                        "generation": slot.generation,
                        "state": slot.state.value,
                        "pending_consumers": sorted(
                            consumer.value for consumer in slot.pending_consumers
                        ),
                    }
                    for slot in self._slots
                ],
            }

    def _validated_slot(self, lease: ArenaLease) -> _ArenaSlot:
        if lease.arena_id != self.arena_id:
            self.enabled = False
            self._counters.stale_generations += 1
            raise ArenaLifecycleError("arena identity mismatch")
        if lease.slot_id < 0 or lease.slot_id >= len(self._slots):
            self.enabled = False
            self._counters.stale_generations += 1
            raise ArenaLifecycleError("slot identity out of range")
        slot = self._slots[lease.slot_id]
        if slot.generation != lease.generation:
            self._counters.stale_generations += 1
            self._quarantine(slot)
            raise ArenaLifecycleError("stale slot generation")
        # A matching generation in FREE state is a duplicate cleanup/consume.
        # Its shape metadata has already been cleared by the first release, so
        # let the caller classify it as a double operation.
        if slot.state is SlotState.FREE:
            return slot
        if slot.state is SlotState.QUARANTINED:
            self.enabled = False
            self._counters.stale_generations += 1
            raise ArenaLifecycleError("slot is quarantined")
        if slot.shape != lease.shape or slot.dtype != lease.dtype:
            self._counters.stale_generations += 1
            self._quarantine(slot)
            raise ArenaLifecycleError("stale slot generation")
        return slot

    def _double_consume(self, slot: _ArenaSlot, message: str) -> None:
        self._counters.double_consumes += 1
        self._quarantine(slot)
        raise ArenaLifecycleError(message)

    def _quarantine(self, slot: _ArenaSlot) -> None:
        if slot.state is not SlotState.QUARANTINED:
            self._counters.quarantined_slots += 1
        slot.state = SlotState.QUARANTINED
        slot.pending_consumers.clear()
        self.enabled = False

    @staticmethod
    def _release(slot: _ArenaSlot) -> None:
        slot.state = SlotState.FREE
        slot.shape = ()
        slot.dtype = ""
        slot.request_fingerprint = ""
        slot.pending_consumers.clear()
