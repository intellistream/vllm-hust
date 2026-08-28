# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# Load the lifecycle helper without importing vllm.__init__. This is part of
# the host gate: the tests must not require torch, CUDA, or an NPU runtime.
MODULE_PATH = (
    Path(__file__).parents[4]
    / "vllm"
    / "v1"
    / "worker"
    / "gpu"
    / "pinned_output_arena.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_pinned_output_arena_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
ARENA_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ARENA_MODULE
SPEC.loader.exec_module(ARENA_MODULE)

ArenaConsumer = ARENA_MODULE.ArenaConsumer
ArenaLease = ARENA_MODULE.ArenaLease
ArenaLifecycleError = ARENA_MODULE.ArenaLifecycleError
PinnedOutputArena = ARENA_MODULE.PinnedOutputArena

OUTPUT_ONLY = frozenset({ArenaConsumer.OUTPUT})
TWO_CONSUMERS = frozenset({ArenaConsumer.OUTPUT, ArenaConsumer.INPUT_BATCH})


def acquire(
    arena: Any,
    fingerprint: str,
    consumers: frozenset[Any] = OUTPUT_ONLY,
) -> Any:
    lease = arena.acquire(
        shape=(4, 1),
        dtype="int64",
        request_fingerprint=fingerprint,
        consumers=consumers,
    )
    assert lease is not None
    return lease


def counters(arena: Any) -> dict[str, int]:
    value = arena.snapshot()["counters"]
    assert isinstance(value, dict)
    return value


def test_default_off_preserves_native_fallback() -> None:
    arena = PinnedOutputArena(arena_id="test", enabled=False)

    assert (
        arena.acquire(
            shape=(1, 1),
            dtype="int64",
            request_fingerprint="req-0",
            consumers=OUTPUT_ONLY,
        )
        is None
    )
    arena.record_native_fallback()

    assert counters(arena) == {
        "eligible_outputs": 1,
        "arena_activations": 0,
        "arena_hits": 0,
        "arena_misses": 0,
        "native_fallbacks": 1,
        "stale_generations": 0,
        "request_mismatches": 0,
        "double_consumes": 0,
        "cancel_cleanups": 0,
        "exception_cleanups": 0,
        "quarantined_slots": 0,
    }


def test_two_slots_are_not_reacquired_while_live() -> None:
    arena = PinnedOutputArena(arena_id="test", enabled=True)
    first = acquire(arena, "req-0")
    second = acquire(arena, "req-1")

    assert first.slot_id != second.slot_id
    assert (
        arena.acquire(
            shape=(4, 1),
            dtype="int64",
            request_fingerprint="req-2",
            consumers=OUTPUT_ONLY,
        )
        is None
    )
    arena.record_native_fallback()
    assert counters(arena)["arena_misses"] == 1


@pytest.mark.parametrize(
    "first_consumer",
    [ArenaConsumer.OUTPUT, ArenaConsumer.INPUT_BATCH],
)
def test_slot_waits_for_both_consumers(first_consumer: Any) -> None:
    arena = PinnedOutputArena(arena_id="test", capacity=1, enabled=True)
    lease = acquire(arena, "req-0", TWO_CONSUMERS)
    arena.mark_ready(lease)
    arena.consume(lease, first_consumer, request_fingerprint="req-0")

    assert arena.snapshot()["slots"][0]["state"] == "leased"
    assert (
        arena.acquire(
            shape=(4, 1),
            dtype="int64",
            request_fingerprint="req-1",
            consumers=OUTPUT_ONLY,
        )
        is None
    )

    second_consumer = (
        ArenaConsumer.INPUT_BATCH
        if first_consumer is ArenaConsumer.OUTPUT
        else ArenaConsumer.OUTPUT
    )
    arena.consume(lease, second_consumer, request_fingerprint="req-0")
    next_lease = acquire(arena, "req-1")
    assert next_lease.generation == lease.generation + 1


def test_double_consume_quarantines_and_disables_arena() -> None:
    arena = PinnedOutputArena(arena_id="test", capacity=1, enabled=True)
    lease = acquire(arena, "req-0", TWO_CONSUMERS)
    arena.mark_ready(lease)
    arena.consume(lease, ArenaConsumer.OUTPUT)

    with pytest.raises(ArenaLifecycleError, match="duplicate consumer"):
        arena.consume(lease, ArenaConsumer.OUTPUT)

    assert arena.snapshot()["enabled"] is False
    assert counters(arena)["double_consumes"] == 1
    assert counters(arena)["quarantined_slots"] == 1


def test_old_generation_cannot_release_reused_slot() -> None:
    arena = PinnedOutputArena(arena_id="test", capacity=1, enabled=True)
    old = acquire(arena, "req-0")
    arena.mark_ready(old)
    arena.consume(old, ArenaConsumer.OUTPUT)
    current = acquire(arena, "req-1")

    with pytest.raises(ArenaLifecycleError, match="stale slot generation"):
        arena.consume(old, ArenaConsumer.OUTPUT)

    assert current.generation == old.generation + 1
    assert arena.snapshot()["slots"][0]["state"] == "quarantined"
    assert counters(arena)["stale_generations"] == 1


def test_request_mismatch_never_consumes_slot() -> None:
    arena = PinnedOutputArena(arena_id="test", enabled=True)
    lease = acquire(arena, "req-0")
    arena.mark_ready(lease)

    with pytest.raises(ArenaLifecycleError, match="fingerprint mismatch"):
        arena.consume(
            lease,
            ArenaConsumer.OUTPUT,
            request_fingerprint="different-request",
        )

    assert counters(arena)["request_mismatches"] == 1
    assert arena.snapshot()["enabled"] is False


def test_duplicate_cancel_is_fail_closed() -> None:
    arena = PinnedOutputArena(arena_id="test", capacity=1, enabled=True)
    lease = acquire(arena, "req-0")
    arena.cancel(lease, copy_complete=True)

    with pytest.raises(ArenaLifecycleError, match="cancelled after release"):
        arena.cancel(lease, copy_complete=True)

    assert counters(arena)["double_consumes"] == 1
    assert arena.snapshot()["enabled"] is False


@pytest.mark.parametrize("method_name", ["cancel", "fail"])
@pytest.mark.parametrize("copy_complete", [True, False])
def test_cancel_and_exception_cleanup(method_name: str, copy_complete: bool) -> None:
    arena = PinnedOutputArena(arena_id="test", capacity=1, enabled=True)
    lease = acquire(arena, "req-0")

    getattr(arena, method_name)(lease, copy_complete=copy_complete)

    state = arena.snapshot()["slots"][0]["state"]
    assert state == ("free" if copy_complete else "quarantined")
    counter = "cancel_cleanups" if method_name == "cancel" else "exception_cleanups"
    assert counters(arena)[counter] == 1


def test_ineligible_valid_count_is_not_reported_as_miss() -> None:
    arena = PinnedOutputArena(arena_id="test", enabled=True)

    assert (
        arena.acquire(
            shape=(4,),
            dtype="int64",
            request_fingerprint="valid-count-inactive",
            consumers=OUTPUT_ONLY,
            eligible=False,
        )
        is None
    )
    assert counters(arena)["eligible_outputs"] == 0
    assert counters(arena)["arena_misses"] == 0
