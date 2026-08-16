# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler receipt boundary for owner-local prefix hits."""

from types import SimpleNamespace

import pytest

from tests.v1.core.test_scheduler_owner_admission import (
    _apply_pool_receipts,
    _apply_receipts,
    _make_scheduler,
    _pool,
    _request,
)
from vllm.v1.core.request_owned_prefix import OwnerPrefixDescriptor
from vllm.v1.core.sched.ownership import OwnerReceipt
from vllm.v1.core.sched.request_owned_prefix_directory import (
    RequestOwnedPrefixDirectory,
    RequestOwnedPrefixScheduler,
)
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import PrefixCacheStats


def _prefix_scheduler(world_size: int = 1):
    scheduler = _make_scheduler(
        world_size=world_size, max_num_scheduled_tokens=8
    )
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=True)
    scheduler.block_size = 4
    scheduler.hash_block_size = 4
    scheduler._owner_prefix = RequestOwnedPrefixScheduler(
        True, world_size, 4, 4
    )
    return scheduler


def test_reserve_carries_hashes_and_exact_hit_advances_logical_progress():
    scheduler = _prefix_scheduler()
    request = _request("shared", num_prompt_tokens=8)
    request.block_hashes = [b"prefix-0", b"prefix-1"]
    scheduler.add_request(request)

    output = scheduler.schedule()
    (command,) = output.owner_commands
    assert command.allocation is not None
    assert command.allocation.prefix == OwnerPrefixDescriptor(
        (b"prefix-0", b"prefix-1")
    )

    _apply_receipts(
        scheduler,
        output,
        {
            0: [
                OwnerReceipt(
                    key=command.key,
                    owner_id=0,
                    command_seq=command.command_seq,
                    accepted=True,
                    runnable_num_tokens=command.required_num_tokens,
                    prefix_cache_hit_tokens=4,
                )
            ]
        },
    )
    assert request.num_computed_tokens == 4

    dispatch = scheduler.schedule()
    assert dispatch.num_scheduled_tokens == {request.request_id: 4}
    (new_data,) = dispatch.scheduled_new_reqs
    assert new_data.num_computed_tokens == 4
    assert new_data.prompt_token_ids == list(range(8))


def test_reserve_excludes_generated_suffix_hashes_from_prefix_wire():
    scheduler = _prefix_scheduler()
    request = _request("resumed", num_prompt_tokens=8)
    request.block_hashes = [b"prompt-0", b"prompt-1", b"generated-suffix"]
    scheduler.add_request(request)

    output = scheduler.schedule()
    (command,) = output.owner_commands

    assert command.allocation is not None
    assert command.allocation.prefix == OwnerPrefixDescriptor(
        (b"prompt-0", b"prompt-1")
    )


def test_prefix_enabled_reserve_missing_physical_hit_fails_before_mutation():
    scheduler = _prefix_scheduler()
    request = _request("missing", num_prompt_tokens=8)
    request.block_hashes = [b"prefix-0", b"prefix-1"]
    scheduler.add_request(request)
    output = scheduler.schedule()
    (command,) = output.owner_commands

    with pytest.raises(RuntimeError, match="missing its exact owner-local prefix hit"):
        _apply_receipts(
            scheduler,
            output,
            {
                0: [
                    OwnerReceipt(
                        key=command.key,
                        owner_id=0,
                        command_seq=command.command_seq,
                        accepted=True,
                        runnable_num_tokens=command.required_num_tokens,
                    )
                ]
            },
        )
    assert request.num_computed_tokens == 0


def test_prefix_affinity_spills_after_one_fresh_request_of_load_skew():
    directory = RequestOwnedPrefixDirectory(4, 4, 4)
    hashes = [b"prefix-0", b"prefix-1"]
    directory.observe_computed_prefix(0, hashes, 8, 8)

    local = directory.select_with_bounded_affinity(
        hashes,
        8,
        projected_free={owner: 100 for owner in range(4)},
        fresh_demand={owner: 10 for owner in range(4)},
        live_leases={owner: 0 for owner in range(4)},
    )
    assert local is not None
    assert (local.owner_id, local.matched_tokens) == (0, 4)

    # More than one conservative fresh-request footprint behind the least
    # loaded rank: locality loses and the cold rank becomes a future replica.
    spill = directory.select_with_bounded_affinity(
        hashes,
        8,
        projected_free={0: 79, 1: 100, 2: 100, 3: 100},
        fresh_demand={owner: 10 for owner in range(4)},
        live_leases={owner: 0 for owner in range(4)},
    )
    assert spill is not None
    assert (spill.owner_id, spill.matched_tokens) == (1, 0)


def test_prefix_directory_reset_drops_affinity_without_physical_authority():
    directory = RequestOwnedPrefixDirectory(2, 4, 4)
    hashes = [b"prefix-0", b"prefix-1"]
    directory.observe_computed_prefix(0, hashes, 8, 8)
    assert directory.longest_match(0, hashes, 8) == 4

    directory.reset()

    assert directory.longest_match(0, hashes, 8) == 0


def test_prefix_directory_is_bounded_and_retains_long_chained_boundaries():
    directory = RequestOwnedPrefixDirectory(
        1, 4, 4, max_hashes_per_owner=2
    )
    hashes = [f"prefix-{index}".encode() for index in range(4)]
    directory.observe_computed_prefix(0, hashes, 16, 16)

    assert len(directory._hashes_by_owner[0]) == 2
    # Boundary three remains sufficient because every block hash commits the
    # full preceding chain; the last prompt block is held back for logits.
    assert directory.longest_match(0, hashes, 16) == 12
    assert directory.longest_match(0, hashes, 8) == 0


def test_exact_owner_hit_uses_standard_prefix_cache_metrics():
    request = _request("metrics", num_prompt_tokens=8)
    request.block_hashes = [b"prefix-0", b"prefix-1"]
    stats = PrefixCacheStats()
    adapter = RequestOwnedPrefixScheduler(True, 1, 4, 4)

    adapter.record_lookup(request, 4, stats)

    assert (stats.requests, stats.queries, stats.hits) == (1, 8, 4)
    assert (stats.block_queries, stats.block_hits) == (2, 1)


def test_terminal_observation_keeps_prefix_of_request_finishing_same_step():
    request = _request("one-step", num_prompt_tokens=8, max_tokens=1)
    request.block_hashes = [b"prefix-0", b"prefix-1"]
    request.attention_owner = 0
    request.num_computed_tokens = 8
    adapter = RequestOwnedPrefixScheduler(True, 2, 4, 4)

    adapter.observe_scheduled({request.request_id: request}, [request.request_id])

    assert adapter.directory is not None
    assert adapter.directory.longest_match(0, request.block_hashes, 8) == 4


@pytest.mark.parametrize(("worker_reset", "expected"), [(True, True), (False, False)])
def test_engine_reset_composes_scheduler_and_owner_physical_cache(
    worker_reset, expected
):
    calls: list[str] = []
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_request_owned_attention=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    engine.scheduler = SimpleNamespace(
        _reset_prefix_cache_local=lambda *args: calls.append("scheduler") or True
    )
    engine.model_executor = SimpleNamespace(
        reset_request_owned_prefix_cache=lambda: calls.append("worker")
        or worker_reset
    )

    assert engine.reset_prefix_cache() is expected
    assert calls == ["scheduler", "worker"]


def test_force_reset_fails_closed_until_owner_controls_drain():
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_request_owned_attention=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    engine.scheduler = SimpleNamespace(
        _reset_prefix_cache_local=lambda *args: True
    )
    engine.model_executor = SimpleNamespace(
        reset_request_owned_prefix_cache=lambda: False
    )

    with pytest.raises(RuntimeError, match="Drain owner control/release"):
        engine.reset_prefix_cache(reset_running_requests=True)


def test_direct_scheduler_reset_fails_closed_for_owner_physical_cache():
    scheduler = _prefix_scheduler()

    with pytest.raises(RuntimeError, match="through EngineCore"):
        scheduler.reset_prefix_cache()


@pytest.mark.parametrize(
    ("free_by_owner", "expected_owner"),
    [({0: 100, 1: 100}, 0), ({0: 79, 1: 100}, 1)],
)
def test_scheduler_admission_uses_bounded_prefix_affinity(
    free_by_owner, expected_owner
):
    scheduler = _prefix_scheduler(world_size=2)
    empty = scheduler.schedule()
    _apply_pool_receipts(
        scheduler,
        empty,
        {
            owner: _pool(
                owner,
                free_blocks=free,
                effective_tokens_per_block=(4,),
            )
            for owner, free in free_by_owner.items()
        },
    )
    hashes = [b"prefix-0", b"prefix-1"]
    assert scheduler._owner_prefix.directory is not None
    scheduler._owner_prefix.directory.observe_computed_prefix(
        0, hashes, 8, 8
    )
    request = _request("shared", num_prompt_tokens=8)
    request.block_hashes = hashes
    scheduler.add_request(request)

    output = scheduler.schedule()
    (command,) = output.owner_commands
    assert command.owner_id == expected_owner
