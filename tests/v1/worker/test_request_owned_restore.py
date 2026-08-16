# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.v1.core.sched.ownership import OwnerLeaseKey
from vllm.v1.core.sched.restore_contract import (
    RestoreCertificateStatus,
    RestoreDeadlineGroup,
    RestoreIntent,
    RestorePhase,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.request_owned_kv import RequestOwnedKVSnapshot
from vllm.v1.worker.request_owned_restore import (
    CanonicalRestoreSlice,
    CanonicalRestoreSpan,
    PackedRestoreGeometry,
    RequestOwnedRestoreError,
    RestoreDemandUnknownError,
    RestoreGroupGeometry,
    build_group_full_page_restore_plan,
    demand_receipt_for_current_plan,
    hot_restore_certificate,
    terminal_restore_certificate,
)


def _span(offset: int, size: int, alias: str) -> CanonicalRestoreSpan:
    item = CanonicalRestoreSlice(offset, size, (alias,))
    return CanonicalRestoreSpan(offset, size, (alias,), (item,))


def _geometry() -> PackedRestoreGeometry:
    return PackedRestoreGeometry(
        block_size_tokens=128,
        block_stride_bytes=64,
        runtime_num_blocks=16,
        groups=(
            RestoreGroupGeometry(0, 8, _span(0, 40, "layer.a")),
            RestoreGroupGeometry(1, 8, _span(40, 24, "layer.b")),
        ),
    )


def _intent(**changes) -> RestoreIntent:
    base = RestoreIntent(
        request_uid="req",
        owner_rank=1,
        owner_epoch=2,
        activation_generation=3,
        phase=RestorePhase.DECODE_RESUME,
        required_token_extent=128,
        valid_prefix_token_extent=12,
        first_consume_step=9,
        max_wait_steps=1,
        urgency_class="landing",
        policy_reason="preempt-return",
    )
    return replace(base, **changes)


def _snapshot(
    tables: tuple[tuple[int, ...], ...] = ((1, 2), (3, 4)),
) -> RequestOwnedKVSnapshot:
    return RequestOwnedKVSnapshot(
        key=OwnerLeaseKey("req", 2),
        owner_rank=1,
        allocation_generation=7,
        num_computed_tokens=12,
        reserved_num_tokens=128,
        pending_free=False,
        tables=tables,
    )


def _plan(**changes):
    kwargs = {
        "intent": _intent(),
        "destination": _snapshot(),
        "geometry": _geometry(),
        "plan_seq": 5,
        "actual_restore_block_ids": ((1,), (3, 4)),
        "valid_token_extents": ((8,), (8, 4)),
        "deadline_groups": (
            RestoreDeadlineGroup.LANDING,
            RestoreDeadlineGroup.TAIL,
        ),
        "reserved_final_footprint_blocks": 4,
    }
    kwargs.update(changes)
    return build_group_full_page_restore_plan(**kwargs)


def _attention(page_head_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=page_head_size,
        dtype=torch.float16,
    )


def test_live_packed_geometry_canonicalizes_alias_slices_exactly_once():
    a = _attention(4)  # 16 B page
    b = _attention(8)  # 32 B page
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(8 * 48, ["a"], offset=0, block_stride=48),
            KVCacheTensor(8 * 48, ["b", "c"], offset=16, block_stride=48),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["a", "b"],
                UniformTypeKVCacheSpecs(
                    block_size=1,
                    kv_cache_specs={"a": a, "b": b},
                ),
            ),
            KVCacheGroupSpec(
                ["c"],
                UniformTypeKVCacheSpecs(
                    block_size=1,
                    kv_cache_specs={"c": b},
                ),
            ),
        ],
    )
    geometry = PackedRestoreGeometry.from_kv_cache_config(config, block_size_tokens=128)
    first, second = geometry.groups
    assert geometry.block_stride_bytes == 48
    assert (first.canonical_span.offset_bytes, first.canonical_span.page_bytes) == (
        0,
        48,
    )
    assert [item.aliases for item in first.canonical_span.slices] == [
        ("a",),
        ("b",),
    ]
    assert (second.canonical_span.offset_bytes, second.canonical_span.page_bytes) == (
        16,
        32,
    )
    assert second.canonical_span.slices[0].aliases == ("c",)
    assert (
        geometry.fingerprint
        == PackedRestoreGeometry.from_kv_cache_config(
            config, block_size_tokens=128
        ).fingerprint
    )


def test_uniform_mixed_compression_geometry_uses_allocation_binding_extent():
    high = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        compress_ratio=128,
    )
    low = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        compress_ratio=4,
    )
    high_page = high.page_size_bytes
    low_page = low.page_size_bytes
    stride = high_page + low_page
    num_blocks = 8
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                num_blocks * stride,
                ["high"],
                offset=0,
                block_stride=stride,
            ),
            KVCacheTensor(
                num_blocks * stride,
                ["low"],
                offset=high_page,
                block_stride=stride,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["high", "low"],
                UniformTypeKVCacheSpecs(
                    block_size=128,
                    kv_cache_specs={"high": high, "low": low},
                ),
            )
        ],
    )

    geometry = PackedRestoreGeometry.from_kv_cache_config(
        config,
        block_size_tokens=128,
    )

    assert geometry.groups[0].effective_tokens_per_block == 128 * 4
    assert geometry.groups[0].canonical_span.page_bytes == stride


def test_canonical_span_rejects_unpriced_gaps():
    with pytest.raises(ValueError, match="exactly tile"):
        CanonicalRestoreSpan(
            offset_bytes=0,
            page_bytes=24,
            aliases=("a", "b"),
            slices=(
                CanonicalRestoreSlice(0, 8, ("a",)),
                CanonicalRestoreSlice(16, 8, ("b",)),
            ),
        )


def test_live_geometry_rejects_duplicate_physical_descriptors():
    a = _attention(4)
    b = _attention(4)
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(8 * 16, ["a"], offset=0, block_stride=16),
            KVCacheTensor(8 * 16, ["b"], offset=0, block_stride=16),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], a),
            KVCacheGroupSpec(["b"], b),
        ],
    )
    with pytest.raises(RequestOwnedRestoreError, match="non-overlapping"):
        PackedRestoreGeometry.from_kv_cache_config(config, block_size_tokens=128)


def test_plan_uses_group_canonical_bytes_not_packed_stride():
    plan = _plan()
    assert plan.total_destination_ids == 3
    assert plan.expected_bytes == 40 + 2 * 24
    assert plan.expected_bytes != 3 * plan.block_stride_bytes
    assert plan.landing_required_blocks == 1
    assert plan.tail_required_blocks == 2
    assert plan.jobs[1].valid_token_extents == (8, 4)


def test_cross_group_local_ids_must_be_disjoint():
    with pytest.raises(RequestOwnedRestoreError, match="disjoint across groups"):
        _plan(
            destination=_snapshot(((1, 2), (1, 4))),
            actual_restore_block_ids=((1,), (1,)),
            valid_token_extents=((8,), (8,)),
        )


def test_unknown_actual_demand_never_falls_back_to_logical_scale_or_stride():
    with pytest.raises(RestoreDemandUnknownError, match="not substitutes"):
        _plan(actual_restore_block_ids=None)
    with pytest.raises(RestoreDemandUnknownError, match="not substitutes"):
        _plan(valid_token_extents=None)


def test_zero_demand_is_an_explicit_valid_plan_and_hot_certificate():
    plan = _plan(
        actual_restore_block_ids=((), ()),
        valid_token_extents=((), ()),
    )
    assert plan.jobs == ()
    assert plan.expected_bytes == 0
    certificate = hot_restore_certificate(plan, required_blocks=2)
    assert certificate.status is RestoreCertificateStatus.HOT
    assert certificate.scheduled_bytes == certificate.completed_bytes == 0
    assert certificate.hot_blocks == 2


@pytest.mark.parametrize(
    ("intent", "allocation_generation", "plan_seq", "geometry"),
    [
        (_intent(owner_rank=0), 7, 5, _geometry()),
        (_intent(owner_epoch=3), 7, 5, _geometry()),
        (_intent(activation_generation=4), 7, 5, _geometry()),
        (_intent(), 8, 5, _geometry()),
        (_intent(), 7, 6, _geometry()),
        (
            _intent(),
            7,
            5,
            replace(_geometry(), block_stride_bytes=65),
        ),
    ],
)
def test_stale_owner_epoch_generation_plan_or_geometry_fails_closed(
    intent, allocation_generation, plan_seq, geometry
):
    with pytest.raises(RequestOwnedRestoreError, match="stale restore"):
        _plan().assert_current(
            intent=intent,
            allocation_generation=allocation_generation,
            plan_seq=plan_seq,
            geometry=geometry,
        )


def test_plan_jobs_are_structurally_bound_to_live_geometry():
    plan = _plan()
    forged_job = replace(
        plan.jobs[0],
        canonical_span=_span(1, 40, "layer.a"),
    )
    forged = replace(plan, jobs=(forged_job, *plan.jobs[1:]))
    with pytest.raises(RequestOwnedRestoreError, match="stale restore"):
        forged.assert_current(
            intent=_intent(),
            allocation_generation=7,
            plan_seq=5,
            geometry=_geometry(),
        )


def test_first_real_correctness_scope_is_bounded_not_a_capacity_claim():
    _plan().assert_bounded_correctness_scope(max_ids=4)
    with pytest.raises(RequestOwnedRestoreError, match="bounded 2-ID"):
        _plan().assert_bounded_correctness_scope(max_ids=2)
    with pytest.raises(RequestOwnedRestoreError, match="bounded 4-ID"):
        _plan(
            intent=_intent(valid_prefix_token_extent=24),
            destination=replace(
                _snapshot(((1, 2, 3), (4, 5))),
                num_computed_tokens=24,
            ),
            actual_restore_block_ids=((1, 2, 3), (4, 5)),
            valid_token_extents=((8, 8, 8), (8, 8)),
            reserved_final_footprint_blocks=5,
        )


def test_plan_rejects_destination_outside_owner_table_and_short_reservation():
    with pytest.raises(RequestOwnedRestoreError, match="non-destination"):
        _plan(actual_restore_block_ids=((9,), (3,)))
    with pytest.raises(ValueError, match="final footprint"):
        _plan(reserved_final_footprint_blocks=2)
    with pytest.raises(RequestOwnedRestoreError, match="valid token extents"):
        _plan(valid_token_extents=((8,), (8, 8)))


def test_demand_receipt_is_deterministic_and_preserves_partial_tail():
    plan = _plan()
    receipt = demand_receipt_for_current_plan(
        intent=_intent(),
        plan=plan,
        geometry=_geometry(),
        current_allocation_generation=7,
        current_plan_seq=5,
        wave_id="decode-9",
        source_provenance="core@abc+ascend@def",
        workload_provenance="synthetic-a0",
        required_blocks=4,
        resident_blocks=1,
        host_only_blocks=3,
        restoring_blocks=0,
        logical_128_token_units_proxy=1,
        scheduled_step=7,
        completed_step=8,
        terminal_status=RestoreCertificateStatus.HOT,
        observed_start_ns=100,
        observed_end_ns=200,
    )
    replay = demand_receipt_for_current_plan(
        intent=_intent(),
        plan=plan,
        geometry=_geometry(),
        current_allocation_generation=7,
        current_plan_seq=5,
        wave_id="decode-9",
        source_provenance="core@abc+ascend@def",
        workload_provenance="synthetic-a0",
        required_blocks=4,
        resident_blocks=1,
        host_only_blocks=3,
        restoring_blocks=0,
        logical_128_token_units_proxy=1,
        scheduled_step=7,
        completed_step=8,
        terminal_status=RestoreCertificateStatus.HOT,
        observed_start_ns=300,
        observed_end_ns=400,
    )
    assert receipt.canonical_bytes() == replay.canonical_bytes()
    assert receipt.jobs[1].blocks == 2
    assert receipt.jobs[1].effective_tokens_per_block == 8
    assert receipt.jobs[1].valid_token_extents == (8, 4)
    assert receipt.scheduled_bytes == plan.expected_bytes


def test_demand_receipt_rechecks_live_allocation_and_plan_fences():
    with pytest.raises(RequestOwnedRestoreError, match="stale restore"):
        demand_receipt_for_current_plan(
            intent=_intent(),
            plan=_plan(),
            geometry=_geometry(),
            current_allocation_generation=8,
            current_plan_seq=5,
            wave_id="decode-9",
            source_provenance="core@abc",
            workload_provenance="synthetic-a0",
            required_blocks=4,
            resident_blocks=1,
            host_only_blocks=3,
            restoring_blocks=0,
            logical_128_token_units_proxy=1,
            scheduled_step=7,
            completed_step=8,
            terminal_status=RestoreCertificateStatus.HOT,
        )


def test_failed_abort_and_release_cleanup_drop_hot_authority():
    failed = terminal_restore_certificate(
        _intent(),
        status=RestoreCertificateStatus.FAILED,
        failure_reason="abort",
    )
    released = terminal_restore_certificate(
        _intent(), status=RestoreCertificateStatus.RELEASED
    )
    assert not failed.certifies(_intent())
    assert not released.certifies(_intent())
    assert failed.failure_reason == "abort"
