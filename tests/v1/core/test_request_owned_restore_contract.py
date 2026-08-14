# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace

import pytest

from vllm.v1.core.sched.restore_contract import (
    RESTORE_SCHEDULER_WIRE_FORBIDDEN_FIELDS,
    RestoreCertificate,
    RestoreCertificateStatus,
    RestoreDeadlineGroup,
    RestoreDemandJobReceipt,
    RestoreDemandReceipt,
    RestoreIntent,
    RestorePhase,
    aggregate_restore_demand,
    canonical_json_bytes,
)


def _intent(**changes) -> RestoreIntent:
    base = RestoreIntent(
        request_uid="req-7",
        owner_rank=2,
        owner_epoch=3,
        activation_generation=4,
        phase=RestorePhase.DECODE_RESUME,
        required_token_extent=512,
        valid_prefix_token_extent=384,
        first_consume_step=17,
        max_wait_steps=2,
        urgency_class="landing",
        policy_reason="preempt-return",
    )
    return replace(base, **changes)


def _certificate(**changes) -> RestoreCertificate:
    base = RestoreCertificate(
        request_uid="req-7",
        owner_rank=2,
        owner_epoch=3,
        activation_generation=4,
        required_blocks=3,
        reserved_blocks=4,
        restoring_blocks=0,
        hot_blocks=3,
        landing_hot_watermark=1,
        tail_hot_watermark=3,
        scheduled_bytes=96,
        completed_bytes=96,
        deadline_miss_count=0,
        status=RestoreCertificateStatus.HOT,
    )
    return replace(base, **changes)


def _demand(
    request_uid: str,
    *,
    owner_rank: int,
    generation: int,
    phase: RestorePhase,
    wave_id: str,
    blocks: int,
) -> RestoreDemandReceipt:
    jobs = (
        (
            RestoreDemandJobReceipt(
                group_index=0,
                deadline_group=RestoreDeadlineGroup.LANDING,
                effective_tokens_per_block=128,
                valid_token_extents=(128,) * blocks,
                blocks=blocks,
                scheduled_bytes=blocks * 11,
                completed_bytes=blocks * 11,
                scheduled_step=10,
                completed_step=11,
            ),
        )
        if blocks
        else ()
    )
    return RestoreDemandReceipt(
        request_uid=request_uid,
        owner_rank=owner_rank,
        owner_epoch=1,
        activation_generation=generation,
        phase=phase,
        wave_id=wave_id,
        source_provenance="core@abc",
        workload_provenance="synthetic-a0",
        required_blocks=max(1, blocks),
        resident_blocks=max(1, blocks) - blocks,
        host_only_blocks=blocks,
        restoring_blocks=0,
        newly_restored_blocks=blocks,
        logical_128_token_units_proxy=99,
        final_footprint_reserved_blocks=max(1, blocks),
        jobs=jobs,
        wait_steps=1,
        deadline_miss_reason=None,
        terminal_status=RestoreCertificateStatus.HOT,
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_restore_intent_wire_is_explicitly_block_id_and_geometry_free():
    wire = _intent().to_wire_dict()
    assert not (set(_all_keys(wire)) & RESTORE_SCHEDULER_WIRE_FORBIDDEN_FIELDS)
    assert wire["phase"] == "decode-resume"
    assert wire["schema"] == "request-owned-restore-intent/v1"

    demand = _demand(
        "req-7",
        owner_rank=2,
        generation=4,
        phase=RestorePhase.DECODE_RESUME,
        wave_id="wave-1",
        blocks=1,
    )
    for public_wire in (
        _certificate().to_wire_dict(),
        json.loads(demand.canonical_bytes()),
    ):
        assert not (
            set(_all_keys(public_wire)) & RESTORE_SCHEDULER_WIRE_FORBIDDEN_FIELDS
        )


def test_restore_intent_rejects_extent_regression_and_bool_counts():
    with pytest.raises(ValueError, match="must not exceed"):
        _intent(required_token_extent=10, valid_prefix_token_extent=11)
    with pytest.raises(TypeError, match="non-bool"):
        _intent(max_wait_steps=True)


def test_hot_certificate_is_the_only_dispatch_authority():
    intent = _intent()
    certificate = _certificate()
    assert certificate.certifies(intent)
    assert not replace(
        certificate,
        activation_generation=5,
    ).certifies(intent)
    assert not RestoreCertificate(
        request_uid="req-7",
        owner_rank=2,
        owner_epoch=3,
        activation_generation=4,
        required_blocks=3,
        reserved_blocks=4,
        restoring_blocks=3,
        hot_blocks=0,
        landing_hot_watermark=0,
        tail_hot_watermark=0,
        scheduled_bytes=96,
        completed_bytes=0,
        deadline_miss_count=0,
        status=RestoreCertificateStatus.RESTORING,
    ).certifies(intent)


def test_certificate_rejects_inferred_hot_and_invalid_failure_payload():
    with pytest.raises(ValueError, match="every required block"):
        _certificate(hot_blocks=2, tail_hot_watermark=2)
    with pytest.raises(ValueError, match="exact byte completion"):
        _certificate(completed_bytes=95)
    with pytest.raises(TypeError, match="failure_reason"):
        _certificate(
            status=RestoreCertificateStatus.FAILED,
            restoring_blocks=0,
            hot_blocks=0,
            landing_hot_watermark=0,
            tail_hot_watermark=0,
            completed_bytes=0,
        )


def test_demand_aggregation_preserves_zero_activations_and_wave_rank_totals():
    receipts = (
        _demand(
            "zero",
            owner_rank=0,
            generation=1,
            phase=RestorePhase.PREFILL,
            wave_id="a",
            blocks=0,
        ),
        _demand(
            "one",
            owner_rank=0,
            generation=1,
            phase=RestorePhase.PREFILL,
            wave_id="a",
            blocks=2,
        ),
        _demand(
            "two",
            owner_rank=1,
            generation=1,
            phase=RestorePhase.PREFILL,
            wave_id="a",
            blocks=4,
        ),
    )
    aggregate = aggregate_restore_demand(reversed(receipts))
    distribution = aggregate["activation_distributions"][0]["blocks"]
    assert distribution == {
        "count": 3,
        "maximum": 4,
        "p50": 2,
        "p90": 4,
        "p95": 4,
        "p99": 4,
        "total": 6,
        "zero_count": 1,
    }
    assert aggregate["wave_rank_rows"] == [
        {
            "deadline_miss_count": 0,
            "newly_restored_blocks": 2,
            "owner_rank": 0,
            "phase": "prefill",
            "scheduled_bytes": 22,
            "wave_id": "a",
        },
        {
            "deadline_miss_count": 0,
            "newly_restored_blocks": 4,
            "owner_rank": 1,
            "phase": "prefill",
            "scheduled_bytes": 44,
            "wave_id": "a",
        },
    ]
    assert canonical_json_bytes(aggregate) == canonical_json_bytes(
        aggregate_restore_demand(receipts)
    )


def test_demand_receipt_is_byte_identical_except_explicit_time_fields():
    base = _demand(
        "req",
        owner_rank=0,
        generation=1,
        phase=RestorePhase.PREFILL,
        wave_id="w",
        blocks=0,
    )
    later = replace(base, observed_start_ns=100, observed_end_ns=200)
    assert base.canonical_bytes() == later.canonical_bytes()
    assert base.canonical_bytes(include_timing=True) != later.canonical_bytes(
        include_timing=True
    )


def test_hot_demand_requires_exact_per_job_completion():
    demand = _demand(
        "req",
        owner_rank=0,
        generation=1,
        phase=RestorePhase.PREFILL,
        wave_id="w",
        blocks=1,
    )
    incomplete = replace(
        demand.jobs[0],
        completed_bytes=0,
        completed_step=None,
    )
    with pytest.raises(ValueError, match="exact completion"):
        replace(demand, jobs=(incomplete,))


def test_duplicate_demand_activation_fails_closed():
    demand = _demand(
        "req",
        owner_rank=0,
        generation=1,
        phase=RestorePhase.PREFILL,
        wave_id="w",
        blocks=0,
    )
    with pytest.raises(ValueError, match="duplicate activation"):
        aggregate_restore_demand((demand, demand))
