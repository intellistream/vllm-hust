# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def test_restore_gate_records_computed_tokens_reduced_before_prefill_planning():
    request = SimpleNamespace(
        request_id="restore-gate",
        num_tokens=64,
        kv_restore_computed_tokens_reduced=0,
        kv_restore_local_computed_tokens=0,
        kv_restore_fallback_reason=None,
    )
    computed_tokens_reduced: dict[str, int] = {}
    restore_fallback_reasons: dict[str, str] = {}

    Scheduler._record_restore_planning_gate(
        request,
        num_external_computed_tokens=32,
        computed_tokens_reduced=computed_tokens_reduced,
        restore_fallback_reasons=restore_fallback_reasons,
    )

    num_computed_tokens = request.kv_restore_computed_tokens_reduced
    num_new_tokens = request.num_tokens - num_computed_tokens

    assert num_new_tokens == 32
    assert request.kv_restore_computed_tokens_reduced == 32
    assert request.kv_restore_fallback_reason is None
    # Async restores emit this field only when the request is actually
    # scheduled; this helper first records the scheduler-side registration.
    assert computed_tokens_reduced == {}
    assert restore_fallback_reasons == {}


def test_restore_gate_emits_reduction_only_after_scheduled_forward():
    request = SimpleNamespace(
        request_id="restore-forward",
        num_tokens=64,
        kv_restore_computed_tokens_reduced=0,
        kv_restore_local_computed_tokens=0,
        kv_restore_fallback_reason=None,
    )
    computed_tokens_reduced: dict[str, int] = {}
    restore_fallback_reasons: dict[str, str] = {}

    Scheduler._record_restore_planning_gate(
        request,
        num_external_computed_tokens=32,
        computed_tokens_reduced=computed_tokens_reduced,
        restore_fallback_reasons=restore_fallback_reasons,
    )

    num_computed_tokens = request.kv_restore_computed_tokens_reduced
    num_new_tokens = request.num_tokens - num_computed_tokens
    assert num_new_tokens == 32
    assert computed_tokens_reduced == {}

    Scheduler._record_restore_scheduled_forward(
        request,
        computed_tokens_reduced=computed_tokens_reduced,
        restore_fallback_reasons=restore_fallback_reasons,
    )

    assert computed_tokens_reduced == {
        request.request_id: 32,
    }
    assert restore_fallback_reasons == {}


def test_restore_gate_records_fallback_reason_when_registration_fails():
    request = SimpleNamespace(
        request_id="restore-fallback",
        num_tokens=64,
        kv_restore_computed_tokens_reduced=16,
        kv_restore_local_computed_tokens=0,
        kv_restore_fallback_reason=None,
    )
    computed_tokens_reduced: dict[str, int] = {}
    restore_fallback_reasons: dict[str, str] = {}

    Scheduler._record_restore_planning_gate(
        request,
        num_external_computed_tokens=None,
        computed_tokens_reduced=computed_tokens_reduced,
        restore_fallback_reasons=restore_fallback_reasons,
        fallback_reason="connector_match_unavailable",
    )

    assert request.kv_restore_computed_tokens_reduced == 0
    assert request.kv_restore_fallback_reason == "connector_match_unavailable"
    assert computed_tokens_reduced == {}
    assert restore_fallback_reasons == {
        request.request_id: "connector_match_unavailable",
    }


def test_restore_gate_emits_fallback_reason_without_reduction():
    request = SimpleNamespace(
        request_id="restore-forward-fallback",
        num_tokens=64,
        kv_restore_computed_tokens_reduced=0,
        kv_restore_local_computed_tokens=0,
        kv_restore_fallback_reason="connector_match_unavailable",
    )
    computed_tokens_reduced: dict[str, int] = {}
    restore_fallback_reasons: dict[str, str] = {}

    Scheduler._record_restore_scheduled_forward(
        request,
        computed_tokens_reduced=computed_tokens_reduced,
        restore_fallback_reasons=restore_fallback_reasons,
    )

    assert computed_tokens_reduced == {}
    assert restore_fallback_reasons == {
        request.request_id: "connector_match_unavailable",
    }


def test_restore_gate_reads_connector_specific_fallback_reason():
    request = SimpleNamespace(request_id="restore-connector-fallback")
    connector = SimpleNamespace(
        get_restore_fallback_reason=lambda req: (
            "prefix_chain_incomplete"
            if req.request_id == request.request_id
            else None
        )
    )

    assert (
        Scheduler._get_connector_restore_fallback_reason(connector, request)
        == "prefix_chain_incomplete"
    )


def test_restore_gate_requires_explicit_connector_candidate():
    request = SimpleNamespace(request_id="restore-candidate")
    ordinary_connector = SimpleNamespace()
    tiered_connector = SimpleNamespace(
        has_restore_candidate=lambda req: req.request_id == request.request_id
    )

    assert not Scheduler._connector_has_restore_candidate(
        ordinary_connector, request
    )
    assert Scheduler._connector_has_restore_candidate(tiered_connector, request)


def test_complete_restore_load_failure_revokes_avoided_tokens():
    request = SimpleNamespace(
        num_computed_tokens=16,
        kv_restore_local_computed_tokens=16,
        kv_restore_computed_tokens_reduced=32,
        kv_restore_fallback_reason=None,
    )

    Scheduler._record_restore_load_failure(request)

    assert request.kv_restore_computed_tokens_reduced == 0
    assert request.kv_restore_fallback_reason == "restore_load_failure"


def test_partial_restore_load_failure_keeps_only_valid_prefix_credit():
    request = SimpleNamespace(
        num_computed_tokens=32,
        kv_restore_local_computed_tokens=16,
        kv_restore_computed_tokens_reduced=32,
        kv_restore_fallback_reason=None,
    )

    Scheduler._record_restore_load_failure(request)

    assert request.kv_restore_computed_tokens_reduced == 16
    assert request.kv_restore_fallback_reason == "restore_load_partial_failure"
