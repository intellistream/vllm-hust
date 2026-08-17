# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact observation and non-acting policy for owner-local prefixes."""

from tests.v1.core.test_scheduler_owner_admission import _request
from vllm.v1.core.sched.ownership import (
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerReceipt,
)
from vllm.v1.core.sched.request_owned_prefix_directory import (
    RequestOwnedPrefixScheduler,
)
from vllm.v1.core.sched.request_owned_prefix_observability import (
    OwnerPrefixAdmissionObservation,
    OwnerPrefixCandidateObservation,
    OwnerPrefixPublicationObservation,
    OwnerPrefixReserveObservation,
    RequestOwnedPrefixReplicaShadow,
)


def _candidate(
    owner_id: int,
    projected_free: int,
    *,
    hinted_tokens: int = 0,
    live_leases: int = 0,
) -> OwnerPrefixCandidateObservation:
    return OwnerPrefixCandidateObservation(
        owner_id=owner_id,
        projected_free=projected_free,
        fresh_demand=10,
        live_leases=live_leases,
        hinted_tokens=hinted_tokens,
        affinity_eligible=True,
    )


def _receipt(
    request_id: str,
    owner_id: int,
    *,
    accepted: bool = True,
    hit_tokens: int | None = 0,
) -> OwnerReceipt:
    return OwnerReceipt(
        key=OwnerLeaseKey(request_id, 0),
        owner_id=owner_id,
        command_seq=1,
        accepted=accepted,
        runnable_num_tokens=8 if accepted else None,
        prefix_cache_hit_tokens=hit_tokens if accepted else None,
    )


def test_observer_distinguishes_affinity_hit_and_exact_worker_hit() -> None:
    observations = []
    adapter = RequestOwnedPrefixScheduler(
        True,
        2,
        4,
        4,
        observation_sink=observations.append,
    )
    assert adapter.directory is not None
    hashes = [b"prefix-0", b"prefix-1"]
    adapter.directory.observe_computed_prefix(0, hashes, 8, 8)
    request = _request("warm", num_prompt_tokens=8)
    request.block_hashes = hashes

    selected = adapter.select_owner(
        request,
        projected_free={0: 100, 1: 100},
        fresh_demand={0: 10, 1: 10},
        live_leases={0: 0, 1: 0},
    )
    hit = adapter.validate_reserve_receipt(
        _receipt(request.request_id, 0, hit_tokens=4),
        (1, OwnerCommandKind.RESERVE),
        request,
    )

    assert selected == 0
    assert hit == 4
    admission, reserve = observations
    assert isinstance(admission, OwnerPrefixAdmissionObservation)
    assert admission.reason == "affinity_hit"
    assert admission.selected_hinted_tokens == 4
    assert admission.prefix_digest is not None
    assert [(item.owner_id, item.hinted_tokens) for item in admission.candidates] == [
        (0, 4),
        (1, 0),
    ]
    assert isinstance(reserve, OwnerPrefixReserveObservation)
    assert reserve.outcome == "exact_hit"
    assert reserve.exact_hit_tokens == 4
    assert reserve.prefix_digest == admission.prefix_digest


def test_observer_labels_bounded_spill_without_claiming_a_hit() -> None:
    observations = []
    adapter = RequestOwnedPrefixScheduler(
        True,
        2,
        4,
        4,
        observation_sink=observations.append,
    )
    assert adapter.directory is not None
    hashes = [b"prefix-0", b"prefix-1"]
    adapter.directory.observe_computed_prefix(0, hashes, 8, 8)
    request = _request("spill", num_prompt_tokens=8)
    request.block_hashes = hashes

    selected = adapter.select_owner(
        request,
        projected_free={0: 79, 1: 100},
        fresh_demand={0: 10, 1: 10},
        live_leases={0: 0, 1: 0},
    )
    hit = adapter.validate_reserve_receipt(
        _receipt(request.request_id, 1),
        (1, OwnerCommandKind.RESERVE),
        request,
    )

    assert selected == 1
    assert hit == 0
    admission, reserve = observations
    assert isinstance(admission, OwnerPrefixAdmissionObservation)
    assert admission.reason == "bounded_spill"
    assert admission.selected_hinted_tokens == 0
    assert [item.affinity_eligible for item in admission.candidates] == [
        False,
        True,
    ]
    assert isinstance(reserve, OwnerPrefixReserveObservation)
    assert reserve.outcome == "cold_miss"
    assert reserve.exact_hit_tokens == 0

    request.attention_owner = 1
    request.num_computed_tokens = 4
    adapter.observe_scheduled({request.request_id: request}, [request.request_id])
    publication = observations[-1]
    assert isinstance(publication, OwnerPrefixPublicationObservation)
    assert publication.owner_id == 1
    assert publication.directory_hint_was_new is True


def test_observer_separates_stale_hint_miss_from_later_publication() -> None:
    observations = []
    adapter = RequestOwnedPrefixScheduler(
        True,
        1,
        4,
        4,
        observation_sink=observations.append,
    )
    assert adapter.directory is not None
    hashes = [b"prefix-0", b"prefix-1"]
    adapter.directory.observe_computed_prefix(0, hashes, 8, 8)
    request = _request("stale", num_prompt_tokens=8)
    request.block_hashes = hashes
    adapter.select_owner(
        request,
        projected_free={0: 100},
        fresh_demand={0: 10},
        live_leases={0: 0},
    )
    adapter.validate_reserve_receipt(
        _receipt(request.request_id, 0),
        (1, OwnerCommandKind.RESERVE),
        request,
    )

    reserve = observations[-1]
    assert isinstance(reserve, OwnerPrefixReserveObservation)
    assert reserve.outcome == "stale_hint_miss"

    request.attention_owner = 0
    request.num_computed_tokens = 4
    adapter.observe_scheduled({request.request_id: request}, [request.request_id])
    publication = observations[-1]
    assert isinstance(publication, OwnerPrefixPublicationObservation)
    assert publication.previous_tokens == 0
    assert publication.published_tokens == 4
    assert publication.exact_reserve_hit_tokens == 0
    assert publication.directory_hint_was_new is False

    num_observations = len(observations)
    adapter.observe_scheduled({request.request_id: request}, [request.request_id])
    assert len(observations) == num_observations


def test_rejected_reserve_is_observed_without_physical_claim() -> None:
    observations = []
    adapter = RequestOwnedPrefixScheduler(
        True,
        1,
        4,
        4,
        observation_sink=observations.append,
    )
    request = _request("rejected", num_prompt_tokens=8)
    request.block_hashes = [b"prefix-0", b"prefix-1"]
    adapter.select_owner(
        request,
        projected_free={0: 100},
        fresh_demand={0: 10},
        live_leases={0: 0},
    )

    assert (
        adapter.validate_reserve_receipt(
            _receipt(request.request_id, 0, accepted=False),
            (1, OwnerCommandKind.RESERVE),
            request,
        )
        is None
    )
    reserve = observations[-1]
    assert isinstance(reserve, OwnerPrefixReserveObservation)
    assert reserve.outcome == "rejected_reserve"
    assert reserve.accepted is False
    assert reserve.exact_hit_tokens is None


def test_disabled_adapter_emits_no_observation_or_prefix_descriptor() -> None:
    observations = []
    adapter = RequestOwnedPrefixScheduler(
        False,
        2,
        4,
        4,
        observation_sink=observations.append,
    )
    request = _request("disabled", num_prompt_tokens=8)
    request.block_hashes = [b"prefix-0", b"prefix-1"]

    assert adapter.descriptor(request) is None
    assert (
        adapter.select_owner(
            request,
            projected_free={0: 100, 1: 100},
            fresh_demand={0: 10, 1: 10},
            live_leases={0: 0, 1: 0},
        )
        is None
    )
    assert observations == []


def test_shadow_expands_by_seat_demand_with_deterministic_target() -> None:
    shadow = RequestOwnedPrefixReplicaShadow(world_size=8, rows_per_owner=4)
    candidates = (
        _candidate(0, 80, hinted_tokens=32),
        _candidate(1, 100, live_leases=1),
        _candidate(2, 100),
    )

    decision = shadow.evaluate(
        predicted_runnable_seats=9,
        hinted_owners=frozenset({0, 1}),
        candidates=candidates,
        required_free_blocks=20,
        lead_steps=2,
        materialization_steps=2,
        execution_slack=True,
    )

    assert decision.action == "expand"
    assert decision.reason == "expand_with_slack"
    assert decision.current_replicas == 2
    assert decision.desired_replicas == 3
    assert decision.target_owner == 2


def test_shadow_refuses_unearned_materialization_and_keeps_sufficient_set() -> None:
    shadow = RequestOwnedPrefixReplicaShadow(world_size=8, rows_per_owner=4)
    candidates = (_candidate(0, 80), _candidate(1, 100))
    common = {
        "predicted_runnable_seats": 5,
        "hinted_owners": frozenset({0}),
        "candidates": candidates,
        "required_free_blocks": 20,
        "lead_steps": 2,
        "materialization_steps": 2,
    }

    no_slack = shadow.evaluate(**common, execution_slack=False)
    assert (no_slack.action, no_slack.reason) == (
        "refuse",
        "insufficient_execution_slack",
    )

    too_late = shadow.evaluate(**(common | {"lead_steps": 1}), execution_slack=True)
    assert (too_late.action, too_late.reason) == (
        "refuse",
        "insufficient_lead",
    )

    no_capacity = shadow.evaluate(
        **(common | {"required_free_blocks": 101}), execution_slack=True
    )
    assert (no_capacity.action, no_capacity.reason) == (
        "refuse",
        "insufficient_capacity",
    )

    enough = shadow.evaluate(
        **(common | {"hinted_owners": frozenset({0, 1})}),
        execution_slack=False,
    )
    assert (enough.action, enough.reason, enough.desired_replicas) == (
        "keep",
        "replica_set_sufficient",
        2,
    )
