# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU tests for the G0 request-owner protocol reference state machine.

These tests exercise :class:`OwnerLeaseCoordinator` and
:class:`AttentionLeaseManager` together (scheduler side + worker side) plus
the backward-compatible owner/epoch carriers on Request, NewRequestData, and
SchedulerOutput.  No GPU model runner is constructed.
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    EpochFenceError,
    OwnerAssignmentObservation,
    OwnerCommandKind,
    OwnerLeaseCoordinator,
    OwnerLeaseKey,
    OwnerReceipt,
    OwnershipError,
    PublicationViolationError,
)
from vllm.v1.request import Request, RequestStatus


def _key(request_id: str = "req-0", epoch: int = 0) -> OwnerLeaseKey:
    return OwnerLeaseKey(request_id=request_id, owner_epoch=epoch)


def _grant(
    coordinator: OwnerLeaseCoordinator, manager: AttentionLeaseManager, command
) -> OwnerReceipt:
    receipt = manager.apply(command)
    assert coordinator.apply_receipt(receipt)
    return receipt


def _publish(
    coordinator: OwnerLeaseCoordinator, manager: AttentionLeaseManager, step_seq: int
):
    tokens = coordinator.publish(step_seq)
    for token in tokens:
        manager.record_published(token)
    return tokens


def _coordinator_with_owners(*owners: int) -> OwnerLeaseCoordinator:
    coordinator = OwnerLeaseCoordinator()
    for seq, owner in enumerate(owners):
        coordinator.observe(
            OwnerAssignmentObservation(owner_id=owner, observation_seq=seq)
        )
    return coordinator


# -- assignment ---------------------------------------------------------------


def test_least_committed_work_assignment_is_deterministic() -> None:
    coordinator = OwnerLeaseCoordinator()
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=2, observation_seq=1, work=50)
    )
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=5, observation_seq=1, work=30)
    )
    assert coordinator.assign(_key()) == 5
    # Repeated assignment with identical inputs is stable.
    assert coordinator.assign(_key()) == 5
    # Pending DMA counts toward committed work: 5 + 20 = 25 < 30.
    coordinator.observe(
        OwnerAssignmentObservation(
            owner_id=3, observation_seq=2, work=5, pending_dma=20
        )
    )
    assert coordinator.assign(_key("req-1")) == 3
    assert coordinator.assign(_key("req-2")) == 3


def test_tie_break_uses_stable_numeric_global_rank() -> None:
    coordinator = OwnerLeaseCoordinator()
    for owner in (7, 3, 5):
        coordinator.observe(
            OwnerAssignmentObservation(owner_id=owner, observation_seq=1, work=10)
        )
    assert coordinator.assign(_key()) == 3
    # Residency is the secondary tie-break before global rank.
    coordinator = OwnerLeaseCoordinator()
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=4, observation_seq=1, work=10)
    )
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=2, observation_seq=1, work=10, residency=8)
    )
    assert coordinator.assign(_key()) == 2


def test_assignment_is_idempotent_per_key() -> None:
    coordinator = _coordinator_with_owners(1, 2)
    key = _key()
    assert coordinator.assign(key) == 1
    # Later observations change the ranking, but the same key keeps its owner.
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=2, observation_seq=2, work=0)
    )
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=2, work=1000)
    )
    assert coordinator.assign(key) == 1
    assert coordinator.owner_of(key) == 1


def test_assignment_charges_local_commitment_exactly_once() -> None:
    """Least-work includes coordinator-local charges: [8, 3, 3] must spread
    across the equal-min owners instead of stacking on one owner."""
    coordinator = OwnerLeaseCoordinator()
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=3)
    )
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=2, observation_seq=1, work=3)
    )
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=3, observation_seq=1, work=8)
    )
    key1 = _key("req-1")
    key2 = _key("req-2")
    # Equal-min tie resolves by stable numeric global rank.
    assert coordinator.assign(key1, projected_work=1) == 1
    # Owner 1 was charged, so the next equal-min lands on owner 2.
    assert coordinator.assign(key2, projected_work=1) == 2
    # Idempotent re-assignment must not double-charge owner 1.
    assert coordinator.assign(key1, projected_work=1) == 1
    # Both owners are at 4 now; owner 1 wins the tie.  A double charge would
    # leave owner 1 at 5 and hand this request to owner 2.
    assert coordinator.assign(_key("req-3"), projected_work=1) == 1

    # Admit the lease first (finish requires an accepted RESERVE); the
    # reservation itself carries no extra charge.
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    _grant(coordinator, manager, coordinator.reserve(key1, requested_through=1))
    # Release refunds the charge exactly once.
    command = coordinator.finish(key1)
    receipt = manager.apply(command)
    assert receipt.accepted
    assert receipt.released
    assert coordinator.apply_receipt(receipt)
    assert coordinator.release_count() == 1
    # Refunded: owner 1 (3) beats owner 2 (4) again.
    assert coordinator.assign(_key("req-4"), projected_work=1) == 1
    # A duplicate release receipt cannot refund a second time.
    duplicate = OwnerReceipt(
        key=key1,
        owner_id=1,
        command_seq=command.command_seq,
        accepted=True,
        released=True,
        runnable_through=0,
    )
    assert not coordinator.apply_receipt(duplicate)
    assert coordinator.release_count() == 1
    # req-4 charged owner 1 again (5 vs owner 2's 4), so the next request
    # lands on owner 2.
    assert coordinator.assign(_key("req-5"), projected_work=1) == 2


# -- epoch fence ---------------------------------------------------------------


def test_epoch_fence_blocks_stale_request_id_reuse() -> None:
    coordinator = _coordinator_with_owners(1, 2)
    old = _key("req-reused", epoch=0)
    new = _key("req-reused", epoch=1)
    assert coordinator.assign(old) == 1
    # A new epoch is admitted and starts fresh.
    assert coordinator.assign(new) == 1
    assert coordinator.owner_of(new) == 1
    # Stale-epoch assignment is fenced.
    with pytest.raises(EpochFenceError):
        coordinator.assign(old)
    # The old-epoch lease is a tombstone, not silently freed.
    assert coordinator.owner_of(old) == 1
    assert coordinator.is_superseded(old)
    assert not coordinator.is_released(old)
    assert coordinator.release_count() == 0


def test_epoch_reuse_does_not_leak_old_lease_horizons() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    old = _key("req-reused", epoch=0)
    new = _key("req-reused", epoch=1)
    assert coordinator.assign(old) == 1
    _grant(coordinator, manager, coordinator.reserve(old, requested_through=40))
    assert coordinator.runnable_through_of(old) == 40
    # Reused id at a new epoch starts from an empty slate; the old horizon
    # is retained by the old key (never silently freed).
    assert coordinator.assign(new) == 1
    assert coordinator.runnable_through_of(old) == 40
    assert coordinator.runnable_through_of(new) is None
    # The old commitment is freed only by its own RELEASE receipt.
    command = coordinator.finish(old)
    receipt = manager.apply(command)
    assert receipt.accepted
    assert receipt.released
    assert coordinator.apply_receipt(receipt)
    assert coordinator.release_count() == 1
    assert coordinator.is_released(old)
    assert not coordinator.is_released(new)


def test_old_release_new_epoch_isolation() -> None:
    coordinator = _coordinator_with_owners(1, 2)
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    old = _key("req-iso", epoch=0)
    new = _key("req-iso", epoch=1)
    coordinator.assign(old)
    _grant(coordinator, manager, coordinator.reserve(old, requested_through=20))
    command = coordinator.finish(old)
    assert coordinator.is_release_pending(old)
    # Epoch reuse must not clear the old release-pending state.
    coordinator.assign(new)
    assert coordinator.is_release_pending(old)
    # The old key's own RELEASE receipt still frees the old commitment.
    receipt = manager.apply(command)
    assert receipt.accepted and receipt.released
    assert coordinator.apply_receipt(receipt)
    assert coordinator.is_released(old)
    assert not coordinator.is_released(new)
    assert coordinator.release_count() == 1
    # The new-epoch lease is fully independent.
    command_new = coordinator.reserve(new, requested_through=20)
    _grant(coordinator, manager, command_new)
    command_new = coordinator.finish(new)
    _grant(coordinator, manager, command_new)
    assert coordinator.is_released(new)
    assert coordinator.release_count() == 2


# -- horizons and publication ---------------------------------------------------


def test_reserve_extend_chunk_horizon_gating() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000, grant_ceiling=40)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    assert coordinator.assign(key) == 1

    # RESERVE asks for an absolute horizon of 100 tokens; the worker grants a
    # chunk of 40, so publication is gated at 40.
    command = coordinator.reserve(key, requested_through=100)
    receipt = _grant(coordinator, manager, command)
    assert receipt.accepted
    assert receipt.runnable_through == 40
    assert coordinator.requested_through_of(key) == 100
    assert coordinator.runnable_through_of(key) == 40
    tokens = _publish(coordinator, manager, step_seq=1)
    assert [t.runnable_through for t in tokens] == [40]

    # EXTEND grows the granted horizon in chunks; publication follows.
    command = coordinator.extend(key, requested_through=100)
    receipt = _grant(coordinator, manager, command)
    assert receipt.runnable_through == 80
    tokens = _publish(coordinator, manager, step_seq=2)
    assert [t.runnable_through for t in tokens] == [80]

    command = coordinator.extend(key, requested_through=100)
    receipt = _grant(coordinator, manager, command)
    assert receipt.runnable_through == 100
    tokens = _publish(coordinator, manager, step_seq=3)
    assert [t.runnable_through for t in tokens] == [100]

    # Publication never exceeds the receipted horizon.
    assert coordinator.published_through(key) <= coordinator.runnable_through_of(key)


def test_accepted_receipts_fail_closed_on_horizon_violations() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    _grant(coordinator, manager, coordinator.reserve(key, requested_through=50))
    _publish(coordinator, manager, step_seq=1)
    assert coordinator.published_through(key) == 50

    # A receipt that regresses below the published horizon fails closed.
    command = coordinator.extend(key, requested_through=50)
    regress = OwnerReceipt(
        key=key,
        owner_id=1,
        command_seq=command.command_seq,
        accepted=True,
        runnable_through=20,
    )
    with pytest.raises(PublicationViolationError):
        coordinator.apply_receipt(regress)
    assert coordinator.runnable_through_of(key) == 50
    assert coordinator.published_through(key) == 50

    # A receipt that exceeds the command's requested horizon fails closed.
    exceed = OwnerReceipt(
        key=key,
        owner_id=1,
        command_seq=command.command_seq,
        accepted=True,
        runnable_through=999,
    )
    with pytest.raises(PublicationViolationError):
        coordinator.apply_receipt(exceed)
    assert coordinator.runnable_through_of(key) == 50
    assert coordinator.published_through(key) == 50


# -- lifecycle: preempt / restore / resume ---------------------------------------


def test_preempt_releases_capacity_and_resume_reacquires_on_same_owner() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000, grant_ceiling=40)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    _grant(coordinator, manager, coordinator.reserve(key, requested_through=100))
    _publish(coordinator, manager, step_seq=1)
    assert coordinator.runnable_through_of(key) == 40
    assert coordinator.published_through(key) == 40
    # The initial grant commits exactly the granted tokens to capacity.
    assert manager.free_capacity() == 1000 - 40

    # PREEMPT preserves the owner and releases active runnable capacity down
    # to the published horizon.
    command = coordinator.preempt(key, preempt_through=40)
    receipt = _grant(coordinator, manager, command)
    assert receipt.accepted
    assert coordinator.is_preempted(key)
    assert coordinator.owner_of(key) == 1
    assert coordinator.runnable_through_of(key) == 40
    # The physical commitment is released: capacity is fully available
    # again even though the logical horizon fence is retained.
    assert manager.free_capacity() == 1000
    # No new tokens publish for a preempted request.
    assert _publish(coordinator, manager, step_seq=2) == []
    assert manager.free_capacity() == 1000

    # RESTORE is the DMA/cold-residency intent: it does not reacquire
    # runnable capacity.
    command = coordinator.restore(key, requested_through=100)
    receipt = _grant(coordinator, manager, command)
    assert receipt.accepted
    assert coordinator.is_restored(key)
    assert coordinator.is_preempted(key)
    assert coordinator.runnable_through_of(key) == 40

    # RESUME reacquires a lease on the same (sticky) owner.
    command = coordinator.resume(key, requested_through=100)
    assert command.kind is OwnerCommandKind.RESERVE
    receipt = _grant(coordinator, manager, command)
    assert receipt.accepted
    assert not coordinator.is_preempted(key)
    assert coordinator.owner_of(key) == 1
    assert coordinator.runnable_through_of(key) == 80
    # RESUME reacquires the physical commitment for the new grant.
    assert manager.free_capacity() == 1000 - 80
    tokens = _publish(coordinator, manager, step_seq=3)
    assert [t.runnable_through for t in tokens] == [80]
    assert manager.free_capacity() == 1000 - 80


def test_published_tokens_cannot_be_refused() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    _grant(coordinator, manager, coordinator.reserve(key, requested_through=50))
    _publish(coordinator, manager, step_seq=1)
    assert manager.published_through(key) == 50

    # A preempt below the published horizon would refuse published tokens:
    # the worker rejects it and the coordinator state does not change.
    command = coordinator.preempt(key, preempt_through=30)
    receipt = manager.apply(command)
    assert not receipt.accepted
    assert "refuses published tokens" in receipt.error
    assert not coordinator.apply_receipt(receipt)
    assert not coordinator.is_preempted(key)
    assert coordinator.published_through(key) == 50
    assert coordinator.runnable_through_of(key) == 50


# -- receipts ----------------------------------------------------------------------


def test_stale_duplicate_wrong_owner_wrong_epoch_receipts_are_ignored() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    command = coordinator.reserve(key, requested_through=50)
    _grant(coordinator, manager, command)
    assert coordinator.runnable_through_of(key) == 50

    # Stale: an old command sequence cannot advance state.
    stale = OwnerReceipt(
        key=key,
        owner_id=1,
        command_seq=command.command_seq - 1,
        accepted=True,
        runnable_through=999,
    )
    assert not coordinator.apply_receipt(stale)
    assert coordinator.runnable_through_of(key) == 50

    # Duplicate: the same sequence applied twice cannot advance state.
    duplicate = OwnerReceipt(
        key=key,
        owner_id=1,
        command_seq=command.command_seq,
        accepted=True,
        runnable_through=999,
    )
    assert not coordinator.apply_receipt(duplicate)
    assert coordinator.runnable_through_of(key) == 50

    # Wrong owner.
    wrong_owner = OwnerReceipt(
        key=key,
        owner_id=9,
        command_seq=coordinator.extend(key, requested_through=50).command_seq,
        accepted=True,
        runnable_through=999,
    )
    assert not coordinator.apply_receipt(wrong_owner)
    assert coordinator.runnable_through_of(key) == 50

    # Wrong epoch: no lease exists for the un-admitted epoch key.
    wrong_epoch = OwnerReceipt(
        key=_key(request_id="req-0", epoch=1),
        owner_id=1,
        command_seq=1,
        accepted=True,
        runnable_through=999,
    )
    assert not coordinator.apply_receipt(wrong_epoch)
    assert coordinator.runnable_through_of(key) == 50


def test_finish_abort_release_pending_and_exact_once_release() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    _grant(coordinator, manager, coordinator.reserve(key, requested_through=50))
    _publish(coordinator, manager, step_seq=1)

    # Finish/abort leaves the commitment release-pending.
    command = coordinator.finish(key)
    assert command.kind is OwnerCommandKind.RELEASE
    assert coordinator.is_release_pending(key)
    assert not coordinator.is_released(key)
    # No new publication while release is pending.
    assert _publish(coordinator, manager, step_seq=2) == []

    # Only the matching RELEASE receipt frees the commitment, exactly once.
    receipt = manager.apply(command)
    assert receipt.accepted
    assert receipt.released
    assert coordinator.apply_receipt(receipt)
    assert not coordinator.is_release_pending(key)
    assert coordinator.is_released(key)
    assert coordinator.release_count() == 1

    # A duplicate of the same receipt cannot free again.
    duplicate = OwnerReceipt(
        key=key,
        owner_id=1,
        command_seq=command.command_seq,
        accepted=True,
        released=True,
        runnable_through=50,
    )
    assert not coordinator.apply_receipt(duplicate)
    assert coordinator.release_count() == 1

    # A receipt claiming release against a non-RELEASE command cannot free.
    second = _key("req-1")
    coordinator.assign(second)
    _grant(coordinator, manager, coordinator.reserve(second, requested_through=10))
    command = coordinator.extend(second, requested_through=20)
    forged = OwnerReceipt(
        key=second,
        owner_id=1,
        command_seq=command.command_seq,
        accepted=True,
        released=True,
        runnable_through=20,
    )
    assert coordinator.apply_receipt(forged)
    assert not coordinator.is_released(second)
    assert coordinator.release_count() == 1


def test_release_frees_exactly_once_across_manager_and_coordinator() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    keys = [_key(f"req-{i}") for i in range(3)]
    for key in keys:
        coordinator.assign(key)
        _grant(coordinator, manager, coordinator.reserve(key, requested_through=20))
        _publish(coordinator, manager, step_seq=1)
        command = coordinator.finish(key)
        _grant(coordinator, manager, command)
    assert coordinator.release_count() == 3
    assert manager.release_count() == 3
    for key in keys:
        assert coordinator.is_released(key)


# -- provisional assignment: refusal, abandon, retry ------------------------------


def test_zero_capacity_reserve_refused_and_provisional_retried() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager1 = AttentionLeaseManager(owner_rank=1, capacity=0)
    manager2 = AttentionLeaseManager(owner_rank=2, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=2, observation_seq=1, work=0)
    )
    key = _key()
    assert coordinator.assign(key, projected_work=10) == 1

    # Zero-capacity RESERVE is a real pre-publication refusal, not an
    # accepted zero-horizon grant.
    command = coordinator.reserve(key, requested_through=50)
    receipt = manager1.apply(command)
    assert not receipt.accepted
    assert "capacity" in receipt.error
    assert receipt.runnable_through is None
    assert not coordinator.apply_receipt(receipt)
    assert coordinator.runnable_through_of(key) is None
    assert coordinator.published_through(key) == 0

    # Abandon the provisional assignment and retry another owner.
    assert coordinator.abandon(key)
    assert coordinator.owner_of(key) is None
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=2, work=1000)
    )
    assert coordinator.assign(key, projected_work=10) == 2
    command = coordinator.reserve(key, requested_through=50)
    receipt = manager2.apply(command)
    assert receipt.accepted
    assert receipt.runnable_through == 50
    assert coordinator.apply_receipt(receipt)
    assert coordinator.owner_of(key) == 2

    # The admitted owner is sticky: abandon is refused, re-assign is
    # idempotent and does not charge twice.
    with pytest.raises(OwnershipError):
        coordinator.abandon(key)
    assert coordinator.assign(key, projected_work=10) == 2


def test_abandon_refunds_provisional_charge() -> None:
    coordinator = _coordinator_with_owners(1, 2)
    key = _key()
    assert coordinator.assign(key, projected_work=5) == 1
    assert coordinator.abandon(key)
    assert coordinator.owner_of(key) is None
    # The charge was refunded: a fresh tie still picks owner 1.
    assert coordinator.assign(_key("req-2"), projected_work=5) == 1


# -- receipt batch envelope ---------------------------------------------------------


def test_worker_emits_exactly_one_batch_per_step() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    _grant(coordinator, manager, coordinator.reserve(key, requested_through=50))

    batch = manager.emit_batch(emitted_step_seq=7)
    assert batch.owner_rank == 1
    assert batch.emitted_step_seq == 7
    assert len(batch.events) == 1
    assert batch.events[0].key == key
    assert batch.free_capacity == 1000 - 50
    assert batch.resident_pages == 0
    assert batch.pending_dma == 0

    # Empty work still yields a batch: differs from a missing response.
    idle = manager.emit_batch(emitted_step_seq=8)
    assert idle.events == ()
    assert idle.owner_rank == 1


# -- carriers ------------------------------------------------------------------------


def test_request_and_output_carriers_are_backward_compatible() -> None:
    request = Request(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    assert request.attention_owner is None
    assert request.attention_owner_epoch == 0
    assert request.status is RequestStatus.WAITING
    # Setting the owner does not alter RequestStatus.
    request.attention_owner = 3
    request.attention_owner_epoch = 1
    assert request.status is RequestStatus.WAITING

    data = NewRequestData.from_request(request, block_ids=([0],))
    assert data.attention_owner == 3
    assert data.attention_owner_epoch == 1

    # Direct constructors stay compatible.
    direct = NewRequestData(
        req_id="r2",
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
    )
    assert direct.attention_owner is None
    assert direct.attention_owner_epoch == 0

    def _output(**overrides) -> SchedulerOutput:
        kwargs = dict(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        kwargs.update(overrides)
        return SchedulerOutput(**kwargs)

    empty = _output()
    assert empty.step_seq == 0
    assert empty.owner_commands == []
    assert empty.owner_assignment_observations == []
    assert empty.scheduled_owner_leases == []

    out = _output(
        step_seq=5,
        owner_assignment_observations=[
            OwnerAssignmentObservation(owner_id=1, observation_seq=1)
        ],
    )
    assert out.step_seq == 5
    assert out.owner_assignment_observations[0].owner_id == 1
    assert SchedulerOutput.make_empty().step_seq == 0


# -- red-team follow-up: charge semantics, sequencing, finish, publication -------


def test_observation_work_excludes_coordinator_local_charges() -> None:
    """``work`` is the external/base workload observation; coordinator-local
    projected charges are added by ``assign()`` separately and must never be
    folded back into the observation (no double counting)."""
    coordinator = OwnerLeaseCoordinator()
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=3)
    )
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=2, observation_seq=1, work=5)
    )
    assert coordinator.assign(_key("req-1"), projected_work=2) == 1
    # Owner 1's score is 3 (base work) + 2 (local charge) = 5, tying owner
    # 2's base 5.  Re-reporting the same base work keeps the charge: if the
    # producer had folded the charge into ``work`` (5), owner 1 would score
    # 7 and owner 2 would win the next assignment.
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=2, work=3)
    )
    assert coordinator.assign(_key("req-2"), projected_work=1) == 1


def test_owner_command_sequence_is_monotonic_across_keys() -> None:
    """Commands for an owner are delivered reliably and in order: the
    per-owner ``command_seq`` fence increases monotonically across all
    request keys, so the worker consumes one strictly increasing stream."""
    coordinator = _coordinator_with_owners(1)
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    key_a = _key("req-a")
    key_b = _key("req-b")
    coordinator.assign(key_a)
    coordinator.assign(key_b)
    commands = [
        coordinator.reserve(key_a, requested_through=50),
        coordinator.reserve(key_b, requested_through=50),
        coordinator.extend(key_a, requested_through=100),
        coordinator.extend(key_b, requested_through=100),
    ]
    seqs = [c.command_seq for c in commands]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    # The worker consumes the stream in order without any fence violation.
    for command in commands:
        receipt = manager.apply(command)
        assert receipt.accepted


def test_finish_is_idempotent_while_release_pending() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    _grant(coordinator, manager, coordinator.reserve(key, requested_through=50))
    _publish(coordinator, manager, step_seq=1)

    # A double finish before the first receipt converges: the same
    # outstanding RELEASE is returned and command_seq does not advance.
    first = coordinator.finish(key)
    second = coordinator.finish(key)
    assert first == second
    assert first.kind is OwnerCommandKind.RELEASE
    assert coordinator.is_release_pending(key)
    assert not coordinator.is_released(key)

    # Applying the outstanding RELEASE once frees exactly once.
    receipt = manager.apply(first)
    assert receipt.accepted
    assert receipt.released
    assert coordinator.apply_receipt(receipt)
    assert coordinator.release_count() == 1
    assert coordinator.is_released(key)
    with pytest.raises(OwnershipError):
        coordinator.finish(key)


def test_finish_rejected_before_any_accepted_reserve() -> None:
    coordinator = _coordinator_with_owners(1, 2)
    key = _key()
    coordinator.assign(key)
    # Provisional: no RESERVE has been accepted, so finish is refused.
    with pytest.raises(OwnershipError):
        coordinator.finish(key)
    # After a refused RESERVE the lease is still provisional: finish is
    # still refused and the caller must abandon instead.
    command = coordinator.reserve(key, requested_through=50)
    refused = OwnerReceipt(
        key=key,
        owner_id=1,
        command_seq=command.command_seq,
        accepted=False,
    )
    assert not coordinator.apply_receipt(refused)
    with pytest.raises(OwnershipError):
        coordinator.finish(key)
    assert coordinator.abandon(key)
    assert coordinator.owner_of(key) is None


def test_publish_waits_for_receipted_grants() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000, grant_ceiling=40)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)

    # A RESERVE that has not been receipted publishes nothing.
    command = coordinator.reserve(key, requested_through=100)
    assert _publish(coordinator, manager, step_seq=1) == []
    # Only after its accepted receipt does the grant publish.
    _grant(coordinator, manager, command)
    tokens = _publish(coordinator, manager, step_seq=2)
    assert [t.runnable_through for t in tokens] == [40]

    # An EXTEND in flight (issued, not receipted) publishes nothing new.
    command = coordinator.extend(key, requested_through=100)
    assert _publish(coordinator, manager, step_seq=3) == []
    _grant(coordinator, manager, command)
    tokens = _publish(coordinator, manager, step_seq=4)
    assert [t.runnable_through for t in tokens] == [80]


def test_publish_only_from_reserve_or_extend_grants() -> None:
    coordinator = OwnerLeaseCoordinator()
    manager = AttentionLeaseManager(owner_rank=1, capacity=1000, grant_ceiling=40)
    coordinator.observe(
        OwnerAssignmentObservation(owner_id=1, observation_seq=1, work=0)
    )
    key = _key()
    coordinator.assign(key)
    _grant(coordinator, manager, coordinator.reserve(key, requested_through=100))
    tokens = _publish(coordinator, manager, step_seq=1)
    assert [t.runnable_through for t in tokens] == [40]

    # PREEMPT, RESTORE, and RELEASE receipts never advance publication.
    _grant(coordinator, manager, coordinator.preempt(key, preempt_through=40))
    assert _publish(coordinator, manager, step_seq=2) == []
    _grant(coordinator, manager, coordinator.restore(key, requested_through=40))
    assert _publish(coordinator, manager, step_seq=3) == []
    _grant(coordinator, manager, coordinator.finish(key))
    assert _publish(coordinator, manager, step_seq=4) == []
    assert coordinator.is_released(key)
