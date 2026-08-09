# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU tests for the worker-side reference attention lease manager.

The manager under test lives in :class:`AttentionLeaseManager`
(vllm.v1.core.sched.ownership) and is dependency-neutral; no GPU model runner
is constructed here.
"""

import pickle

import pytest

from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
    OwnerReceipt,
    OwnerReceiptBatch,
    PublicationViolationError,
)


def _key(request_id: str = "req-0", epoch: int = 0) -> OwnerLeaseKey:
    return OwnerLeaseKey(request_id=request_id, owner_epoch=epoch)


def _command(
    key: OwnerLeaseKey,
    owner_id: int,
    seq: int,
    kind: OwnerCommandKind,
    requested_through: int,
) -> OwnerCommand:
    return OwnerCommand(
        key=key,
        owner_id=owner_id,
        command_seq=seq,
        kind=kind,
        requested_through=requested_through,
    )


def _token(
    key: OwnerLeaseKey,
    owner_id: int,
    runnable_through: int,
    step_seq: int = 1,
    command_seq: int = 1,
) -> OwnerLeaseToken:
    return OwnerLeaseToken(
        key=key,
        owner_id=owner_id,
        step_seq=step_seq,
        command_seq=command_seq,
        runnable_through=runnable_through,
    )


def _manager(capacity: int = 1000, ceiling: int | None = None) -> AttentionLeaseManager:
    return AttentionLeaseManager(owner_rank=1, capacity=capacity, grant_ceiling=ceiling)


# -- command handling ------------------------------------------------------------


def test_reserve_grants_absolute_horizon() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    receipt = manager.apply(
        _command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100)
    )
    assert receipt.accepted
    assert receipt.error is None
    assert receipt.runnable_through == 40
    assert receipt.free_capacity == 160
    assert manager.published_through(key) == 0


def test_extend_grows_horizon_in_chunks_and_never_shrinks() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    assert (
        manager.apply(
            _command(key, 1, 2, OwnerCommandKind.EXTEND, requested_through=100)
        ).runnable_through
        == 80
    )
    assert (
        manager.apply(
            _command(key, 1, 3, OwnerCommandKind.EXTEND, requested_through=100)
        ).runnable_through
        == 100
    )
    # An extend below the current horizon is rejected.
    receipt = manager.apply(
        _command(key, 1, 4, OwnerCommandKind.EXTEND, requested_through=50)
    )
    assert not receipt.accepted
    assert "shrink" in receipt.error


def test_command_sequences_are_monotonically_fenced() -> None:
    manager = _manager()
    key = _key()
    first = manager.apply(
        _command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=10)
    )
    assert first.accepted
    # Replays and older sequences are rejected.
    replay = manager.apply(
        _command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=10)
    )
    assert not replay.accepted
    assert "stale or duplicate" in replay.error
    older = manager.apply(
        _command(key, 1, 0, OwnerCommandKind.EXTEND, requested_through=20)
    )
    assert not older.accepted
    # Newer sequences proceed.
    newer = manager.apply(
        _command(key, 1, 2, OwnerCommandKind.EXTEND, requested_through=20)
    )
    assert newer.accepted


def test_wrong_owner_commands_are_rejected() -> None:
    manager = _manager()
    receipt = manager.apply(
        _command(_key(), 2, 1, OwnerCommandKind.RESERVE, requested_through=10)
    )
    assert not receipt.accepted
    assert "wrong owner rank" in receipt.error


def test_epoch_reuse_supersedes_older_lease() -> None:
    manager = _manager()
    old = _key("req-reused", epoch=0)
    new = _key("req-reused", epoch=1)
    assert manager.apply(
        _command(old, 1, 1, OwnerCommandKind.RESERVE, requested_through=10)
    ).accepted
    # A newer epoch supersedes the old lease, but the old lease's own
    # RELEASE must still be honored so its commitment frees exactly once
    # instead of leaking forever.
    assert manager.apply(
        _command(new, 1, 2, OwnerCommandKind.RESERVE, requested_through=10)
    ).accepted
    stale_release = manager.apply(
        _command(old, 1, 3, OwnerCommandKind.RELEASE, requested_through=10)
    )
    assert stale_release.accepted
    assert stale_release.released
    assert manager.release_count() == 1
    # A second old-epoch release cannot free again.
    again = manager.apply(
        _command(old, 1, 4, OwnerCommandKind.RELEASE, requested_through=10)
    )
    assert not again.accepted
    assert "already released" in again.error
    assert manager.release_count() == 1
    # The new-epoch lease is released on its own.
    assert manager.apply(
        _command(new, 1, 5, OwnerCommandKind.RELEASE, requested_through=10)
    ).accepted
    assert manager.release_count() == 2


def test_lower_epoch_commands_are_rejected_except_tombstone_release() -> None:
    manager = _manager()
    old = _key("req-fenced", epoch=0)
    new = _key("req-fenced", epoch=1)
    assert manager.apply(
        _command(old, 1, 1, OwnerCommandKind.RESERVE, requested_through=10)
    ).accepted
    assert manager.apply(
        _command(new, 1, 2, OwnerCommandKind.RESERVE, requested_through=10)
    ).accepted
    # A stale lower-epoch RESERVE cannot recreate state for the old epoch.
    reserve = manager.apply(
        _command(old, 1, 3, OwnerCommandKind.RESERVE, requested_through=10)
    )
    assert not reserve.accepted
    assert "stale request epoch" in reserve.error
    # A stale lower-epoch EXTEND is equally rejected.
    extend = manager.apply(
        _command(old, 1, 4, OwnerCommandKind.EXTEND, requested_through=20)
    )
    assert not extend.accepted
    assert "stale request epoch" in extend.error
    # Only the matching old-epoch RELEASE is honored; the superseded
    # commitment frees exactly once while the new lease keeps its grant.
    release = manager.apply(
        _command(old, 1, 5, OwnerCommandKind.RELEASE, requested_through=10)
    )
    assert release.accepted
    assert release.released
    assert manager.release_count() == 1
    assert manager.free_capacity() == manager.capacity - 10


# -- publication invariants --------------------------------------------------------


def test_published_tokens_cannot_be_refused() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    manager.record_published(_token(key, 1, runnable_through=40))
    assert manager.published_through(key) == 40

    # Preempt below the published horizon refuses published tokens.
    preempt = manager.apply(
        _command(key, 1, 2, OwnerCommandKind.PREEMPT, requested_through=30)
    )
    assert not preempt.accepted
    assert "refuses published tokens" in preempt.error
    # Release below the published horizon is equally illegal.
    release = manager.apply(
        _command(key, 1, 3, OwnerCommandKind.RELEASE, requested_through=30)
    )
    assert not release.accepted
    assert "refuses published tokens" in release.error
    assert manager.release_count() == 0
    # Honoring the published horizon works for both.
    assert manager.apply(
        _command(key, 1, 4, OwnerCommandKind.PREEMPT, requested_through=40)
    ).accepted
    assert manager.apply(
        _command(key, 1, 5, OwnerCommandKind.RELEASE, requested_through=40)
    ).accepted
    assert manager.release_count() == 1


def test_record_published_above_granted_horizon_raises() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    with pytest.raises(PublicationViolationError):
        manager.record_published(_token(key, 1, runnable_through=50))


def test_preempt_then_restore_then_resume() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    assert manager.apply(
        _command(key, 1, 2, OwnerCommandKind.PREEMPT, requested_through=40)
    ).accepted
    # Extending a preempted lease is rejected until RESUME.
    extend = manager.apply(
        _command(key, 1, 3, OwnerCommandKind.EXTEND, requested_through=60)
    )
    assert not extend.accepted
    assert "preempted" in extend.error
    # RESTORE only signals the DMA/cold-residency intent; the lease stays
    # preempted, so EXTEND remains rejected.
    assert manager.apply(
        _command(key, 1, 4, OwnerCommandKind.RESTORE, requested_through=100)
    ).accepted
    still_preempted = manager.apply(
        _command(key, 1, 5, OwnerCommandKind.EXTEND, requested_through=100)
    )
    assert not still_preempted.accepted
    assert "preempted" in still_preempted.error
    # RESUME (a RESERVE on the preempted lease) reacquires capacity on the
    # same owner; only then does EXTEND grow the horizon again.
    resume = manager.apply(
        _command(key, 1, 6, OwnerCommandKind.RESERVE, requested_through=100)
    )
    assert resume.accepted
    assert resume.runnable_through == 80
    assert manager.apply(
        _command(key, 1, 7, OwnerCommandKind.EXTEND, requested_through=100)
    ).accepted


# -- exact-once release ------------------------------------------------------------


def test_release_frees_commitment_exactly_once() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    receipt = manager.apply(
        _command(key, 1, 2, OwnerCommandKind.RELEASE, requested_through=40)
    )
    assert receipt.accepted
    assert receipt.released
    assert receipt.free_capacity == 200
    assert manager.release_count() == 1
    # A second release cannot free again.
    again = manager.apply(
        _command(key, 1, 3, OwnerCommandKind.RELEASE, requested_through=40)
    )
    assert not again.accepted
    assert "already released" in again.error
    assert manager.release_count() == 1


# -- batch emission ----------------------------------------------------------------


def test_emit_batch_carries_events_and_facts() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    manager.apply(_command(key, 1, 2, OwnerCommandKind.EXTEND, requested_through=100))
    batch = manager.emit_batch(emitted_step_seq=3)
    assert isinstance(batch, OwnerReceiptBatch)
    assert batch.owner_rank == 1
    assert batch.emitted_step_seq == 3
    assert len(batch.events) == 2
    assert batch.free_capacity == 200 - 80
    assert batch.resident_pages == 0
    assert batch.pending_dma == 0
    # The outbox is drained; the next batch has no events but still exists.
    idle = manager.emit_batch(emitted_step_seq=4)
    assert idle.events == ()
    assert idle.emitted_step_seq == 4


def test_batch_receipts_are_ordered_and_complete() -> None:
    manager = _manager()
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=10))
    manager.apply(_command(key, 1, 2, OwnerCommandKind.EXTEND, requested_through=20))
    batch = manager.emit_batch(emitted_step_seq=1)
    assert [r.command_seq for r in batch.events] == [1, 2]
    assert all(r.key == key for r in batch.events)
    assert all(r.owner_id == 1 for r in batch.events)


# -- protocol types ----------------------------------------------------------------


def test_protocol_types_are_frozen_and_pickle_safe() -> None:
    key = _key()
    command = _command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=10)
    receipt = OwnerReceipt(
        key=key,
        owner_id=1,
        command_seq=1,
        accepted=True,
        runnable_through=10,
        released=False,
    )
    token = _token(key, 1, runnable_through=10)
    batch = OwnerReceiptBatch(owner_rank=1, emitted_step_seq=2, events=(receipt,))
    for value in (key, command, receipt, token, batch):
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value
        with pytest.raises(AttributeError):
            value.owner_epoch = 9  # type: ignore[misc]
            value.runnable_through = 9  # type: ignore[misc]
    assert isinstance(receipt, OwnerReceipt)
    assert command.kind is OwnerCommandKind.RESERVE


# -- red-team follow-up: publication enforcement, facts on release --------------


def test_record_published_refuses_inactive_leases() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    # Unknown lease.
    with pytest.raises(PublicationViolationError):
        manager.record_published(_token(key, 1, runnable_through=10))
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    # Preempted lease.
    manager.apply(_command(key, 1, 2, OwnerCommandKind.PREEMPT, requested_through=40))
    with pytest.raises(PublicationViolationError):
        manager.record_published(_token(key, 1, runnable_through=40))
    # Released lease.
    manager.apply(_command(key, 1, 3, OwnerCommandKind.RELEASE, requested_through=40))
    with pytest.raises(PublicationViolationError):
        manager.record_published(_token(key, 1, runnable_through=40))
    # Superseded lease (old epoch tombstoned by a newer epoch).
    old = _key("req-sup", epoch=0)
    new = _key("req-sup", epoch=1)
    manager.apply(_command(old, 1, 4, OwnerCommandKind.RESERVE, requested_through=10))
    manager.apply(_command(new, 1, 5, OwnerCommandKind.RESERVE, requested_through=10))
    with pytest.raises(PublicationViolationError):
        manager.record_published(_token(old, 1, runnable_through=10))


def test_record_published_fences_command_and_step_sequences() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    # A token carrying a stale command sequence is refused.
    stale = _token(key, 1, runnable_through=40, step_seq=1, command_seq=2)
    with pytest.raises(PublicationViolationError):
        manager.record_published(stale)
    # The correct sequence records; duplicate or regressed steps are refused.
    manager.record_published(
        _token(key, 1, runnable_through=40, step_seq=1, command_seq=1)
    )
    duplicate = _token(key, 1, runnable_through=40, step_seq=1, command_seq=1)
    with pytest.raises(PublicationViolationError):
        manager.record_published(duplicate)
    regressed = _token(key, 1, runnable_through=40, step_seq=0, command_seq=1)
    with pytest.raises(PublicationViolationError):
        manager.record_published(regressed)
    # A later step with a fresh grant sequence proceeds.
    manager.apply(_command(key, 1, 2, OwnerCommandKind.EXTEND, requested_through=100))
    manager.record_published(
        _token(key, 1, runnable_through=80, step_seq=2, command_seq=2)
    )
    assert manager.published_through(key) == 80


def test_record_published_wrong_owner_ignored() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    # Tokens for other owners are ignored, not errors.
    manager.record_published(_token(key, 9, runnable_through=40))
    assert manager.published_through(key) == 0


def test_release_clears_dma_and_residency_facts() -> None:
    manager = _manager(capacity=200, ceiling=40)
    key = _key()
    manager.apply(_command(key, 1, 1, OwnerCommandKind.RESERVE, requested_through=100))
    manager.apply(_command(key, 1, 2, OwnerCommandKind.PREEMPT, requested_through=40))
    restore = manager.apply(
        _command(key, 1, 3, OwnerCommandKind.RESTORE, requested_through=100)
    )
    assert restore.accepted
    assert restore.pending_dma == 1
    inflight = manager.emit_batch(emitted_step_seq=4)
    assert inflight.pending_dma == 1
    assert inflight.resident_pages == 0

    release = manager.apply(
        _command(key, 1, 5, OwnerCommandKind.RELEASE, requested_through=40)
    )
    assert release.accepted
    assert release.pending_dma == 0
    assert manager.release_count() == 1
    # The freed lease contributes no DMA/residency facts to the batch and
    # its capacity returns.
    batch = manager.emit_batch(emitted_step_seq=6)
    assert batch.pending_dma == 0
    assert batch.resident_pages == 0
    assert batch.free_capacity == 200
    assert [r.released for r in batch.events] == [True]
