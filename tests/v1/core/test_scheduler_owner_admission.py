# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused tests for the G2 scheduler-side receipt-gated owner admission.

The scheduler's request-owned control plane replaces ordinary KV scheduling:
it assigns least-committed-work owners, issues fenced owner commands, applies
worker receipts, and publishes scheduled leases only after accepted receipts.
These tests exercise that plane through a minimal ``Scheduler.__new__``
harness (a real Scheduler needs a full engine stack), mirroring
``test_owner_step_seq.py``.
"""

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.ownership import (
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerReceipt,
    OwnerReceiptBatch,
)
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


class _KVCacheSpy:
    """KV cache manager stub that fails closed on any KV-block API.

    ``take_events`` and ``reset_prefix_cache`` are the only entry points the
    G2 scheduler may touch; every other attribute raises so a control-plane
    step accidentally reaching a KV-block API fails the test loudly.
    """

    FORBIDDEN = (
        "allocate_slots",
        "get_blocks",
        "free",
        "pop_blocks_for_free",
        "take_new_block_ids",
        "new_step_starts",
    )

    def __init__(self) -> None:
        self.take_events_calls = 0
        self.reset_prefix_cache_calls = 0

    def take_events(self):
        self.take_events_calls += 1
        return None

    def reset_prefix_cache(self) -> bool:
        self.reset_prefix_cache_calls += 1
        return True

    def __getattr__(self, name: str):
        if name in self.FORBIDDEN:
            raise AssertionError(
                f"kv_cache_manager.{name}() must never run in G2 mode"
            )
        raise AttributeError(name)


def _make_scheduler(
    *,
    world_size: int = 2,
    max_num_scheduled_tokens: int = 64,
    max_num_running_reqs: int = 16,
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler_config = SimpleNamespace(enable_request_owned_attention=True)
    scheduler.parallel_config = SimpleNamespace(world_size=world_size)
    scheduler.current_step = 0
    scheduler.max_num_scheduled_tokens = max_num_scheduled_tokens
    scheduler.max_num_running_reqs = max_num_running_reqs
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.running = []
    scheduler.requests = {}
    scheduler.log_stats = False
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: [], free=lambda request: None
    )
    scheduler.finished_req_ids = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.num_spec_tokens = 0
    scheduler.enable_return_routed_experts = False
    scheduler._inflight_prefills = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.defer_block_free = False
    scheduler.kv_cache_manager = _KVCacheSpy()
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.prev_step_scheduled_req_ids = set()
    # G2 control-plane state (normally initialized at the tail of __init__).
    scheduler.owner_coordinator = None
    scheduler._owner_key = {}
    scheduler._owner_epoch = {}
    scheduler._owner_pending_command = {}
    scheduler._owner_emitted_command_seq = {}
    scheduler._owner_outbox = []
    scheduler._owner_token_plans = {}
    scheduler._init_request_owned_control_plane()
    return scheduler


def _request(request_id: str, num_prompt_tokens: int = 32, max_tokens: int = 8):
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_prompt_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
    )


def _apply_receipts(scheduler, scheduler_output, events_by_rank) -> None:
    """Feed the all-worker receipt envelope back into the scheduler."""
    batches = [
        OwnerReceiptBatch(
            owner_rank=rank,
            emitted_step_seq=scheduler_output.step_seq,
            events=tuple(events_by_rank.get(rank, ())),
        )
        for rank in range(scheduler.parallel_config.world_size)
    ]
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[], req_id_to_index={}, owner_receipt_batches=batches
        ),
    )


def _receipt(
    key: OwnerLeaseKey,
    owner_id: int,
    command_seq: int,
    *,
    accepted: bool = True,
    runnable: int | None = None,
    released: bool = False,
    error: str | None = None,
) -> OwnerReceipt:
    return OwnerReceipt(
        key=key,
        owner_id=owner_id,
        command_seq=command_seq,
        accepted=accepted,
        runnable_num_tokens=runnable,
        released=released,
        error=error,
    )


def _admit(scheduler, request, *, grant: int):
    """Admit ``request``: RESERVE step, accepted receipt, promote step.

    Returns the promoted step output.  ``grant`` is the runnable count the
    worker grants for the initial RESERVE.
    """
    out1 = scheduler.schedule()
    (command,) = out1.owner_commands
    assert command.kind is OwnerCommandKind.RESERVE
    _apply_receipts(
        scheduler,
        out1,
        {command.owner_id: [_receipt(command.key, command.owner_id,
                                     command.command_seq, runnable=grant)]},
    )
    out2 = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    return out2


def test_empty_control_step_is_valid_and_stamps_step() -> None:
    scheduler = _make_scheduler()
    out = scheduler.schedule()
    assert out.step_seq == 1
    assert scheduler.current_step == 1
    assert out.total_num_scheduled_tokens == 0
    assert out.num_scheduled_tokens == {}
    assert out.scheduled_new_reqs == []
    assert out.new_block_ids_to_zero is None
    assert out.num_common_prefix_blocks == []
    assert out.owner_commands == []
    assert out.scheduled_owner_leases == []
    assert [o.owner_id for o in out.owner_assignment_observations] == [0, 1]
    # The empty envelope round-trips.
    _apply_receipts(scheduler, out, {})


def test_two_step_admission_reserves_then_promotes_and_publishes() -> None:
    scheduler = _make_scheduler()
    request = _request("req-0")
    scheduler.add_request(request)

    # Step 1: RESERVE(WAITING) with descriptor; request stays provisional.
    out1 = scheduler.schedule()
    (command,) = out1.owner_commands
    assert command.kind is OwnerCommandKind.RESERVE
    assert command.key == OwnerLeaseKey("req-0", 0)
    assert command.owner_id == 0
    assert command.required_num_tokens == 32
    assert command.allocation == OwnerAllocationDescriptor(
        key=OwnerLeaseKey("req-0", 0),
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_tokens=32,
        status=OwnerAdmissionStatus.WAITING,
    )
    assert out1.scheduled_owner_leases == []
    assert request.status == RequestStatus.WAITING
    assert request.attention_owner is None
    assert scheduler.has_requests()

    # No promotion until the accepted receipt is applied.
    _apply_receipts(scheduler, out1, {0: [_receipt(command.key, 0,
                                                   command.command_seq,
                                                   runnable=32)]})
    assert request.attention_owner == 0
    assert request.attention_owner_epoch == 0
    assert request.status == RequestStatus.WAITING

    # Step 2: accepted lease promotes and publishes the lease token.
    out2 = scheduler.schedule()
    assert out2.owner_commands == []
    assert request.status == RequestStatus.RUNNING
    assert request in scheduler.running
    assert request.attention_owner == 0
    (token,) = out2.scheduled_owner_leases
    assert token.key == OwnerLeaseKey("req-0", 0)
    assert token.owner_id == 0
    assert token.step_seq == 2
    assert token.runnable_num_tokens == 32


def test_refused_reserve_abandons_and_retries_without_token_mutation() -> None:
    scheduler = _make_scheduler()
    request = _request("req-0")
    scheduler.add_request(request)

    out1 = scheduler.schedule()
    (command,) = out1.owner_commands
    key = command.key
    _apply_receipts(
        scheduler,
        out1,
        {0: [_receipt(key, 0, command.command_seq, accepted=False,
                      runnable=None, error="insufficient capacity")]},
    )
    # Refusal abandons the provisional assignment; nothing was mutated.
    assert key not in scheduler._owner_key
    assert scheduler.owner_coordinator.owner_of(key) is None
    assert request.attention_owner is None
    assert request.status == RequestStatus.WAITING
    assert request.num_computed_tokens == 0

    # The next step re-issues RESERVE with a fresh command sequence.
    out2 = scheduler.schedule()
    (retry,) = out2.owner_commands
    assert retry.kind is OwnerCommandKind.RESERVE
    assert retry.key == key
    assert retry.command_seq == command.command_seq + 1
    assert retry.required_num_tokens == 32
    assert request.status == RequestStatus.WAITING


def test_horizon_cap_extend_stall_and_refused_extend_retry() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)
    request.num_computed_tokens = 8

    # At the granted horizon with work remaining: EXTEND and stall.
    out3 = scheduler.schedule()
    (extend,) = out3.owner_commands
    assert extend.kind is OwnerCommandKind.EXTEND
    assert extend.required_num_tokens == 16
    assert scheduler._owner_token_plans["req-0"] == 0
    # No lease publishes while the EXTEND receipt is in flight.
    assert out3.scheduled_owner_leases == []

    # Refused EXTEND: no abandon (already admitted), and the next step
    # re-issues EXTEND at the horizon.
    _apply_receipts(
        scheduler,
        out3,
        {0: [_receipt(extend.key, 0, extend.command_seq, accepted=False,
                      runnable=None, error="no capacity")]},
    )
    out4 = scheduler.schedule()
    (retry,) = out4.owner_commands
    assert retry.kind is OwnerCommandKind.EXTEND
    assert retry.command_seq == extend.command_seq + 1
    assert retry.required_num_tokens == 16
    assert out4.scheduled_owner_leases == []

    # Accepted EXTEND: plan resumes and the lease publishes past the horizon.
    _apply_receipts(
        scheduler,
        out4,
        {0: [_receipt(retry.key, 0, retry.command_seq, runnable=16)]},
    )
    out5 = scheduler.schedule()
    assert out5.owner_commands == []
    assert scheduler._owner_token_plans["req-0"] == 8
    (token,) = out5.scheduled_owner_leases
    assert token.command_seq == retry.command_seq
    # Cumulative exclusive horizon: positions [0, 16) runnable now.
    assert token.runnable_num_tokens == 16


def test_preempt_receipt_resume_keeps_sticky_owner_and_restarts_prefill() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)
    request.num_computed_tokens = 8
    key = OwnerLeaseKey("req-0", 0)

    # Preempt through the public reset path: sticky owner+epoch, zero
    # recompute, PREEMPT command, no scheduler KV free.
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert request.attention_owner == 0
    assert request in scheduler.waiting
    out = scheduler.schedule()
    (preempt,) = out.owner_commands
    assert preempt.kind is OwnerCommandKind.PREEMPT
    assert preempt.key == key
    assert out.scheduled_owner_leases == []
    assert "req-0" in out.preempted_req_ids

    _apply_receipts(
        scheduler,
        out,
        {0: [_receipt(preempt.key, 0, preempt.command_seq, runnable=8)]},
    )
    assert scheduler.owner_coordinator.is_preempted(key)
    assert scheduler.owner_coordinator.owner_of(key) == 0

    # Resume: RESERVE with a PREEMPTED descriptor on the sticky owner.
    out2 = scheduler.schedule()
    (resume,) = out2.owner_commands
    assert resume.kind is OwnerCommandKind.RESERVE
    assert resume.key == key
    assert resume.owner_id == 0
    assert resume.allocation.status is OwnerAdmissionStatus.PREEMPTED
    assert resume.allocation.num_computed_tokens == 0
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0

    _apply_receipts(
        scheduler,
        out2,
        {0: [_receipt(resume.key, 0, resume.command_seq, runnable=8)]},
    )
    out3 = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request.attention_owner == 0
    assert request.attention_owner_epoch == 0
    # Resume reacquires the same horizon as the published watermark, so no
    # new lease token publishes until EXTEND passes the watermark.
    assert out3.scheduled_owner_leases == []

    # EXTEND past the resumed horizon publishes new runnable tokens.
    request.num_computed_tokens = 8
    out4 = scheduler.schedule()
    (extend,) = out4.owner_commands
    assert extend.kind is OwnerCommandKind.EXTEND
    assert extend.required_num_tokens == 16
    _apply_receipts(
        scheduler,
        out4,
        {0: [_receipt(extend.key, 0, extend.command_seq, runnable=16)]},
    )
    out5 = scheduler.schedule()
    (token,) = out5.scheduled_owner_leases
    assert token.key == key
    assert token.owner_id == 0
    assert token.runnable_num_tokens == 16


def test_release_is_emitted_and_keeps_liveness_until_accepted_receipt() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)
    key = OwnerLeaseKey("req-0", 0)

    scheduler.finish_requests(["req-0"], RequestStatus.FINISHED_ABORTED)
    assert scheduler._owner_key["req-0"] == key

    # RELEASE is actually emitted on the wire (regression: it used to be
    # recorded as pending but never appended to the outbox).
    out = scheduler.schedule()
    (release,) = out.owner_commands
    assert release.kind is OwnerCommandKind.RELEASE
    assert release.key == key
    assert release.required_num_tokens == 8
    assert out.scheduled_owner_leases == []
    # Liveness survives the step that flushed finished_req_ids.
    assert scheduler.has_requests()

    # A second finish cannot duplicate the outstanding RELEASE.
    scheduler.finish_requests(["req-0"], RequestStatus.FINISHED_ABORTED)
    out2 = scheduler.schedule()
    assert out2.owner_commands == []
    assert scheduler.has_requests()
    assert scheduler.owner_coordinator.release_count() == 0

    # Accepted RELEASE receipt completes the incarnation and restores idle.
    _apply_receipts(
        scheduler,
        out2,
        {0: [_receipt(key, 0, release.command_seq, runnable=8, released=True)]},
    )
    assert scheduler.owner_coordinator.is_released(key)
    assert scheduler.owner_coordinator.release_count() == 1
    assert key not in scheduler._owner_key
    assert not scheduler.has_requests()


def test_refused_release_fails_closed() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)
    key = OwnerLeaseKey("req-0", 0)
    scheduler.finish_requests(["req-0"], RequestStatus.FINISHED_ABORTED)
    out = scheduler.schedule()
    (release,) = out.owner_commands

    # A matching-current rejected RELEASE can never become accepted (the
    # worker fence rejects a duplicate command_seq and the scheduler never
    # re-issues RELEASE), so the receipt fails closed instead of stalling.
    with pytest.raises(RuntimeError, match="RELEASE.*unrecoverable"):
        _apply_receipts(
            scheduler,
            out,
            {0: [_receipt(key, 0, release.command_seq, accepted=False,
                          runnable=None, error="busy")]},
        )


def test_refused_preempt_fails_closed() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)
    scheduler.reset_prefix_cache(reset_running_requests=True)
    out = scheduler.schedule()
    (preempt,) = out.owner_commands
    assert preempt.kind is OwnerCommandKind.PREEMPT

    # Same reasoning as refused RELEASE: a rejected PREEMPT is unrecoverable.
    with pytest.raises(RuntimeError, match="PREEMPT.*unrecoverable"):
        _apply_receipts(
            scheduler,
            out,
            {0: [_receipt(preempt.key, 0, preempt.command_seq, accepted=False,
                          runnable=None, error="not honored")]},
        )


def test_request_id_reuse_fences_to_a_fresh_epoch() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)
    old_key = OwnerLeaseKey("req-0", 0)
    scheduler.finish_requests(["req-0"], RequestStatus.FINISHED_ABORTED)
    out = scheduler.schedule()
    (release,) = out.owner_commands
    _apply_receipts(
        scheduler,
        out,
        {0: [_receipt(old_key, 0, release.command_seq, runnable=8,
                      released=True)]},
    )

    # Same request id is admitted at epoch 1; the old lease is gone.
    request2 = _request("req-0")
    scheduler.add_request(request2)
    out2 = scheduler.schedule()
    (reserve,) = out2.owner_commands
    assert reserve.key == OwnerLeaseKey("req-0", 1)
    assert scheduler.owner_coordinator.is_released(old_key)
    assert scheduler._owner_epoch["req-0"] == 1
    _apply_receipts(
        scheduler,
        out2,
        {0: [_receipt(reserve.key, 0, reserve.command_seq, runnable=8)]},
    )
    out3 = scheduler.schedule()
    assert request2.status == RequestStatus.RUNNING
    (token,) = out3.scheduled_owner_leases
    assert token.key == OwnerLeaseKey("req-0", 1)
    assert token.runnable_num_tokens == 8


def test_zero_token_control_envelope_is_legal() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=0)
    request = _request("req-0")
    scheduler.add_request(request)

    out1 = scheduler.schedule()
    (reserve,) = out1.owner_commands
    assert reserve.kind is OwnerCommandKind.RESERVE
    assert reserve.required_num_tokens == 0
    # Accepted empty lease: runnable 0 is a legal grant, no token published.
    _apply_receipts(
        scheduler,
        out1,
        {0: [_receipt(reserve.key, 0, reserve.command_seq, runnable=0)]},
    )
    out2 = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert scheduler._owner_token_plans["req-0"] == 0
    assert out2.scheduled_owner_leases == []
    # Work remains, so an empty EXTEND is issued at the zero horizon.
    (extend,) = out2.owner_commands
    assert extend.kind is OwnerCommandKind.EXTEND
    assert extend.required_num_tokens == 0


def test_deterministic_owner_assignment() -> None:
    # Zero-required reserves carry no projected charge: all requests tie on
    # committed work and the stable lowest global rank wins.
    scheduler = _make_scheduler(world_size=3, max_num_scheduled_tokens=0)
    a = _request("req-a")
    b = _request("req-b")
    scheduler.add_request(a)
    scheduler.add_request(b)
    out = scheduler.schedule()
    assert [(c.owner_id, c.command_seq) for c in out.owner_commands] == [
        (0, 1),
        (0, 2),
    ]
    assert out.owner_commands == sorted(
        out.owner_commands, key=lambda c: (c.owner_id, c.command_seq)
    )

    # With charged reserves, the second request goes to the least-committed
    # owner (rank 1) instead of piling onto rank 0.
    scheduler2 = _make_scheduler(world_size=3, max_num_scheduled_tokens=64)
    scheduler2.add_request(_request("req-a"))
    scheduler2.add_request(_request("req-b"))
    out2 = scheduler2.schedule()
    owners = [c.owner_id for c in out2.owner_commands]
    assert owners == [0, 1]
    assert out2.owner_commands == sorted(
        out2.owner_commands, key=lambda c: (c.owner_id, c.command_seq)
    )


def test_wire_purity_no_scheduler_kv_apis_and_no_new_request_data() -> None:
    scheduler = _make_scheduler()
    request = _request("req-0")
    scheduler.add_request(request)
    out1 = scheduler.schedule()
    _apply_receipts(
        scheduler,
        out1,
        {0: [_receipt(OwnerLeaseKey("req-0", 0), 0, 1, runnable=32)]},
    )
    out2 = scheduler.schedule()
    for out in (out1, out2):
        assert out.scheduled_new_reqs == []
        assert out.num_scheduled_tokens == {}
        assert out.total_num_scheduled_tokens == 0
        assert out.new_block_ids_to_zero is None
        assert out.num_common_prefix_blocks == []
        assert out.scheduled_spec_decode_tokens == {}
    # new_step_starts / allocate_slots / get_blocks / free /
    # pop_blocks_for_free / take_new_block_ids never ran (the spy raises if
    # any is touched). take_events is the one legitimate output-path call.
    assert scheduler.kv_cache_manager.take_events_calls == 1
    assert request.status == RequestStatus.RUNNING
    (token,) = out2.scheduled_owner_leases
    assert token.runnable_num_tokens == 32
