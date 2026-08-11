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

import json
from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.owner_layout_probe import OwnerLayoutProbe
from vllm.v1.core.sched.ownership import (
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerCachePoolSnapshot,
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
            raise AssertionError(f"kv_cache_manager.{name}() must never run in G2 mode")
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
    # MRV1 logical-token state (the request-owned path preserves it).
    scheduler.use_v2_model_runner = False
    scheduler.use_pp = False
    # G2/G3 control-plane state (normally initialized at the tail of
    # __init__).
    scheduler.owner_coordinator = None
    scheduler._owner_key = {}
    scheduler._owner_epoch = {}
    scheduler._owner_pending_command = {}
    scheduler._owner_emitted_command_seq = {}
    scheduler._owner_outbox = []
    scheduler._owner_token_plans = {}
    scheduler._owner_pending_dispatch = {}
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


def _pool(
    owner_rank: int,
    total_blocks: int = 1000,
    free_blocks: int = 300,
) -> OwnerCachePoolSnapshot:
    return OwnerCachePoolSnapshot(
        owner_rank=owner_rank,
        total_blocks=total_blocks,
        free_blocks=free_blocks,
    )


def _apply_pool_receipts(
    scheduler,
    scheduler_output,
    pools_by_rank,
    events_by_rank=None,
) -> None:
    """Feed an all-worker receipt envelope with per-rank pool snapshots."""
    batches = [
        OwnerReceiptBatch(
            owner_rank=rank,
            emitted_step_seq=scheduler_output.step_seq,
            events=tuple((events_by_rank or {}).get(rank, ())),
            cache_pool=pools_by_rank.get(rank),
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
        {
            command.owner_id: [
                _receipt(
                    command.key, command.owner_id, command.command_seq, runnable=grant
                )
            ]
        },
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


def test_layout_probe_records_worker_confirmed_post_step_capacity(tmp_path) -> None:
    scheduler = _make_scheduler()
    scheduler._owner_layout_probe = OwnerLayoutProbe(
        path=tmp_path / "layout.jsonl",
        run_id="post-step-cell",
        world_size=2,
        max_records=4,
        max_bytes=4096,
    )
    request = _request("req-0")
    scheduler.add_request(request)

    reserve_step = scheduler.schedule()
    # schedule() only freezes the plan; it must not pair that plan with the
    # stale pool snapshot from the preceding worker step.
    assert len((tmp_path / "layout.jsonl").read_text().splitlines()) == 1
    (command,) = reserve_step.owner_commands
    _apply_pool_receipts(
        scheduler,
        reserve_step,
        {
            0: _pool(0, total_blocks=1000, free_blocks=900),
            1: _pool(1, total_blocks=1000, free_blocks=800),
        },
        {
            command.owner_id: [
                _receipt(
                    command.key,
                    command.owner_id,
                    command.command_seq,
                    runnable=32,
                )
            ]
        },
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "layout.jsonl").read_text().splitlines()
    ]
    assert records[1]["step_seq"] == reserve_step.step_seq
    assert records[1]["total_scheduled_tokens"] == 0
    assert [pool["free_blocks"] for pool in records[1]["owner_cache_pools"]] == [
        900,
        800,
    ]


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
    _apply_receipts(
        scheduler,
        out1,
        {0: [_receipt(command.key, 0, command.command_seq, runnable=32)]},
    )
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
        {
            0: [
                _receipt(
                    key,
                    0,
                    command.command_seq,
                    accepted=False,
                    runnable=None,
                    error="insufficient capacity",
                )
            ]
        },
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
        {
            0: [
                _receipt(
                    extend.key,
                    0,
                    extend.command_seq,
                    accepted=False,
                    runnable=None,
                    error="no capacity",
                )
            ]
        },
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
    # Resume reacquires the same horizon as the published watermark, so the
    # resumed first step is a token-bearing decode that re-publishes the
    # authorization at the unchanged horizon.
    assert out3.num_scheduled_tokens == {"req-0": 8}
    assert out3.total_num_scheduled_tokens == 8
    assert out3.scheduled_new_reqs == []
    assert out3.scheduled_cached_reqs.req_ids == ["req-0"]
    assert "req-0" in out3.scheduled_cached_reqs.resumed_req_ids
    # MRV1 _update_states requires a non-None reset for resumed requests.
    assert out3.scheduled_cached_reqs.new_block_ids == [()]
    assert out3.scheduled_cached_reqs.all_token_ids == {"req-0": list(range(32))}
    (resumed_token,) = out3.scheduled_owner_leases
    assert resumed_token.key == key
    assert resumed_token.owner_id == 0
    assert resumed_token.step_seq == 5
    assert resumed_token.runnable_num_tokens == 8
    assert request.num_computed_tokens == 8

    # EXTEND past the resumed horizon publishes new runnable tokens.
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
            {
                0: [
                    _receipt(
                        key,
                        0,
                        release.command_seq,
                        accepted=False,
                        runnable=None,
                        error="busy",
                    )
                ]
            },
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
            {
                0: [
                    _receipt(
                        preempt.key,
                        0,
                        preempt.command_seq,
                        accepted=False,
                        runnable=None,
                        error="not honored",
                    )
                ]
            },
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
        {0: [_receipt(old_key, 0, release.command_seq, runnable=8, released=True)]},
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
    # Before any physical snapshot, first admission balances by live/
    # provisional lease count then stable rank: the second request leaves
    # rank 0 even when its RESERVE carries no projected charge.
    scheduler = _make_scheduler(world_size=3, max_num_scheduled_tokens=0)
    a = _request("req-a")
    b = _request("req-b")
    scheduler.add_request(a)
    scheduler.add_request(b)
    out = scheduler.schedule()
    assert [(c.owner_id, c.command_seq) for c in out.owner_commands] == [
        (0, 1),
        (1, 1),
    ]
    assert out.owner_commands == sorted(
        out.owner_commands, key=lambda c: (c.owner_id, c.command_seq)
    )

    # With charged reserves, lease-count balancing gives the same owners.
    scheduler2 = _make_scheduler(world_size=3, max_num_scheduled_tokens=64)
    scheduler2.add_request(_request("req-a"))
    scheduler2.add_request(_request("req-b"))
    out2 = scheduler2.schedule()
    owners = [c.owner_id for c in out2.owner_commands]
    assert owners == [0, 1]
    assert out2.owner_commands == sorted(
        out2.owner_commands, key=lambda c: (c.owner_id, c.command_seq)
    )


def test_wire_purity_no_scheduler_kv_apis_and_empty_block_ids() -> None:
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
    # The RESERVE step is a command-only zero-token heartbeat.
    assert out1.scheduled_new_reqs == []
    assert out1.num_scheduled_tokens == {}
    assert out1.total_num_scheduled_tokens == 0
    assert out1.scheduled_owner_leases == []
    assert out1.new_block_ids_to_zero is None
    assert out1.num_common_prefix_blocks == []
    assert out1.scheduled_spec_decode_tokens == {}
    # The promotion step carries the first logical token payload with empty
    # block-ID fields and exactly one authorization lease.
    assert out2.total_num_scheduled_tokens == 32
    assert out2.num_scheduled_tokens == {"req-0": 32}
    assert out2.new_block_ids_to_zero is None
    assert out2.num_common_prefix_blocks == []
    assert out2.scheduled_spec_decode_tokens == {}
    (new_data,) = out2.scheduled_new_reqs
    assert new_data.req_id == "req-0"
    assert new_data.block_ids == ()
    assert new_data.num_computed_tokens == 0
    assert new_data.attention_owner == 0
    assert out2.scheduled_cached_reqs.req_ids == []
    (token,) = out2.scheduled_owner_leases
    assert token.key == OwnerLeaseKey("req-0", 0)
    assert token.step_seq == 2
    assert token.runnable_num_tokens == 32
    # new_step_starts / allocate_slots / get_blocks / free /
    # pop_blocks_for_free / take_new_block_ids never ran (the spy raises if
    # any is touched). take_events is the one legitimate output-path call.
    assert scheduler.kv_cache_manager.take_events_calls == 1
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 32
    # MRV1: the first-dispatch request is recorded for the next step.
    assert scheduler.prev_step_scheduled_req_ids == {"req-0"}


def test_no_snapshot_admission_balances_by_live_lease_count() -> None:
    scheduler = _make_scheduler(world_size=3)
    for request_id in ("req-a", "req-b", "req-c", "req-d"):
        scheduler.add_request(_request(request_id))
    out = scheduler.schedule()
    # Before any physical snapshot, first admission balances on the live/
    # provisional lease count and then stable rank.  The wire drains in
    # per-owner command order, so assert the assignment map itself.
    assert [c.owner_id for c in out.owner_commands] == [0, 0, 1, 2]
    for request_id, owner in (("req-a", 0), ("req-b", 1), ("req-c", 2), ("req-d", 0)):
        assert (
            scheduler.owner_coordinator.owner_of(OwnerLeaseKey(request_id, 0)) == owner
        )


def test_greater_free_blocks_wins_regardless_of_token_count() -> None:
    scheduler = _make_scheduler()
    out0 = scheduler.schedule()
    _apply_pool_receipts(
        scheduler,
        out0,
        {
            0: _pool(0, total_blocks=1000, free_blocks=100),
            1: _pool(1, total_blocks=1000, free_blocks=900),
        },
    )
    # A long and a short request both land on the emptiest pool (rank 1):
    # free blocks dominate, never token counts.
    scheduler.add_request(_request("req-a", num_prompt_tokens=256))
    scheduler.add_request(_request("req-b", num_prompt_tokens=16))
    out1 = scheduler.schedule()
    assert [c.owner_id for c in out1.owner_commands] == [1, 1]


def test_same_step_admissions_balance_on_provisional_count_under_equal_free() -> None:
    scheduler = _make_scheduler(world_size=3)
    out0 = scheduler.schedule()
    _apply_pool_receipts(
        scheduler,
        out0,
        {
            0: _pool(0, free_blocks=500),
            1: _pool(1, free_blocks=500),
            2: _pool(2, free_blocks=500),
        },
    )
    for request_id in ("req-a", "req-b", "req-c", "req-d"):
        scheduler.add_request(_request(request_id))
    out1 = scheduler.schedule()
    # Equal free blocks: provisional choices made earlier in this same
    # admission pass break the tie, then stable rank.  The wire drains in
    # per-owner command order, so assert the assignment map itself.
    assert [c.owner_id for c in out1.owner_commands] == [0, 0, 1, 2]
    for request_id, owner in (("req-a", 0), ("req-b", 1), ("req-c", 2), ("req-d", 0)):
        assert (
            scheduler.owner_coordinator.owner_of(OwnerLeaseKey(request_id, 0)) == owner
        )


def test_partial_snapshot_envelope_fails_closed() -> None:
    scheduler = _make_scheduler()
    out = scheduler.schedule()
    with pytest.raises(RuntimeError, match="cache_pool snapshot"):
        _apply_pool_receipts(scheduler, out, {0: _pool(0, free_blocks=500)})


def test_missing_snapshot_after_first_fails_closed() -> None:
    scheduler = _make_scheduler()
    out0 = scheduler.schedule()
    _apply_pool_receipts(
        scheduler,
        out0,
        {0: _pool(0, free_blocks=500), 1: _pool(1, free_blocks=500)},
    )
    # Once physical snapshots are in use, an all-None envelope is stale
    # facts, not a legacy control step.
    out1 = scheduler.schedule()
    with pytest.raises(RuntimeError, match="all-None envelope"):
        _apply_receipts(scheduler, out1, {})


def test_observations_reflect_physical_facts_without_ids() -> None:
    scheduler = _make_scheduler()
    out0 = scheduler.schedule()
    _apply_pool_receipts(
        scheduler,
        out0,
        {
            0: _pool(0, total_blocks=1000, free_blocks=300),
            1: _pool(1, total_blocks=2000, free_blocks=800),
        },
    )
    out1 = scheduler.schedule()
    by_owner = {o.owner_id: o for o in out1.owner_assignment_observations}
    # work = total_blocks - free_blocks; residency/pending_dma stay 0.
    assert by_owner[0].work == 700
    assert by_owner[1].work == 1200
    assert by_owner[0].residency == 0
    assert by_owner[0].pending_dma == 0
    # Stored snapshots are block-ID-free facts per rank.
    for rank, free in ((0, 300), (1, 800)):
        snapshot = scheduler._owner_pool_snapshots[rank]
        assert snapshot.owner_rank == rank
        assert snapshot.free_blocks == free
        assert not hasattr(snapshot, "block_id")


def test_sticky_owner_unaffected_by_later_pool_imbalance() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    # Rank 0 has the emptiest pool at first admission.
    out0 = scheduler.schedule()
    _apply_pool_receipts(
        scheduler,
        out0,
        {0: _pool(0, free_blocks=800), 1: _pool(1, free_blocks=200)},
    )
    scheduler.add_request(request)
    out1 = scheduler.schedule()
    (reserve,) = out1.owner_commands
    assert reserve.owner_id == 0
    _apply_pool_receipts(
        scheduler,
        out1,
        {0: _pool(0, free_blocks=800), 1: _pool(1, free_blocks=200)},
        {0: [_receipt(reserve.key, 0, reserve.command_seq, runnable=8)]},
    )
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request.attention_owner == 0

    # Preempt, then flip the pools so rank 1 is by far the emptiest.
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    out3 = scheduler.schedule()
    (preempt,) = out3.owner_commands
    assert preempt.kind is OwnerCommandKind.PREEMPT
    _apply_pool_receipts(
        scheduler,
        out3,
        {
            0: _pool(0, total_blocks=1000, free_blocks=50),
            1: _pool(1, total_blocks=1000, free_blocks=950),
        },
        {0: [_receipt(preempt.key, 0, preempt.command_seq, runnable=8)]},
    )
    # Resume stays on the sticky owner despite the later imbalance.
    out4 = scheduler.schedule()
    (resume,) = out4.owner_commands
    assert resume.kind is OwnerCommandKind.RESERVE
    assert resume.owner_id == 0
    assert request.attention_owner == 0


def test_wrong_owner_receipt_does_not_clear_pending_command() -> None:
    scheduler = _make_scheduler()
    request = _request("req-0")
    scheduler.add_request(request)
    out1 = scheduler.schedule()
    (command,) = out1.owner_commands
    assert command.owner_id == 0

    # A wrong-owner batch event (rank 1 echoes key/seq) must not clear the
    # real in-flight command: the coordinator ignores it and the pending
    # state survives so the genuine receipt can still land.
    _apply_receipts(
        scheduler,
        out1,
        {1: [_receipt(command.key, 1, command.command_seq, runnable=8)]},
    )
    assert scheduler._owner_pending_command["req-0"] == (
        command.command_seq,
        OwnerCommandKind.RESERVE,
    )
    assert scheduler.owner_coordinator.owner_of(command.key) == 0

    # The genuine owner-0 receipt still promotes the request.
    _apply_receipts(
        scheduler,
        out1,
        {0: [_receipt(command.key, 0, command.command_seq, runnable=8)]},
    )
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request.attention_owner == 0


# ---------------------------------------------------------------------------
# G3 scheduler slice: budgeted logical token payload + per-step authorizations
# ---------------------------------------------------------------------------


def test_aggregate_budget_freezes_plan_in_running_order() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=48)
    a = _request("req-a")
    b = _request("req-b")
    scheduler.add_request(a)
    scheduler.add_request(b)

    out1 = scheduler.schedule()
    commands = {c.key.request_id: c for c in out1.owner_commands}
    assert set(commands) == {"req-a", "req-b"}
    _apply_receipts(
        scheduler,
        out1,
        {
            c.owner_id: [_receipt(c.key, c.owner_id, c.command_seq, runnable=32)]
            for c in commands.values()
        },
    )
    # Both requests promote in RUNNING (admission) order; the global budget
    # freezes the plan: req-a keeps its full 32, req-b takes the remainder.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens == {"req-a": 32, "req-b": 16}
    assert out2.total_num_scheduled_tokens == 48
    assert [d.req_id for d in out2.scheduled_new_reqs] == ["req-a", "req-b"]
    for data in out2.scheduled_new_reqs:
        assert data.block_ids == ()
    # Exactly one authorization lease per scheduled key, in key order.
    assert [
        (t.key.request_id, t.runnable_num_tokens) for t in out2.scheduled_owner_leases
    ] == [("req-a", 32), ("req-b", 32)]
    assert scheduler._owner_token_plans == {"req-a": 32, "req-b": 16}

    # Decode continuation with the horizon unchanged re-publishes exactly
    # the scheduled key (req-b), at the same count, via the cached payload.
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens == {"req-b": 16}
    assert out3.scheduled_new_reqs == []
    assert out3.scheduled_cached_reqs.req_ids == ["req-b"]
    assert out3.scheduled_cached_reqs.resumed_req_ids == set()
    assert out3.scheduled_cached_reqs.new_block_ids == [None]
    # req-b was scheduled last step (MRV1): full token ids are not resent.
    assert out3.scheduled_cached_reqs.all_token_ids == {}
    assert [
        (t.key.request_id, t.runnable_num_tokens) for t in out3.scheduled_owner_leases
    ] == [("req-b", 32)]
    assert scheduler.prev_step_scheduled_req_ids == {"req-b"}


def test_zero_budget_promotion_remains_new_until_first_dispatch() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=32)
    requests = [_request(f"req-{index}", num_prompt_tokens=64) for index in range(3)]
    for request in requests:
        scheduler.add_request(request)

    reserve_step = scheduler.schedule()
    commands = {
        command.key.request_id: command
        for command in reserve_step.owner_commands
    }
    assert set(commands) == {"req-0", "req-1", "req-2"}
    events_by_rank = {}
    for command in commands.values():
        events_by_rank.setdefault(command.owner_id, []).append(
            _receipt(
                command.key,
                command.owner_id,
                command.command_seq,
                runnable=32,
            )
        )
    _apply_receipts(scheduler, reserve_step, events_by_rank)

    # All three leases promote, but the global budget dispatches only req-0.
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"req-0": 32}
    assert [data.req_id for data in first.scheduled_new_reqs] == ["req-0"]
    assert scheduler._owner_pending_dispatch == {
        "req-1": RequestStatus.WAITING,
        "req-2": RequestStatus.WAITING,
    }

    # req-0 now needs an EXTEND.  req-1 receives its first positive budget
    # one schedule later and must still be NewRequestData, never cached data.
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"req-1": 32}
    assert [data.req_id for data in second.scheduled_new_reqs] == ["req-1"]
    assert second.scheduled_cached_reqs.req_ids == []
    assert [command.key.request_id for command in second.owner_commands] == ["req-0"]
    assert scheduler._owner_pending_dispatch == {
        "req-2": RequestStatus.WAITING,
    }


def test_command_only_extend_step_excludes_payload_and_leases() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)
    request.append_output_token_ids([42])
    # computed == horizon with work remaining: EXTEND step is command-only.
    out = scheduler.schedule()
    assert request.num_computed_tokens == 8
    (extend,) = out.owner_commands
    assert extend.kind is OwnerCommandKind.EXTEND
    assert out.num_scheduled_tokens == {}
    assert out.total_num_scheduled_tokens == 0
    assert out.scheduled_new_reqs == []
    assert out.scheduled_cached_reqs.req_ids == []
    assert out.scheduled_owner_leases == []
    # No same-key command + execution token in one step.
    assert [c.key for c in out.owner_commands] == [OwnerLeaseKey("req-0", 0)]


def test_first_dispatch_then_cached_continuation_preserves_mrv1_state() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=16)
    request = _request("req-0", num_prompt_tokens=64)
    scheduler.add_request(request)
    _admit(scheduler, request, grant=16)
    # First dispatch: NewRequestData with empty block IDs.
    assert scheduler.prev_step_scheduled_req_ids == {"req-0"}
    assert request.num_computed_tokens == 16

    # At the horizon with new output work: EXTEND command-only step.
    request.append_output_token_ids([42])
    out_ext = scheduler.schedule()
    (extend,) = out_ext.owner_commands
    assert extend.kind is OwnerCommandKind.EXTEND
    assert out_ext.num_scheduled_tokens == {}
    assert out_ext.scheduled_owner_leases == []
    _apply_receipts(
        scheduler,
        out_ext,
        {0: [_receipt(extend.key, 0, extend.command_seq, runnable=32)]},
    )

    # Decode continuation: cached payload, full token ids only on first
    # appearance (never on a continuation).
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"req-0": 16}
    assert out.scheduled_new_reqs == []
    assert out.scheduled_cached_reqs.req_ids == ["req-0"]
    assert out.scheduled_cached_reqs.resumed_req_ids == set()
    assert out.scheduled_cached_reqs.new_block_ids == [None]
    # MRV1: after a command-only gap step the persistent batch was not
    # scheduled, so full token ids are re-propagated on the next step.
    assert out.scheduled_cached_reqs.all_token_ids == {"req-0": list(range(64)) + [42]}
    assert out.scheduled_cached_reqs.num_computed_tokens == [16]
    (token,) = out.scheduled_owner_leases
    assert token.key == OwnerLeaseKey("req-0", 0)
    assert token.runnable_num_tokens == 32
    assert request.num_computed_tokens == 32
    assert scheduler.prev_step_scheduled_req_ids == {"req-0"}


def test_resume_resets_new_block_ids_empty_while_continuation_keeps_none() -> None:
    scheduler = _make_scheduler(max_num_scheduled_tokens=8)
    request = _request("req-0")
    scheduler.add_request(request)
    _admit(scheduler, request, grant=8)

    # Ordinary decode continuation: None (no new blocks to append).
    request.append_output_token_ids([42])
    out_ext = scheduler.schedule()
    (extend,) = out_ext.owner_commands
    assert extend.kind is OwnerCommandKind.EXTEND
    _apply_receipts(
        scheduler,
        out_ext,
        {0: [_receipt(extend.key, 0, extend.command_seq, runnable=16)]},
    )
    out = scheduler.schedule()
    assert out.scheduled_cached_reqs.req_ids == ["req-0"]
    assert out.scheduled_cached_reqs.resumed_req_ids == set()
    assert out.scheduled_cached_reqs.new_block_ids == [None]

    # PREEMPTED resume: non-None empty reset () so the MRV1 runner can
    # replace the cached block ids with an ID-free empty set.
    scheduler.reset_prefix_cache(reset_running_requests=True)
    out_preempt = scheduler.schedule()
    (preempt,) = out_preempt.owner_commands
    _apply_receipts(
        scheduler,
        out_preempt,
        {0: [_receipt(preempt.key, 0, preempt.command_seq, runnable=16)]},
    )
    out_resume = scheduler.schedule()
    (resume,) = out_resume.owner_commands
    _apply_receipts(
        scheduler,
        out_resume,
        {0: [_receipt(resume.key, 0, resume.command_seq, runnable=16)]},
    )
    out_restart = scheduler.schedule()
    assert out_restart.scheduled_cached_reqs.req_ids == ["req-0"]
    assert "req-0" in out_restart.scheduled_cached_reqs.resumed_req_ids
    assert out_restart.scheduled_cached_reqs.new_block_ids == [()]
