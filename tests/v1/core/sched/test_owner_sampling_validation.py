# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused tests for the G3 scheduler-side owner-sampling envelope validation.

CPU-only: exercises the pure ``Scheduler._validate_owner_sampling_envelope``
validator and the pre-mutation call seam in ``update_from_output`` through a
minimal ``Scheduler.__new__`` harness (a real Scheduler needs a full engine
stack), mirroring ``test_scheduler_owner_admission.py``.
"""

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.owner_layout import GlobalRowId
from vllm.v1.core.sched.ownership import (
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
from vllm.v1.outputs import ModelRunnerOutput, OwnerSamplingBatch
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    request_id: str,
    owner_epoch: int,
    position: int,
    lane: int = 0,
) -> GlobalRowId:
    return GlobalRowId(
        OwnerLeaseKey(request_id=request_id, owner_epoch=owner_epoch),
        position,
        lane,
    )


def _merged_output(
    req_ids: list[str],
    sampled_token_ids: list[list[int]] | None = None,
) -> ModelRunnerOutput:
    if sampled_token_ids is None:
        sampled_token_ids = [[1] for _ in req_ids]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: idx for idx, rid in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
    )


def _valid_envelope_args(*, step: int = 7) -> dict:
    """A contract-valid envelope: three owners, ragged rows, chunked n>1.

    Terminal rows (pre-step computed + scheduled count - 1, lane 0):
    req-0: 10 + 4 - 1 = 13; req-1: 20 + 1 - 1 = 20; req-2: 0 + 2 - 1 = 1.
    """
    return dict(
        scheduler_step_seq=step,
        num_scheduled_tokens={"req-0": 4, "req-1": 1, "req-2": 2},
        pre_step_num_computed_tokens={"req-0": 10, "req-1": 20, "req-2": 0},
        authoritative_owner_by_request_id={"req-0": 2, "req-1": 0, "req-2": 1},
        authoritative_epoch_by_request_id={"req-0": 1, "req-1": 3, "req-2": 0},
        owner_sampling_batches=[
            OwnerSamplingBatch(
                owner_rank=0,
                emitted_step_seq=step,
                row_ids=(_row("req-1", 3, 20),),
            ),
            OwnerSamplingBatch(
                owner_rank=1,
                emitted_step_seq=step,
                row_ids=(_row("req-2", 0, 1),),
            ),
            OwnerSamplingBatch(
                owner_rank=2,
                emitted_step_seq=step,
                row_ids=(_row("req-0", 1, 13),),
            ),
        ],
        model_runner_output=_merged_output(
            ["req-1", "req-2", "req-0"], [[7], [], [3]]
        ),
    )


def _expect_invalid(args: dict, needle: str) -> None:
    with pytest.raises(RuntimeError, match=needle):
        Scheduler._validate_owner_sampling_envelope(**args)


class _KVCacheSpy:
    """KV cache manager stub that fails closed on any KV-block API."""

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
    sampling_enabled: bool = False,
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler_config = SimpleNamespace(
        enable_request_owned_attention=True,
        enable_request_owned_sampling=sampling_enabled,
    )
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
    scheduler.use_v2_model_runner = False
    scheduler.use_pp = False
    scheduler.owner_coordinator = None
    scheduler._owner_key = {}
    scheduler._owner_epoch = {}
    scheduler._owner_pending_command = {}
    scheduler._owner_emitted_command_seq = {}
    scheduler._owner_outbox = []
    scheduler._owner_token_plans = {}
    scheduler._owner_promoted = {}
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


def _receipt(
    key: OwnerLeaseKey,
    owner_id: int,
    command_seq: int,
    *,
    runnable: int | None = None,
) -> OwnerReceipt:
    return OwnerReceipt(
        key=key,
        owner_id=owner_id,
        command_seq=command_seq,
        accepted=True,
        runnable_num_tokens=runnable,
        released=False,
        error=None,
    )


def _apply_receipts(scheduler, scheduler_output, events_by_rank) -> None:
    batches = [
        OwnerReceiptBatch(
            owner_rank=rank,
            emitted_step_seq=scheduler_output.step_seq,
            events=tuple(events_by_rank.get(rank, ())),
        )
        for rank in range(scheduler.parallel_config.world_size)
    ]
    sampling_batches = None
    if scheduler.scheduler_config.enable_request_owned_sampling:
        sampling_batches = [
            OwnerSamplingBatch(
                owner_rank=rank,
                emitted_step_seq=scheduler_output.step_seq,
            )
            for rank in range(scheduler.parallel_config.world_size)
        ]
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            owner_receipt_batches=batches,
            owner_sampling_batches=sampling_batches,
        ),
    )


def _admit(scheduler, request, *, grant: int):
    """Admit ``request``: RESERVE step, accepted receipt, promote step."""
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


# ---------------------------------------------------------------------------
# Pure validator: contract-valid envelopes
# ---------------------------------------------------------------------------


def test_valid_mixed_owners_chunked_terminal_rows_accepted() -> None:
    Scheduler._validate_owner_sampling_envelope(**_valid_envelope_args())


def test_valid_empty_owner_batch_allowed() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"] = [
        OwnerSamplingBatch(owner_rank=0, emitted_step_seq=7, row_ids=()),
        OwnerSamplingBatch(owner_rank=1, emitted_step_seq=7, row_ids=()),
        OwnerSamplingBatch(owner_rank=2, emitted_step_seq=7, row_ids=()),
    ]
    # Every expected owner may carry an empty batch: an empty envelope only
    # validates if no request is scheduled on this step.
    args["num_scheduled_tokens"] = {}
    args["pre_step_num_computed_tokens"] = {}
    args["authoritative_owner_by_request_id"] = {}
    args["authoritative_epoch_by_request_id"] = {}
    args["model_runner_output"] = _merged_output([], [])
    Scheduler._validate_owner_sampling_envelope(**args)


def test_valid_empty_token_list_for_discarded_request() -> None:
    args = _valid_envelope_args()
    args["model_runner_output"] = _merged_output(
        ["req-1", "req-2", "req-0"], [[], [], [3]]
    )
    Scheduler._validate_owner_sampling_envelope(**args)


def test_valid_zero_token_heartbeat_with_empty_envelope() -> None:
    args = dict(
        scheduler_step_seq=7,
        num_scheduled_tokens={},
        pre_step_num_computed_tokens={},
        authoritative_owner_by_request_id={},
        authoritative_epoch_by_request_id={},
        owner_sampling_batches=[
            OwnerSamplingBatch(owner_rank=0, emitted_step_seq=7),
            OwnerSamplingBatch(owner_rank=1, emitted_step_seq=7),
        ],
        model_runner_output=_merged_output([], []),
    )
    Scheduler._validate_owner_sampling_envelope(**args)


# ---------------------------------------------------------------------------
# Pure validator: contract violations
# ---------------------------------------------------------------------------


def test_rejects_wrong_step() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"][0] = OwnerSamplingBatch(
        owner_rank=0, emitted_step_seq=8, row_ids=(_row("req-1", 3, 20),)
    )
    _expect_invalid(args, "emitted_step_seq")


def test_rejects_wrong_owner_rank() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"][0] = OwnerSamplingBatch(
        owner_rank=1, emitted_step_seq=7, row_ids=(_row("req-1", 3, 20),)
    )
    _expect_invalid(args, "owner 0.*owner 1")


def test_rejects_stale_lease_epoch() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"][0] = OwnerSamplingBatch(
        owner_rank=0, emitted_step_seq=7, row_ids=(_row("req-1", 2, 20),)
    )
    _expect_invalid(args, "lease epoch 2, expected 3")


def test_rejects_wrong_terminal_position() -> None:
    # One row per scheduled token (position pre + count - 2) fails.
    args = _valid_envelope_args()
    args["owner_sampling_batches"][2] = OwnerSamplingBatch(
        owner_rank=2, emitted_step_seq=7, row_ids=(_row("req-0", 1, 12),)
    )
    _expect_invalid(args, "pre-step computed 10 \\+ scheduled 4 - 1 = 13")


def test_rejects_nonzero_lane() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"][2] = OwnerSamplingBatch(
        owner_rank=2, emitted_step_seq=7, row_ids=(_row("req-0", 1, 13, lane=1),)
    )
    _expect_invalid(args, "lane 1 must be 0")


def test_rejects_missing_scheduled_request() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"] = [
        OwnerSamplingBatch(
            owner_rank=2, emitted_step_seq=7, row_ids=(_row("req-0", 1, 13),)
        )
    ]
    args["model_runner_output"] = _merged_output(["req-0"], [[3]])
    _expect_invalid(args, "missing sampling identity.*req-1.*req-2")


def test_rejects_extra_unknown_request() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"].append(
        OwnerSamplingBatch(
            owner_rank=2, emitted_step_seq=7, row_ids=(_row("req-3", 0, 5),)
        )
    )
    _expect_invalid(args, "unscheduled/unknown request.*req-3")


def test_rejects_duplicate_request_across_batches() -> None:
    args = _valid_envelope_args()
    args["owner_sampling_batches"].append(
        OwnerSamplingBatch(
            owner_rank=2, emitted_step_seq=7, row_ids=(_row("req-1", 3, 20),)
        )
    )
    _expect_invalid(args, "duplicate request 'req-1'")


def test_rejects_unknown_or_finished_request() -> None:
    args = _valid_envelope_args()
    args["authoritative_owner_by_request_id"] = {"req-0": 2, "req-2": 1}
    args["authoritative_epoch_by_request_id"] = {"req-0": 1, "req-2": 0}
    _expect_invalid(args, "req-1.*unknown or finished")


def test_rejects_missing_pre_step_snapshot() -> None:
    args = _valid_envelope_args()
    args["pre_step_num_computed_tokens"] = {"req-0": 10, "req-1": 20}
    _expect_invalid(args, "no pre-step num_computed_tokens snapshot.*req-2")


def test_rejects_merged_req_set_mismatch() -> None:
    args = _valid_envelope_args()
    args["model_runner_output"] = _merged_output(["req-1", "req-0"], [[7], [3]])
    _expect_invalid(args, "does not match the envelope request set")


def test_rejects_merged_req_map_not_bijective() -> None:
    args = _valid_envelope_args()
    output = args["model_runner_output"]
    args["model_runner_output"] = ModelRunnerOutput(
        req_ids=output.req_ids,
        req_id_to_index={"req-1": 0, "req-2": 2, "req-0": 1},
        sampled_token_ids=output.sampled_token_ids,
    )
    _expect_invalid(args, "not bijective")


def test_rejects_merged_req_map_extra_key() -> None:
    args = _valid_envelope_args()
    output = args["model_runner_output"]
    args["model_runner_output"] = ModelRunnerOutput(
        req_ids=output.req_ids,
        req_id_to_index={"req-1": 0, "req-2": 1, "req-0": 2, "req-9": 3},
        sampled_token_ids=output.sampled_token_ids,
    )
    _expect_invalid(args, "not bijective")


def test_rejects_multi_token_sampled_entry() -> None:
    args = _valid_envelope_args()
    args["model_runner_output"] = _merged_output(
        ["req-1", "req-2", "req-0"], [[7, 8], [], [3]]
    )
    _expect_invalid(args, "multi-token")


def test_rejects_misaligned_sampled_token_ids() -> None:
    args = _valid_envelope_args()
    args["model_runner_output"] = _merged_output(
        ["req-1", "req-2", "req-0"], [[7], [3]]
    )
    _expect_invalid(args, "aligned 1:1")


def test_rejects_zero_token_heartbeat_with_rows() -> None:
    args = dict(
        scheduler_step_seq=7,
        num_scheduled_tokens={},
        pre_step_num_computed_tokens={},
        authoritative_owner_by_request_id={},
        authoritative_epoch_by_request_id={},
        owner_sampling_batches=[
            OwnerSamplingBatch(
                owner_rank=0, emitted_step_seq=7, row_ids=(_row("req-1", 3, 20),)
            )
        ],
        model_runner_output=_merged_output([], []),
    )
    _expect_invalid(args, "zero-token heartbeat")


def test_rejects_zero_token_heartbeat_with_merged_requests() -> None:
    args = dict(
        scheduler_step_seq=7,
        num_scheduled_tokens={},
        pre_step_num_computed_tokens={},
        authoritative_owner_by_request_id={},
        authoritative_epoch_by_request_id={},
        owner_sampling_batches=[
            OwnerSamplingBatch(owner_rank=0, emitted_step_seq=7),
        ],
        model_runner_output=_merged_output(["req-1"], [[7]]),
    )
    _expect_invalid(args, "does not match the envelope request set")


# ---------------------------------------------------------------------------
# Seam: update_from_output call boundary on the __new__ harness
# ---------------------------------------------------------------------------


def test_seam_default_off_output_preserves_control_only_step() -> None:
    scheduler = _make_scheduler()
    out = scheduler.schedule()
    assert out.total_num_scheduled_tokens == 0
    # No owner_sampling_batches (default-off output): the control-only step
    # round-trips exactly as before this boundary existed.
    _apply_receipts(scheduler, out, {})
    assert scheduler.owner_coordinator is not None


def test_seam_accepts_token_bearing_envelope_without_mutation() -> None:
    scheduler = _make_scheduler(
        max_num_scheduled_tokens=64, sampling_enabled=True
    )
    request = _request("req-0")
    scheduler.add_request(request)
    out = _admit(scheduler, request, grant=8)

    req_id = request.request_id
    count = out.num_scheduled_tokens[req_id]
    # The schedule() call already advanced request.num_computed_tokens; the
    # validation must use the pre-step snapshot (0) carried by the
    # scheduler_output payload, not the post-mutation request state.
    (pre,) = [new_req.num_computed_tokens for new_req in out.scheduled_new_reqs]
    assert pre == 0
    assert request.num_computed_tokens == count
    owner_rank = scheduler.owner_coordinator.owner_of(
        scheduler._owner_key[req_id]
    )
    assert owner_rank is not None

    before = _scheduler_state_snapshot(scheduler, request)
    scheduler._validate_request_owned_sampling_envelope(
        out,
        ModelRunnerOutput(
            req_ids=[req_id],
            req_id_to_index={req_id: 0},
            sampled_token_ids=[[123]],
            owner_sampling_batches=[
                OwnerSamplingBatch(
                    owner_rank=owner_rank,
                    emitted_step_seq=out.step_seq,
                    row_ids=(_row(req_id, 0, count - 1),),
                )
            ],
        ),
    )
    # Validation is pure: nothing observable changed.
    assert _scheduler_state_snapshot(scheduler, request) == before


def test_seam_rejects_invalid_envelope_before_any_mutation() -> None:
    scheduler = _make_scheduler(
        max_num_scheduled_tokens=64, sampling_enabled=True
    )
    request = _request("req-0")
    scheduler.add_request(request)
    out = _admit(scheduler, request, grant=8)

    req_id = request.request_id
    owner_rank = scheduler.owner_coordinator.owner_of(
        scheduler._owner_key[req_id]
    )
    assert owner_rank is not None
    count = out.num_scheduled_tokens[req_id]
    before = _scheduler_state_snapshot(scheduler, request)

    # Wrong terminal position (one row per scheduled token) must fail from
    # the pre-mutation seam even though the receipt envelope is valid.
    (pre,) = [new_req.num_computed_tokens for new_req in out.scheduled_new_reqs]
    with pytest.raises(
        RuntimeError, match=rf"pre-step computed {pre} \+ scheduled {count} - 1"
    ):
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=[req_id],
                req_id_to_index={req_id: 0},
                sampled_token_ids=[[123]],
                owner_receipt_batches=[
                    OwnerReceiptBatch(
                        owner_rank=rank,
                        emitted_step_seq=out.step_seq,
                        events=(),
                    )
                    for rank in range(scheduler.parallel_config.world_size)
                ],
                owner_sampling_batches=[
                    OwnerSamplingBatch(
                        owner_rank=owner_rank,
                        emitted_step_seq=out.step_seq,
                        row_ids=(_row(req_id, 0, count - 2),),
                    )
                ],
            ),
        )

    # Atomic no-mutation: rejection left every observable scheduler/request
    # state untouched.
    assert _scheduler_state_snapshot(scheduler, request) == before


def test_seam_rejects_unknown_finished_scheduled_request() -> None:
    scheduler = _make_scheduler(
        max_num_scheduled_tokens=64, sampling_enabled=True
    )
    request = _request("req-0")
    scheduler.add_request(request)
    out = _admit(scheduler, request, grant=8)
    request.status = RequestStatus.FINISHED_ABORTED

    owner_rank = scheduler.owner_coordinator.owner_of(
        scheduler._owner_key[request.request_id]
    )
    with pytest.raises(RuntimeError, match="unknown or finished"):
        scheduler._validate_request_owned_sampling_envelope(
            out,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[123]],
                owner_sampling_batches=[
                    OwnerSamplingBatch(
                        owner_rank=owner_rank,
                        emitted_step_seq=out.step_seq,
                        row_ids=(
                            _row(request.request_id, 0, 31),
                        ),
                    )
                ],
            ),
        )


def test_seam_rejects_missing_lease_key() -> None:
    scheduler = _make_scheduler(
        max_num_scheduled_tokens=64, sampling_enabled=True
    )
    request = _request("req-0")
    scheduler.add_request(request)
    out = _admit(scheduler, request, grant=8)
    scheduler._owner_key.pop(request.request_id)

    with pytest.raises(RuntimeError, match="no live owner lease key"):
        scheduler._validate_request_owned_sampling_envelope(
            out,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[123]],
                owner_sampling_batches=[
                    OwnerSamplingBatch(
                        owner_rank=0,
                        emitted_step_seq=out.step_seq,
                        row_ids=(_row(request.request_id, 0, 31),),
                    )
                ],
            ),
        )


def test_seam_rejects_envelope_when_scheduler_not_owner_enabled() -> None:
    scheduler = _make_scheduler()
    scheduler.scheduler_config = SimpleNamespace(
        enable_request_owned_attention=False,
        enable_request_owned_sampling=True,
    )
    with pytest.raises(RuntimeError, match="enable_request_owned_attention"):
        scheduler._validate_request_owned_sampling_envelope(
            SimpleNamespace(step_seq=1, num_scheduled_tokens={}),
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                owner_sampling_batches=[
                    OwnerSamplingBatch(owner_rank=0, emitted_step_seq=1),
                ],
            ),
        )


def test_seam_sampling_enabled_requires_explicit_batches_even_on_heartbeat() -> None:
    scheduler = _make_scheduler(sampling_enabled=True)
    out = scheduler.schedule()
    assert out.total_num_scheduled_tokens == 0

    with pytest.raises(RuntimeError, match="carries no owner_sampling_batches"):
        scheduler._validate_request_owned_sampling_envelope(
            out,
            ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        )


def test_seam_sampling_disabled_rejects_unexpected_batches() -> None:
    scheduler = _make_scheduler(sampling_enabled=False)
    out = scheduler.schedule()

    with pytest.raises(RuntimeError, match="sampling is disabled"):
        scheduler._validate_request_owned_sampling_envelope(
            out,
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                owner_sampling_batches=[
                    OwnerSamplingBatch(owner_rank=0, emitted_step_seq=out.step_seq),
                    OwnerSamplingBatch(owner_rank=1, emitted_step_seq=out.step_seq),
                ],
            ),
        )


def test_seam_rejects_mutated_non_bool_sampling_gate() -> None:
    scheduler = _make_scheduler()
    scheduler.scheduler_config.enable_request_owned_sampling = 1

    with pytest.raises(RuntimeError, match="must remain a bool"):
        scheduler._validate_request_owned_sampling_envelope(
            scheduler.schedule(),
            ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        )


def _scheduler_state_snapshot(scheduler, request) -> dict:
    req_id = request.request_id
    return {
        "status": request.status,
        "num_computed_tokens": request.num_computed_tokens,
        "attention_owner": request.attention_owner,
        "attention_owner_epoch": request.attention_owner_epoch,
        "owner_key": scheduler._owner_key.get(req_id),
        "owner_epoch": scheduler._owner_epoch.get(req_id),
        "pending_command": scheduler._owner_pending_command.get(req_id),
        "outbox": list(scheduler._owner_outbox),
        "pool_snapshots": dict(scheduler._owner_pool_snapshots),
        "pool_snapshot_seen": scheduler._owner_pool_snapshot_seen,
    }
