from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.native_recapture_observer import (
    NativeRecaptureScopeObserver,
)
from vllm.v1.core.sched.priority_scheduling_observer import (
    PrioritySchedulingObserver,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore


def _scheduler() -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_scheduled_tokens = 4096
    scheduler._startup_max_num_running_reqs = 16
    scheduler._startup_max_num_scheduled_tokens = 4096
    scheduler._runtime_config_epoch = 0
    scheduler._staged_runtime_scheduler_limits = None
    scheduler._startup_scheduler_reserve_full_isl = True
    scheduler.scheduler_reserve_full_isl = True
    scheduler._staged_runtime_prefill_admission_guard = None
    scheduler._pending_epoch_requests = []
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.running = []
    scheduler.waiting = create_request_queue(SchedulingPolicy.PRIORITY)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.PRIORITY)
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.requests = {}
    scheduler.connector = None
    scheduler.log_stats = False
    scheduler._native_recapture_scope_observer = NativeRecaptureScopeObserver(
        enabled=True,
        capacity=128,
        request_prefix="chatcmpl-anise-native-recapture-",
    )
    scheduler._priority_scheduling_observer = PrioritySchedulingObserver(
        enabled=True,
        capacity=128,
        request_prefix="chatcmpl-anise-priority-",
    )
    scheduler.policy = SchedulingPolicy.PRIORITY
    return scheduler


def test_native_recapture_observer_records_exact_formed_batch_membership() -> None:
    scheduler = _scheduler()
    output = SimpleNamespace(
        num_scheduled_tokens={
            "chatcmpl-anise-native-recapture-a": 32,
            "chatcmpl-anise-native-recapture-b": 48,
            "chatcmpl-unrelated": 16,
        },
        total_num_scheduled_tokens=96,
    )

    scheduler._record_native_recapture_scope(output)
    state = scheduler.get_native_recapture_scope_state()

    assert state["latest_sequence"] == 1
    assert len(state["receipts"]) == 1
    receipt = state["receipts"][0]
    assert receipt["formed_batch"] is True
    assert receipt["matched_request_count"] == 2
    assert receipt["request_count"] == 3
    assert receipt["request_ids"] == [
        "chatcmpl-anise-native-recapture-a",
        "chatcmpl-anise-native-recapture-b",
        "chatcmpl-unrelated",
    ]
    assert scheduler.get_native_recapture_scope_state(1)["receipts"] == []


def test_native_recapture_observer_ignores_unbound_requests() -> None:
    scheduler = _scheduler()
    output = SimpleNamespace(
        num_scheduled_tokens={"chatcmpl-unrelated": 16},
        total_num_scheduled_tokens=16,
    )

    scheduler._record_native_recapture_scope(output)

    assert scheduler.get_native_recapture_scope_state()["receipts"] == []


def test_priority_observer_records_order_without_request_content() -> None:
    scheduler = _scheduler()
    high_id = "chatcmpl-anise-priority-high"
    low_id = "chatcmpl-anise-priority-low"
    scheduler.requests = {
        high_id: SimpleNamespace(priority=0, arrival_time=2.0),
        low_id: SimpleNamespace(priority=5, arrival_time=1.0),
    }
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=high_id),
            SimpleNamespace(req_id=low_id),
        ],
        num_scheduled_tokens={high_id: 32, low_id: 32},
    )

    scheduler._record_priority_scheduling_scope(output)
    state = scheduler.get_priority_scheduling_state()

    assert state["latest_sequence"] == 1
    receipt = state["receipts"][0]
    assert receipt["queue_policy"] == "priority"
    assert receipt["scheduled_new_request_ids"] == [high_id, low_id]
    assert receipt["priorities"] == {high_id: 0, low_id: 5}
    assert "prompt" not in str(receipt).lower()


def test_priority_observer_returns_immutable_copies() -> None:
    scheduler = _scheduler()
    request_id = "chatcmpl-anise-priority-copy"
    scheduler.requests = {
        request_id: SimpleNamespace(priority=0, arrival_time=1.0),
    }
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request_id)],
        num_scheduled_tokens={request_id: 16},
    )
    scheduler._record_priority_scheduling_scope(output)

    first = scheduler.get_priority_scheduling_state()
    first["receipts"][0]["priorities"][request_id] = 99

    second = scheduler.get_priority_scheduling_state()
    assert second["receipts"][0]["priorities"][request_id] == 0


def test_priority_observer_rejects_scoped_request_on_fcfs() -> None:
    observer = PrioritySchedulingObserver(
        enabled=True,
        capacity=128,
        request_prefix="chatcmpl-anise-priority-",
    )

    with pytest.raises(RuntimeError, match="non-priority queue"):
        observer.record(
            policy="fcfs",
            scheduled_new_request_ids=["chatcmpl-anise-priority-a"],
            request_metadata={
                "chatcmpl-anise-priority-a": {"priority": 0, "arrival_time": 1.0}
            },
            formed_request_ids=["chatcmpl-anise-priority-a"],
            config_epoch=0,
        )


def test_runtime_scheduler_limits_commit_and_rollback_without_restart() -> None:
    scheduler = _scheduler()
    prepared = scheduler.prepare_runtime_scheduler_limits(2048, 8)

    committed = scheduler.commit_runtime_scheduler_limits(prepared, 1)

    assert committed["config_epoch"] == 1
    assert scheduler.max_num_scheduled_tokens == 2048
    assert scheduler.max_num_running_reqs == 8

    rollback = scheduler.rollback_runtime_scheduler_limits(prepared)

    assert rollback["rollback_epoch"] == 2
    assert scheduler.max_num_scheduled_tokens == 4096
    assert scheduler.max_num_running_reqs == 16


def test_runtime_scheduler_limits_cannot_exceed_startup_ceiling() -> None:
    scheduler = _scheduler()

    with pytest.raises(ValueError, match="startup ceiling"):
        scheduler.prepare_runtime_scheduler_limits(8192, 8)


def test_runtime_scheduler_limits_require_idle_epoch() -> None:
    scheduler = _scheduler()
    scheduler.running.append(object())

    with pytest.raises(RuntimeError, match="idle epoch"):
        scheduler.prepare_runtime_scheduler_limits(2048, 8)


def test_runtime_scheduler_limits_do_not_treat_paused_queue_as_idle() -> None:
    scheduler = _scheduler()
    scheduler._pause_state = PauseState.PAUSED_ALL
    scheduler.waiting.add_request(object())

    with pytest.raises(RuntimeError, match="idle epoch"):
        scheduler.prepare_runtime_scheduler_limits(2048, 8)


class _PendingRequest:
    def __init__(self, request_id: str) -> None:
        from vllm.v1.request import RequestStatus

        self.request_id = request_id
        self.resumable = False
        self.status = RequestStatus.WAITING


def test_staged_epoch_accepts_but_does_not_schedule_new_requests() -> None:
    scheduler = _scheduler()
    scheduler.running.append(object())
    scheduler.stage_runtime_scheduler_limits(2048, 8)

    request = _PendingRequest("next-epoch-request")
    scheduler.add_request(request)

    assert list(scheduler.waiting) == []
    assert scheduler._pending_epoch_requests == [request]
    assert scheduler.get_num_unfinished_requests() == 2


def test_staged_epoch_releases_requests_only_after_old_epoch_drains() -> None:
    scheduler = _scheduler()
    scheduler.running.append(object())
    scheduler.stage_runtime_scheduler_limits(2048, 8)
    request = _PendingRequest("next-epoch-request")
    scheduler.add_request(request)

    with pytest.raises(RuntimeError, match="has not drained"):
        scheduler.commit_staged_runtime_scheduler_limits(1)

    scheduler.running.clear()
    committed = scheduler.commit_staged_runtime_scheduler_limits(1)

    assert committed["released_pending_requests"] == 1
    assert list(scheduler.waiting) == [request]
    assert scheduler.max_num_scheduled_tokens == 2048
    assert scheduler._runtime_config_epoch == 1


def test_prefill_admission_guard_commit_and_rollback_without_restart() -> None:
    scheduler = _scheduler()
    prepared = scheduler.prepare_runtime_prefill_admission_guard(False)

    committed = scheduler.commit_runtime_prefill_admission_guard(prepared, 1)

    assert committed == {"config_epoch": 1, "scheduler_reserve_full_isl": 0}
    assert scheduler.scheduler_reserve_full_isl is False

    rollback = scheduler.rollback_runtime_prefill_admission_guard(prepared)

    assert rollback == {"rollback_epoch": 2, "scheduler_reserve_full_isl": 1}
    assert scheduler.scheduler_reserve_full_isl is True


def test_prefill_admission_guard_stages_new_requests_until_drain() -> None:
    scheduler = _scheduler()
    scheduler.running.append(object())
    scheduler.stage_runtime_prefill_admission_guard(False)
    request = _PendingRequest("next-guard-epoch-request")

    scheduler.add_request(request)

    assert list(scheduler.waiting) == []
    assert scheduler._pending_epoch_requests == [request]
    with pytest.raises(RuntimeError, match="has not drained"):
        scheduler.commit_staged_runtime_prefill_admission_guard(1)

    scheduler.running.clear()
    committed = scheduler.commit_staged_runtime_prefill_admission_guard(1)

    assert committed["released_pending_requests"] == 1
    assert list(scheduler.waiting) == [request]
    assert scheduler.scheduler_reserve_full_isl is False


def test_only_one_runtime_epoch_transition_may_be_staged() -> None:
    scheduler = _scheduler()
    scheduler.stage_runtime_prefill_admission_guard(False)

    with pytest.raises(RuntimeError, match="already staged"):
        scheduler.stage_runtime_scheduler_limits(2048, 8)


def test_engine_core_prefill_admission_guard_receipt_is_effective() -> None:
    core = object.__new__(EngineCore)
    core.scheduler = _scheduler()
    core._runtime_control_boot_id = "boot-test"
    config = {"scheduler_reserve_full_isl": False}

    prepared = core.prepare_runtime_transition("prefill_admission_guard", config)
    committed = core.commit_runtime_transition("prefill_admission_guard", prepared, 1)
    proof = core.verify_runtime_transition("prefill_admission_guard", config, 1)

    assert committed["scheduler_reserve_full_isl"] == 0
    assert proof["effective_runtime_profile"] == "prefill_admission_guard"
    assert proof["prefill_admission_guard_receipt"]["boot_id"] == "boot-test"
    assert (
        proof["prefill_admission_guard_receipt"]["limits"]["scheduler_reserve_full_isl"]
        == 0
    )
