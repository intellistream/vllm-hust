from __future__ import annotations

from collections import deque

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.past_future_policy import PastFuturePolicy
from vllm.v1.core.sched.past_future_scheduler import PastFutureSchedulerPort
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


def _scheduler() -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_scheduled_tokens = 4096
    scheduler._startup_max_num_running_reqs = 16
    scheduler._startup_max_num_scheduled_tokens = 4096
    scheduler._runtime_config_epoch = 0
    scheduler._staged_runtime_scheduler_limits = None
    scheduler._pending_epoch_requests = []
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.running = []
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.requests = {}
    scheduler.log_stats = False
    return scheduler


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
    scheduler.waiting.append(object())

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


class _FinishedRequest:
    def __init__(self, status: RequestStatus, output_tokens: int = 7) -> None:
        self.status = status
        self.output_token_ids = [1] * output_tokens


@pytest.mark.parametrize(
    "status",
    [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR],
)
def test_past_future_history_excludes_abort_and_error_terminals(
    monkeypatch: pytest.MonkeyPatch, status: RequestStatus
) -> None:
    scheduler = object.__new__(PastFutureSchedulerPort)
    scheduler._past_future_policy = PastFuturePolicy(seed=1, initial_output_tokens=512)
    before = tuple(scheduler._past_future_policy.history_output_tokens)
    monkeypatch.setattr(Scheduler, "_free_request", lambda *_args, **_kwargs: None)

    scheduler._free_request(_FinishedRequest(status))

    assert tuple(scheduler._past_future_policy.history_output_tokens) == before


@pytest.mark.parametrize(
    "status",
    [
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
        RequestStatus.FINISHED_REPETITION,
    ],
)
def test_past_future_history_records_successful_generation_terminals(
    monkeypatch: pytest.MonkeyPatch, status: RequestStatus
) -> None:
    scheduler = object.__new__(PastFutureSchedulerPort)
    scheduler._past_future_policy = PastFuturePolicy(seed=1, initial_output_tokens=512)
    monkeypatch.setattr(Scheduler, "_free_request", lambda *_args, **_kwargs: None)

    scheduler._free_request(_FinishedRequest(status, output_tokens=37))

    assert tuple(scheduler._past_future_policy.history_output_tokens)[-1] == 37
