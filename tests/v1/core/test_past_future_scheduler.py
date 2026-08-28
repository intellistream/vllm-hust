from __future__ import annotations

import importlib.util
import sys
from enum import Enum, auto
from pathlib import Path
from types import ModuleType

import pytest


class _Scheduler:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.max_num_running_reqs = 16

    def _free_request(self, *_args: object, **_kwargs: object) -> None:
        return None


class _Request:
    pass


class _RequestStatus(Enum):
    FINISHED_ABORTED = auto()
    FINISHED_ERROR = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_REPETITION = auto()


_ROOT = Path(__file__).parents[3]
_POLICY_NAME = "vllm.v1.core.sched.past_future_policy"
_POLICY_SPEC = importlib.util.spec_from_file_location(
    _POLICY_NAME,
    _ROOT / "vllm" / "v1" / "core" / "sched" / "past_future_policy.py",
)
assert _POLICY_SPEC is not None and _POLICY_SPEC.loader is not None
_POLICY_MODULE = importlib.util.module_from_spec(_POLICY_SPEC)
sys.modules[_POLICY_NAME] = _POLICY_MODULE
_POLICY_SPEC.loader.exec_module(_POLICY_MODULE)

_ORIGINAL_SCHEDULER_MODULE = sys.modules.get("vllm.v1.core.sched.scheduler")
_ORIGINAL_REQUEST_MODULE = sys.modules.get("vllm.v1.request")
_SCHEDULER_STUB = ModuleType("vllm.v1.core.sched.scheduler")
_SCHEDULER_STUB.Scheduler = _Scheduler
sys.modules[_SCHEDULER_STUB.__name__] = _SCHEDULER_STUB

_REQUEST_STUB = ModuleType("vllm.v1.request")
_REQUEST_STUB.Request = _Request
_REQUEST_STUB.RequestStatus = _RequestStatus
sys.modules[_REQUEST_STUB.__name__] = _REQUEST_STUB

_SCHEDULER_SPEC = importlib.util.spec_from_file_location(
    "past_future_scheduler_test",
    _ROOT / "vllm" / "v1" / "core" / "sched" / "past_future_scheduler.py",
)
assert _SCHEDULER_SPEC is not None and _SCHEDULER_SPEC.loader is not None
_SCHEDULER_MODULE = importlib.util.module_from_spec(_SCHEDULER_SPEC)
try:
    _SCHEDULER_SPEC.loader.exec_module(_SCHEDULER_MODULE)
finally:
    if _ORIGINAL_SCHEDULER_MODULE is None:
        sys.modules.pop(_SCHEDULER_STUB.__name__, None)
    else:
        sys.modules[_SCHEDULER_STUB.__name__] = _ORIGINAL_SCHEDULER_MODULE
    if _ORIGINAL_REQUEST_MODULE is None:
        sys.modules.pop(_REQUEST_STUB.__name__, None)
    else:
        sys.modules[_REQUEST_STUB.__name__] = _ORIGINAL_REQUEST_MODULE

PastFuturePolicy = _POLICY_MODULE.PastFuturePolicy
PastFutureRequestState = _POLICY_MODULE.PastFutureRequestState
PastFutureSchedulerPort = _SCHEDULER_MODULE.PastFutureSchedulerPort


def _state(request_id: str, *, completed: int = 0, maximum: int = 512):
    return PastFutureRequestState(
        request_id=request_id,
        computed_tokens=64,
        completed_output_tokens=completed,
        max_output_tokens=maximum,
    )


def test_past_future_admission_is_seeded() -> None:
    first = PastFuturePolicy(seed=20260816)
    second = PastFuturePolicy(seed=20260816)

    first_decision = first.decide(
        running=(_state("running", completed=32),),
        candidate=_state("candidate"),
        max_kv_tokens=4096,
    )
    second_decision = second.decide(
        running=(_state("running", completed=32),),
        candidate=_state("candidate"),
        max_kv_tokens=4096,
    )

    assert first_decision == second_decision


def test_past_future_rejects_excessive_kv_peak() -> None:
    policy = PastFuturePolicy(seed=7, initial_output_tokens=512)

    decision = policy.decide(
        running=(_state("running", maximum=2048),),
        candidate=_state("candidate", maximum=2048),
        max_kv_tokens=128,
    )

    assert decision.admitted is False


class _FinishedRequest:
    def __init__(self, status: _RequestStatus, output_tokens: int = 7) -> None:
        self.status = status
        self.output_token_ids = [1] * output_tokens


@pytest.mark.parametrize(
    "status",
    [_RequestStatus.FINISHED_ABORTED, _RequestStatus.FINISHED_ERROR],
)
def test_history_excludes_abort_and_error_terminals(status: _RequestStatus) -> None:
    scheduler = PastFutureSchedulerPort()
    before = tuple(scheduler._past_future_policy.history_output_tokens)

    scheduler._free_request(_FinishedRequest(status))

    assert tuple(scheduler._past_future_policy.history_output_tokens) == before


@pytest.mark.parametrize(
    "status",
    [
        _RequestStatus.FINISHED_STOPPED,
        _RequestStatus.FINISHED_LENGTH_CAPPED,
        _RequestStatus.FINISHED_REPETITION,
    ],
)
def test_history_records_successful_generation_terminals(
    status: _RequestStatus,
) -> None:
    scheduler = PastFutureSchedulerPort()

    scheduler._free_request(_FinishedRequest(status, output_tokens=37))

    assert tuple(scheduler._past_future_policy.history_output_tokens)[-1] == 37
