# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the G3 request-owned sampling activation seam in EngineCore.

With ``scheduler_config.enable_request_owned_sampling`` strictly True,
``EngineCore.step`` uses the proven synchronous Multiproc transport:
``execute_model(..., non_block=False)`` returns the concrete worker result
(or ``None`` for a deferred sampling step) without a Future, and a deferred
``None`` is completed immediately through ``sample_tokens``.  The
default-off path keeps the existing asynchronous overlap
(``non_block=True`` + Future) unchanged, abort processing still happens
before ``update_from_output``, and a post-validation non-bool flag fails
closed before any model dispatch.
"""

import queue
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

pytestmark = pytest.mark.cpu_test

#: Sentinel for the sampling flag: leaves the scheduler-config attribute
#: absent (default off) so ``None`` itself can be tested as an invalid value.
_ABSENT = object()


class _FakeScheduler:
    """Minimal scheduler interface used by ``EngineCore.step``."""

    def __init__(self, scheduler_output: SchedulerOutput) -> None:
        self._output = scheduler_output
        self.grammar = object()
        self.calls: list = []

    def has_requests(self) -> bool:
        self.calls.append("has_requests")
        return True

    def schedule(self, throttle: bool) -> SchedulerOutput:
        self.calls.append("schedule")
        return self._output

    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput):
        self.calls.append("grammar")
        return self.grammar

    def make_stats(self):
        return None

    def finish_requests(self, request_ids, status):
        self.calls.append(("finish_requests", list(request_ids), status))

    def update_from_output(self, scheduler_output, model_output):
        self.calls.append(("update_from_output", model_output))
        return {}


class _FakeExecutor:
    """Mirrors the executor overloads: ``non_block=True`` returns a Future,
    ``non_block=False`` returns the concrete result."""

    def __init__(
        self,
        execute_result,
        sample_result=EMPTY_MODEL_RUNNER_OUTPUT,
        execute_exc=None,
        sample_exc=None,
    ) -> None:
        self.execute_result = execute_result
        self.sample_result = sample_result
        self.execute_exc = execute_exc
        self.sample_exc = sample_exc
        self.execute_calls: list[tuple[str, bool]] = []
        self.sample_calls: list = []
        self.returned_futures: list[Future] = []

    def execute_model(self, scheduler_output, non_block=False):
        self.execute_calls.append(("execute_model", non_block))
        if self.execute_exc is not None:
            raise self.execute_exc
        if non_block:
            future: Future = Future()
            future.set_result(self.execute_result)
            self.returned_futures.append(future)
            return future
        return self.execute_result

    def sample_tokens(self, grammar_output):
        self.sample_calls.append(("sample_tokens", grammar_output))
        if self.sample_exc is not None:
            raise self.sample_exc
        return self.sample_result


def _make_output() -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.step_seq = 1
    output.total_num_scheduled_tokens = 1
    return output


def _engine_core(
    sampling_flag=_ABSENT,
    execute_result=EMPTY_MODEL_RUNNER_OUTPUT,
    sample_result=EMPTY_MODEL_RUNNER_OUTPUT,
    execute_exc=None,
    sample_exc=None,
):
    scheduler = _FakeScheduler(_make_output())
    executor = _FakeExecutor(
        execute_result=execute_result,
        sample_result=sample_result,
        execute_exc=execute_exc,
        sample_exc=sample_exc,
    )
    core = EngineCore.__new__(EngineCore)
    core.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(),
        observability_config=SimpleNamespace(enable_logging_iteration_details=False),
    )
    if sampling_flag is not _ABSENT:
        core.vllm_config.scheduler_config.enable_request_owned_sampling = sampling_flag
    core.scheduler = scheduler
    core.model_executor = executor
    core.aborts_queue = queue.Queue()
    return core, scheduler, executor


# -- default-off path: asynchronous overlap unchanged -------------------------


def test_default_path_keeps_non_block_true_and_future() -> None:
    core, scheduler, executor = _engine_core()

    outputs, model_executed = core.step()

    assert outputs == {}
    assert model_executed is True
    # The flag is absent: the existing asynchronous overlap is unchanged.
    assert executor.execute_calls == [("execute_model", True)]
    assert len(executor.returned_futures) == 1
    assert executor.sample_calls == []
    assert scheduler.calls[-1] == ("update_from_output", EMPTY_MODEL_RUNNER_OUTPUT)


def test_default_path_deferred_none_still_samples() -> None:
    core, scheduler, executor = _engine_core(execute_result=None)

    core.step()

    assert executor.execute_calls == [("execute_model", True)]
    assert len(executor.returned_futures) == 1
    assert len(executor.sample_calls) == 1
    assert executor.sample_calls[0][1] is scheduler.grammar


# -- sampling path: synchronous transport ------------------------------------


def test_sampling_path_executes_synchronously_concrete() -> None:
    core, scheduler, executor = _engine_core(sampling_flag=True)

    outputs, model_executed = core.step()

    assert outputs == {}
    assert model_executed is True
    assert executor.execute_calls == [("execute_model", False)]
    # Concrete result: never a Future, never .result(), no sampling.
    assert executor.returned_futures == []
    assert executor.sample_calls == []
    assert scheduler.calls[-1] == ("update_from_output", EMPTY_MODEL_RUNNER_OUTPUT)


def test_sampling_path_deferred_none_completes_through_sample_tokens() -> None:
    core, scheduler, executor = _engine_core(sampling_flag=True, execute_result=None)

    outputs, model_executed = core.step()

    assert outputs == {}
    assert model_executed is True
    assert executor.execute_calls == [("execute_model", False)]
    assert executor.returned_futures == []
    assert len(executor.sample_calls) == 1
    assert executor.sample_calls[0][1] is scheduler.grammar
    assert scheduler.calls[-1] == ("update_from_output", EMPTY_MODEL_RUNNER_OUTPUT)


# -- exceptions propagate ----------------------------------------------------


def test_sampling_path_execute_exception_propagates() -> None:
    core, scheduler, executor = _engine_core(
        sampling_flag=True, execute_exc=RuntimeError("gpu exploded")
    )

    with pytest.raises(RuntimeError, match="gpu exploded"):
        core.step()

    assert executor.execute_calls == [("execute_model", False)]
    assert executor.sample_calls == []


def test_sampling_path_deferred_sample_exception_propagates() -> None:
    core, scheduler, executor = _engine_core(
        sampling_flag=True,
        execute_result=None,
        sample_exc=RuntimeError("sample exploded"),
    )

    with pytest.raises(RuntimeError, match="sample exploded"):
        core.step()

    assert executor.execute_calls == [("execute_model", False)]


# -- strict gate: non-bool fails before dispatch -----------------------------


@pytest.mark.parametrize("bad", [1, "true", None, 0.0])
def test_non_bool_sampling_flag_fails_before_dispatch(bad) -> None:
    core, scheduler, executor = _engine_core(sampling_flag=bad)

    with pytest.raises(RuntimeError, match="must be a bool"):
        core.step()

    # Strict gate: the model executor is never dispatched.
    assert executor.execute_calls == []
    assert executor.sample_calls == []


# -- abort processing / update ordering preserved ----------------------------


def test_sampling_path_preserves_abort_processing_before_update() -> None:
    core, scheduler, executor = _engine_core(sampling_flag=True, execute_result=None)
    core.aborts_queue.put(["req-1"])

    core.step()

    calls = scheduler.calls
    finish_idx = next(i for i, call in enumerate(calls) if call[0] == "finish_requests")
    update_idx = next(
        i for i, call in enumerate(calls) if call[0] == "update_from_output"
    )
    # Aborts queued during model execution are processed before the
    # scheduler consumes the model output, exactly like the default path.
    assert finish_idx < update_idx
    assert calls[finish_idx][:2] == ("finish_requests", ["req-1"])
