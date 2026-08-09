# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.ownership import OwnerReceiptBatch
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import multiproc_executor as multiproc_executor_module
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.executor.output_aggregator import ModelRunnerOutputAggregator
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)
from vllm.v1.outputs import ModelRunnerOutput


class Mock: ...


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeProcess:
    def __init__(self, clock: _FakeClock, exits_at: float) -> None:
        self.clock = clock
        self.exits_at = exits_at
        self.terminate_called = False

    def is_alive(self) -> bool:
        return self.clock.time() < self.exits_at

    def terminate(self) -> None:
        self.terminate_called = True


@pytest.mark.parametrize(
    ("timeout", "exits_at", "expected_terminate"),
    [
        pytest.param(6, 5, False, id="worker-exits-before-timeout"),
        pytest.param(6, 7, True, id="worker-exceeds-timeout"),
    ],
)
def test_multiproc_executor_worker_termination_timeout(
    monkeypatch, timeout, exits_at, expected_terminate
):
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(timeout))
    clock = _FakeClock()
    monkeypatch.setattr(multiproc_executor_module.time, "time", clock.time)
    monkeypatch.setattr(multiproc_executor_module.time, "sleep", clock.sleep)
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    proc = _FakeProcess(clock, exits_at=exits_at)
    executor._ensure_worker_termination([proc])
    assert proc.terminate_called is expected_terminate


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


class _FakeMq:
    """Minimal MessageQueue stand-in recording broadcast enqueues and serving
    pre-queued (status, result) responses."""

    def __init__(self, responses=None):
        self.enqueued = []
        self._responses = list(responses or [])

    def enqueue(self, item):
        self.enqueued.append(item)

    def dequeue(self, timeout=None):
        if not self._responses:
            raise AssertionError("dequeue called with no queued response")
        return self._responses.pop(0)


def _owner_batch(owner_rank: int, emitted_step_seq: int = 7) -> OwnerReceiptBatch:
    return OwnerReceiptBatch(
        owner_rank=owner_rank,
        emitted_step_seq=emitted_step_seq,
        events=(),
    )


def _g0_fake_executor(
    response_mqs,
    enable_request_owned_attention: bool = True,
    cls: type = MultiprocExecutor,
) -> MultiprocExecutor:
    executor = cls.__new__(cls)
    executor.scheduler_config = SimpleNamespace(
        enable_request_owned_attention=enable_request_owned_attention
    )
    executor.kv_output_aggregator = None
    executor.model_runner_output_aggregator = (
        ModelRunnerOutputAggregator([0, 1, 2])
        if enable_request_owned_attention
        else None
    )
    executor.output_rank = 0
    executor.is_failed = False
    executor.rpc_broadcast_mq = _FakeMq()
    executor.response_mqs = response_mqs
    executor.futures_queue = deque()
    return executor


def _three_worker_responses():
    return [
        (
            WorkerProc.ResponseStatus.SUCCESS,
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                owner_receipt_batches=[_owner_batch(rank)],
            ),
        )
        for rank in (0, 1, 2)
    ]


def test_multiproc_executor_owner_aggregation_sync():
    """G0 owner aggregation broadcasts output_rank=None (every worker
    executes) and drains every response MQ on the sync path."""
    responses = _three_worker_responses()
    response_mqs = [_FakeMq([response]) for response in responses]
    executor = _g0_fake_executor(response_mqs)

    result = executor.execute_model(SchedulerOutput.make_empty(), non_block=False)

    method, _, _, output_rank = executor.rpc_broadcast_mq.enqueued[0]
    assert method == "execute_model"
    assert output_rank is None
    assert all(len(mq._responses) == 0 for mq in response_mqs)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]


def test_multiproc_executor_owner_aggregation_future():
    """G0 owner aggregation drains every worker MQ on the Future path too."""
    responses = _three_worker_responses()
    response_mqs = [_FakeMq([response]) for response in responses]
    executor = _g0_fake_executor(response_mqs)

    future = executor.execute_model(SchedulerOutput.make_empty(), non_block=True)
    result = future.result()

    method, _, _, output_rank = executor.rpc_broadcast_mq.enqueued[0]
    assert method == "execute_model"
    assert output_rank is None
    assert all(len(mq._responses) == 0 for mq in response_mqs)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]


class _LegacyCollectiveRpcExecutor(MultiprocExecutor):
    """Custom override that keeps the historical collective_rpc signature."""

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any:
        self.received_aggregator = kv_output_aggregator
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


def test_multiproc_executor_legacy_collective_rpc_override_compatible():
    """Custom collective_rpc overrides that keep the historical signature
    receive the generic owner aggregator through the existing
    kv_output_aggregator slot: no new keyword, no TypeError, and aggregation
    still drains every worker MQ."""
    responses = _three_worker_responses()
    response_mqs = [_FakeMq([response]) for response in responses]
    executor = _g0_fake_executor(response_mqs, cls=_LegacyCollectiveRpcExecutor)

    result = executor.execute_model(SchedulerOutput.make_empty(), non_block=False)

    assert isinstance(executor.received_aggregator, ModelRunnerOutputAggregator)
    method, _, _, output_rank = executor.rpc_broadcast_mq.enqueued[0]
    assert method == "execute_model"
    assert output_rank is None
    assert all(len(mq._responses) == 0 for mq in response_mqs)
    assert [b.owner_rank for b in result.owner_receipt_batches] == [0, 1, 2]


def test_multiproc_executor_default_off_unchanged():
    """With the gate off, execute_model keeps the existing single-rank
    passthrough: only the output-rank MQ is drained, no aggregation."""
    responses = _three_worker_responses()
    response_mqs = [_FakeMq([response]) for response in responses]
    executor = _g0_fake_executor(response_mqs, enable_request_owned_attention=False)

    result = executor.execute_model(SchedulerOutput.make_empty(), non_block=False)

    method, _, _, output_rank = executor.rpc_broadcast_mq.enqueued[0]
    assert method == "execute_model"
    assert output_rank == 0
    assert len(response_mqs[0]._responses) == 0
    assert len(response_mqs[1]._responses) == 1
    assert len(response_mqs[2]._responses) == 1
    assert result is not None
    assert result.owner_receipt_batches == [_owner_batch(0)]
