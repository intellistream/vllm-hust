# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import queue
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import CacheConfig, SchedulerConfig
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import MicroBatch, MicroBatchCostModel, Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def enable_pp_opt_scheduler(monkeypatch):
    monkeypatch.setenv("VLLM_USE_PP_OPT_SCHEDULER", "1")


class _FakeMMRegistry:
    def supports_multimodal_inputs(self, model_config) -> bool:
        return False


class _FakeStructuredOutputManager:
    def should_advance(self, request: Request) -> bool:
        return False

    def make_cache_stats(self):
        return None


def _create_pp_opt_scheduler(
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
    pipeline_parallel_size: int = 1,
    sliding_window: int | None = None,
) -> Scheduler:
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            is_diffusion=False,
            max_model_len=max_num_batched_tokens,
            enable_return_routed_experts=False,
        ),
        cache_config=cache_config,
        lora_config=None,
        kv_events_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            data_parallel_rank=0,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        observability_config=SimpleNamespace(
            kv_cache_metrics=False,
            kv_cache_metrics_sample=0,
            enable_mfu_metrics=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        ec_transfer_config=None,
        num_speculative_tokens=0,
        use_v2_model_runner=False,
    )
    if sliding_window is None:
        attention_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        )
    else:
        attention_spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=sliding_window,
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                attention_spec,
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=_FakeStructuredOutputManager(),
        mm_registry=_FakeMMRegistry(),
    )


def _model_output_for(
    scheduler_output: SchedulerOutput,
    sampled_token_ids: list[list[int]] | None = None,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    if sampled_token_ids is None:
        sampled_token_ids = [[idx + 1] for idx in range(len(req_ids))]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _add_running_pp_opt_request(
    scheduler,
    request: Request,
    microbatch_id: int,
) -> None:
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.pp_opt_microbatches[microbatch_id].requests.append(request)
    scheduler.request_id_to_microbatch_id[request.request_id] = microbatch_id
    scheduler.microbatch_id_to_request_ids.setdefault(microbatch_id, set()).add(
        request.request_id
    )


def test_pp_opt_admission_balances_waiting_requests_by_microbatch_cost():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=2)
    long_request = create_requests(1, num_tokens=20, req_ids=["long"])[0]
    _add_running_pp_opt_request(scheduler, long_request, microbatch_id=0)

    waiting_requests = create_requests(2, num_tokens=1, req_ids=["a", "b"])
    for request in waiting_requests:
        scheduler.add_request(request)

    scheduler._admit_waiting_requests_pp_opt()

    assert scheduler.request_id_to_microbatch_id["a"] == 1
    assert scheduler.request_id_to_microbatch_id["b"] == 1


def test_pp_opt_admission_sends_external_kv_table_without_stale_aliases():
    block_size = 4
    scheduler = _create_pp_opt_scheduler(
        max_num_batched_tokens=64,
        num_blocks=100,
        block_size=block_size,
        pipeline_parallel_size=2,
        sliding_window=8,
    )
    request = create_requests(
        1,
        num_tokens=20,
        max_tokens=8,
        req_ids=["r"],
    )[0]
    scheduler.add_request(request)

    first_output = scheduler.schedule_pp_opt(0)
    initial_blocks = list(first_output.scheduled_new_reqs[0].block_ids[0])
    null_block_id = scheduler.kv_cache_manager.block_pool.null_block.block_id

    # External tokens outside the attention window must already be represented
    # by null blocks in the first full table sent to the worker.
    assert initial_blocks.count(null_block_id) > 1
    assert initial_blocks == scheduler.kv_cache_manager.get_block_ids("r")[0]

    # Mirror the worker's full-table initialization followed by cached deltas.
    worker_blocks = initial_blocks.copy()
    previous_slot = None
    reused_block_seen = False
    for token_id in range(8):
        scheduler.update_from_output_pp_opt(
            first_output,
            _model_output_for(first_output, [[token_id + 1]]),
        )
        if not scheduler.has_requests_pp_opt():
            break
        first_output = scheduler.schedule_pp_opt(0)
        if not first_output.num_scheduled_tokens:
            break
        new_blocks = first_output.scheduled_cached_reqs.new_block_ids[0]
        if new_blocks is not None:
            if "r" in first_output.scheduled_cached_reqs.block_table_replaced_req_ids:
                worker_blocks = list(new_blocks[0])
            else:
                worker_blocks.extend(new_blocks[0])

        live_blocks = [block for block in worker_blocks if block != null_block_id]
        assert len(live_blocks) == len(set(live_blocks))

        position = first_output.scheduled_cached_reqs.num_computed_tokens[0]
        slot = (
            worker_blocks[position // block_size] * block_size + position % block_size
        )
        if previous_slot is not None and position % block_size:
            assert slot == previous_slot + 1
        if new_blocks is not None:
            reused_block_seen |= new_blocks[0][-1] < max(initial_blocks)
        previous_slot = slot

    assert reused_block_seen


def test_pp_opt_six_feature_cost_model_prediction():
    model = MicroBatchCostModel(
        p0=1.0,
        p1=2.0,
        p2=3.0,
        p3=4.0,
        p4=5.0,
        p5=6.0,
        pp_rank=0,
        layer_num=7,
    )

    assert model.predict(request_num=11, aggregated_ctx_length=13) == (
        1 * 11 * 7 + 2 * 11 + 3 * 13 * 7 + 4 * 13 + 5 * 7 + 6
    )


def _scheduler_for_cost_model_loading(pp_size: int) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.parallel_config = SimpleNamespace(pipeline_parallel_size=pp_size)
    scheduler.cost_model_path = "fit_result.json"
    return scheduler


def _fit_model(cost: str, pp_rank: int, layer_num: int) -> dict:
    return {
        "cost": cost,
        "pp_rank": pp_rank,
        "layer_num": layer_num,
        "coefficients": {
            "p0": 0.0,
            "p1": float(pp_rank + 1),
            "p2": 0.0,
            "p3": 10.0,
            "p4": 100.0,
            "p5": 1000.0,
        },
    }


def test_pp_opt_cost_model_loading_without_partition_uses_stage0_for_all_stages(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    scheduler = _scheduler_for_cost_model_loading(pp_size=2)
    fit_result = {
        "models": [
            _fit_model("total", 0, 16),
            _fit_model("total", 1, 32),
        ],
    }

    models = scheduler._load_pp_opt_cost_models(fit_result, "total")

    assert set(models) == {0, 1}
    assert models[0].pp_rank == 0
    assert models[1].pp_rank == 1
    assert models[0].p1 == 1.0
    assert models[1].p1 == 1.0
    assert models[0].layer_num == 16
    assert models[1].layer_num == 16


def test_pp_opt_cost_model_loading_uses_partition_layer_nums(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "13,17,17,17")
    scheduler = _scheduler_for_cost_model_loading(pp_size=4)
    fit_result = {
        "models": [
            _fit_model("total", 0, 16),
            _fit_model("total", 1, 16),
        ],
    }

    models = scheduler._load_pp_opt_cost_models(fit_result, "total")

    assert [models[rank].layer_num for rank in range(4)] == [13, 17, 17, 17]
    assert [models[rank].p1 for rank in range(4)] == [1.0, 2.0, 1.0, 1.0]


def test_pp_opt_microbatch_cost_uses_max_forward_stage_cost():
    scheduler = _scheduler_for_cost_model_loading(pp_size=2)
    scheduler.forward_cost_models = {
        0: MicroBatchCostModel(
            p0=0.0,
            p1=1.0,
            p2=0.0,
            p3=1.0,
            p4=0.0,
            p5=0.0,
            pp_rank=0,
            layer_num=1,
        ),
        1: MicroBatchCostModel(
            p0=0.0,
            p1=10.0,
            p2=0.0,
            p3=0.0,
            p4=0.0,
            p5=0.0,
            pp_rank=1,
            layer_num=1,
        ),
    }
    requests = create_requests(2, num_tokens=5)
    microbatch = MicroBatch(microbatch_id=0, requests=requests)

    assert scheduler.microbatch_cost(microbatch) == 20.0


def _request(request_id: str, num_tokens: int) -> Request:
    return create_requests(1, num_tokens=num_tokens, req_ids=[request_id])[0]


def test_pp_opt_admission_uses_target_microbatch_selector(monkeypatch):
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=2)
    request = _request("r", 4)
    scheduler.add_request(request)
    calls = []

    def select_target(waiting_request: Request):
        calls.append(waiting_request.request_id)
        return scheduler.pp_opt_microbatches[1]

    monkeypatch.setattr(
        scheduler,
        "_select_pp_opt_target_microbatch",
        select_target,
    )

    scheduler._admit_waiting_requests_pp_opt()

    assert calls == ["r"]
    assert scheduler.request_id_to_microbatch_id["r"] == 1


def test_select_microbatch_pp_opt_skips_empty_free_microbatches():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=3)
    request = create_requests(1, num_tokens=4, req_ids=["r"])[0]
    request.num_computed_tokens = request.num_tokens - 1
    _add_running_pp_opt_request(scheduler, request, microbatch_id=2)

    assert scheduler.select_microbatch_pp_opt(set()) == 2
    assert scheduler.select_microbatch_pp_opt({2}) is None


def test_pp_opt_compacts_free_microbatches_when_waiting_empty(monkeypatch):
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=4,
    )
    for microbatch_id, request in enumerate(
        create_requests(4, num_tokens=4, req_ids=["a", "b", "c", "d"])
    ):
        _add_running_pp_opt_request(scheduler, request, microbatch_id)
        scheduler.pp_opt_first_scheduled_req_ids.add(request.request_id)

    assert scheduler.select_microbatch_pp_opt(set()) == 0

    assert {
        microbatch_id: [
            request.request_id
            for request in scheduler.pp_opt_microbatches[microbatch_id].requests
        ]
        for microbatch_id in range(4)
    } == {0: ["a", "b"], 1: ["c", "d"], 2: [], 3: []}
    assert scheduler.microbatch_id_to_request_ids == {
        0: {"a", "b"},
        1: {"c", "d"},
    }
    assert scheduler.request_id_to_microbatch_id == {
        "a": 0,
        "b": 0,
        "c": 1,
        "d": 1,
    }


def test_pp_opt_compaction_preserves_in_flight_microbatches(monkeypatch):
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=4,
    )
    for microbatch_id, request in enumerate(
        create_requests(4, num_tokens=4, req_ids=["a", "b", "c", "d"])
    ):
        _add_running_pp_opt_request(scheduler, request, microbatch_id)
        scheduler.pp_opt_first_scheduled_req_ids.add(request.request_id)

    scheduler.select_microbatch_pp_opt({2})

    assert {
        microbatch_id: [
            request.request_id
            for request in scheduler.pp_opt_microbatches[microbatch_id].requests
        ]
        for microbatch_id in range(4)
    } == {0: ["a", "b"], 1: ["d"], 2: ["c"], 3: []}
    assert scheduler.request_id_to_microbatch_id == {
        "a": 0,
        "b": 0,
        "c": 2,
        "d": 1,
    }


def test_pp_opt_compaction_can_run_with_waiting_backlog(monkeypatch):
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=4,
    )
    for microbatch_id, request in enumerate(
        create_requests(4, num_tokens=4, req_ids=["a", "b", "c", "d"])
    ):
        _add_running_pp_opt_request(scheduler, request, microbatch_id)
        scheduler.pp_opt_first_scheduled_req_ids.add(request.request_id)
    scheduler.add_request(_request("waiting", 4))

    scheduler._rebalance_pp_opt_free_microbatches(set())

    assert {
        microbatch_id: [
            request.request_id
            for request in scheduler.pp_opt_microbatches[microbatch_id].requests
        ]
        for microbatch_id in range(4)
    } == {0: ["a", "b"], 1: ["c", "d"], 2: [], 3: []}
    assert scheduler.request_id_to_microbatch_id == {
        "a": 0,
        "b": 0,
        "c": 1,
        "d": 1,
    }


def test_pp_opt_dynamic_microbatch_count_expands_with_queue_pressure(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PP_OPT_BATCH_QUEUE_SIZE", "4")
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=2,
    )
    for request in create_requests(
        8,
        num_tokens=4,
        req_ids=[f"r{i}" for i in range(8)],
    ):
        scheduler.add_request(request)

    scheduler._admit_waiting_requests_pp_opt()

    assert scheduler.pp_opt_active_microbatch_num == 4
    assert [
        scheduler.pp_opt_microbatches[microbatch_id].request_num
        for microbatch_id in range(4)
    ] == [2, 2, 2, 2]


def test_pp_opt_dynamic_microbatch_count_uses_configured_target(monkeypatch):
    monkeypatch.setenv("VLLM_PP_OPT_BATCH_QUEUE_SIZE", "4")
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    monkeypatch.setenv("VLLM_PP_OPT_MIN_MICROBATCHES", "1")
    monkeypatch.setenv("VLLM_PP_OPT_TARGET_MICROBATCH_SIZE", "24")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=512,
        pipeline_parallel_size=4,
    )
    for request in create_requests(
        97,
        num_tokens=4,
        req_ids=[f"r{i}" for i in range(97)],
    ):
        scheduler.add_request(request)

    scheduler._admit_waiting_requests_pp_opt()

    assert scheduler.pp_opt_active_microbatch_num == 4
    assert [
        scheduler.pp_opt_microbatches[microbatch_id].request_num
        for microbatch_id in range(4)
    ] == [25, 24, 24, 24]
    assert len(scheduler.waiting) == 0


def test_pp_opt_dynamic_microbatch_count_uses_min_for_small_burst(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PP_OPT_BATCH_QUEUE_SIZE", "4")
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=2,
    )
    for request in create_requests(
        3,
        num_tokens=4,
        req_ids=["a", "b", "c"],
    ):
        scheduler.add_request(request)

    scheduler._admit_waiting_requests_pp_opt()

    assert scheduler.pp_opt_active_microbatch_num == 2
    assert [
        scheduler.pp_opt_microbatches[microbatch_id].request_num
        for microbatch_id in range(4)
    ] == [2, 1, 0, 0]


def test_pp_opt_dynamic_microbatch_count_shrinks_tail_and_compacts(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PP_OPT_BATCH_QUEUE_SIZE", "4")
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=2,
    )
    for microbatch_id, request in zip(
        [2, 3],
        create_requests(2, num_tokens=4, req_ids=["c", "d"]),
    ):
        _add_running_pp_opt_request(scheduler, request, microbatch_id)
        scheduler.pp_opt_first_scheduled_req_ids.add(request.request_id)

    assert scheduler.select_microbatch_pp_opt(set()) == 0

    assert scheduler.pp_opt_active_microbatch_num == 2
    assert {
        microbatch_id: [
            request.request_id
            for request in scheduler.pp_opt_microbatches[microbatch_id].requests
        ]
        for microbatch_id in range(4)
    } == {0: ["c", "d"], 1: [], 2: [], 3: []}
    assert scheduler.request_id_to_microbatch_id == {"c": 0, "d": 0}


def test_pp_opt_dynamic_microbatch_count_preserves_in_flight_tail(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_PP_OPT_BATCH_QUEUE_SIZE", "4")
    monkeypatch.setenv("VLLM_PP_OPT_DYNAMIC_MICROBATCHES", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=2,
    )
    for microbatch_id, request in enumerate(
        create_requests(4, num_tokens=4, req_ids=["a", "b", "c", "d"])
    ):
        _add_running_pp_opt_request(scheduler, request, microbatch_id)
        scheduler.pp_opt_first_scheduled_req_ids.add(request.request_id)

    scheduler.select_microbatch_pp_opt({3})

    assert scheduler.pp_opt_active_microbatch_num == 2
    assert scheduler._get_pp_opt_active_microbatch_ids({3}) == [0, 1, 3]
    assert {
        microbatch_id: [
            request.request_id
            for request in scheduler.pp_opt_microbatches[microbatch_id].requests
        ]
        for microbatch_id in range(4)
    } == {0: ["a", "b"], 1: ["c"], 2: [], 3: ["d"]}
    assert scheduler.request_id_to_microbatch_id == {
        "a": 0,
        "b": 0,
        "c": 1,
        "d": 3,
    }


def test_pp_opt_monitor_logs_microbatch_imbalance(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_PP_OPT_BATCH_QUEUE_SIZE", "4")
    monkeypatch.setenv("VLLM_PP_OPT_MONITOR_INTERVAL", "1")
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=8,
        pipeline_parallel_size=2,
    )
    for request in create_requests(4, num_tokens=4):
        scheduler.add_request(request)

    with caplog.at_level(logging.INFO, logger="vllm.v1.core.sched.scheduler"):
        assert scheduler.select_microbatch_pp_opt(set()) == 0

    assert any(
        "PP optimization monitor:" in record.message
        and "request_count_imbalance" in record.message
        for record in caplog.records
    )


def test_pp_opt_first_schedule_emits_new_request_then_cached_request():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=1)
    request = create_requests(1, num_tokens=4, max_tokens=2, req_ids=["r"])[0]
    scheduler.add_request(request)

    first_output = scheduler.schedule_pp_opt(0)
    assert [req.req_id for req in first_output.scheduled_new_reqs] == ["r"]
    assert first_output.scheduled_cached_reqs.num_reqs == 0
    assert first_output.num_scheduled_tokens == {"r": 1}

    scheduler.update_from_output_pp_opt(first_output, _model_output_for(first_output))
    second_output = scheduler.schedule_pp_opt(0)
    assert second_output.scheduled_new_reqs == []
    assert second_output.scheduled_cached_reqs.req_ids == ["r"]
    assert second_output.num_scheduled_tokens == {"r": 1}


@pytest.mark.skip_global_cleanup
def test_pp_opt_schedule_preserves_scheduler_step_lifecycle(monkeypatch):
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=1)
    request = create_requests(1, num_tokens=4, max_tokens=2, req_ids=["r"])[0]
    scheduler.add_request(request)
    new_step_starts = Mock(wraps=scheduler.kv_cache_manager.new_step_starts)
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "new_step_starts",
        new_step_starts,
    )

    first_output = scheduler.schedule_pp_opt(0)

    assert scheduler.current_step == 1
    new_step_starts.assert_called_once_with()
    assert first_output.kv_cache_usage == scheduler.kv_cache_manager.usage
    assert first_output.num_spec_tokens_to_schedule == scheduler.num_spec_tokens


@pytest.mark.skip_global_cleanup
def test_pp_opt_schedule_exports_new_blocks_for_zeroing(monkeypatch):
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=1)
    request = create_requests(1, num_tokens=4, max_tokens=2, req_ids=["r"])[0]
    scheduler.add_request(request)
    scheduler.needs_kv_cache_zeroing = True
    take_new_block_ids = Mock(return_value=[7, 11])
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "take_new_block_ids",
        take_new_block_ids,
    )

    output = scheduler.schedule_pp_opt(0)

    assert output.new_block_ids_to_zero == [7, 11]
    take_new_block_ids.assert_called_once_with()


@pytest.mark.skip_global_cleanup
def test_pp_opt_schedule_advances_deferred_free_fence():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=1)
    scheduler.defer_block_free = True
    request = create_requests(1, num_tokens=4, max_tokens=2, req_ids=["r"])[0]
    scheduler.add_request(request)

    output = scheduler.schedule_pp_opt(0)

    assert output.total_num_scheduled_tokens > 0
    assert scheduler.sched_step_seq == 1
    assert request.last_sched_seq == scheduler.sched_step_seq


def test_pp_opt_finished_requests_are_removed_from_owning_microbatch():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=1)
    request = create_requests(1, num_tokens=4, max_tokens=1, req_ids=["r"])[0]
    scheduler.add_request(request)

    output = scheduler.schedule_pp_opt(0)
    scheduler.update_from_output_pp_opt(output, _model_output_for(output, [[42]]))

    assert scheduler.pp_opt_microbatches[0].requests == []
    assert "r" not in scheduler.request_id_to_microbatch_id
    assert "r" not in scheduler.requests


def test_pp_opt_multiple_requests_run_to_completion_across_microbatches():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=2)
    for request in create_requests(
        4,
        num_tokens=4,
        max_tokens=3,
        req_ids=["a", "b", "c", "d"],
    ):
        scheduler.add_request(request)

    schedule_counts: dict[str, int] = {}

    while scheduler.has_requests_pp_opt():
        microbatch_id = scheduler.select_microbatch_pp_opt(set())
        if microbatch_id is None:
            break

        output = scheduler.schedule_pp_opt(microbatch_id)
        assert all(count == 1 for count in output.num_scheduled_tokens.values())
        for req_id in output.num_scheduled_tokens:
            schedule_counts[req_id] = schedule_counts.get(req_id, 0) + 1

        scheduler.update_from_output_pp_opt(output, _model_output_for(output))

    assert schedule_counts == {"a": 3, "b": 3, "c": 3, "d": 3}
    assert scheduler.microbatch_id_to_request_ids == {}
    assert scheduler.request_id_to_microbatch_id == {}
    assert all(
        microbatch.requests == []
        for microbatch in scheduler.pp_opt_microbatches.values()
    )


def test_pp_opt_schedules_each_request_in_selected_microbatch():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=1)
    for request in create_requests(
        2,
        num_tokens=4,
        max_tokens=2,
        req_ids=["a", "b"],
    ):
        scheduler.add_request(request)

    first_output = scheduler.schedule_pp_opt(0)
    assert [req.req_id for req in first_output.scheduled_new_reqs] == ["a", "b"]
    assert first_output.scheduled_cached_reqs.num_reqs == 0
    assert first_output.num_scheduled_tokens == {"a": 1, "b": 1}

    scheduler.update_from_output_pp_opt(first_output, _model_output_for(first_output))
    second_output = scheduler.schedule_pp_opt(0)
    assert second_output.scheduled_new_reqs == []
    assert second_output.scheduled_cached_reqs.req_ids == ["a", "b"]
    assert second_output.num_scheduled_tokens == {"a": 1, "b": 1}


def test_default_scheduler_prefills_normally_without_kv_connector():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=2)
    request = create_requests(
        1,
        num_tokens=128,
        max_tokens=2,
        req_ids=["prefill"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"prefill": 128}
    assert [req.req_id for req in output.scheduled_new_reqs] == ["prefill"]
    assert getattr(request, "num_external_computed_tokens", 0) == 0
    assert request.num_computed_tokens == 128


def test_pp_opt_continuously_schedules_multiple_in_flight_microbatches():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=3)
    requests = create_requests(
        3,
        num_tokens=4,
        max_tokens=2,
        req_ids=["a", "b", "c"],
    )
    for request in requests:
        scheduler.add_request(request)

    in_flight_ids: set[int] = set()
    first_wave_outputs: dict[int, SchedulerOutput] = {}
    for expected_microbatch_id in [0, 1, 2]:
        microbatch_id = scheduler.select_microbatch_pp_opt(in_flight_ids)
        assert microbatch_id == expected_microbatch_id
        output = scheduler.schedule_pp_opt(microbatch_id)
        assert len(output.num_scheduled_tokens) == 1
        assert all(count == 1 for count in output.num_scheduled_tokens.values())
        first_wave_outputs[microbatch_id] = output
        in_flight_ids.add(microbatch_id)

    assert scheduler.select_microbatch_pp_opt(in_flight_ids) is None
    assert scheduler.microbatch_id_to_request_ids == {
        0: {"a"},
        1: {"b"},
        2: {"c"},
    }
    assert scheduler.request_id_to_microbatch_id == {"a": 0, "b": 1, "c": 2}
    assert {
        request.request_id: (
            request.num_computed_tokens,
            request.num_tokens_with_spec,
            request.status,
        )
        for request in requests
    } == {
        "a": (4, 4, RequestStatus.RUNNING),
        "b": (4, 4, RequestStatus.RUNNING),
        "c": (4, 4, RequestStatus.RUNNING),
    }
    assert scheduler._get_pp_opt_schedulable_request_count() == 0
    assert scheduler._get_pp_opt_idle_running_request_count() == 3

    scheduler.update_from_output_pp_opt(
        first_wave_outputs[0],
        _model_output_for(first_wave_outputs[0], [[11]]),
    )
    in_flight_ids.remove(0)

    assert scheduler.request_id_to_microbatch_id == {"a": 0, "b": 1, "c": 2}
    assert requests[0].num_computed_tokens == 4
    assert requests[0].num_tokens_with_spec == 5
    assert scheduler.select_microbatch_pp_opt(in_flight_ids) == 0
    second_a_output = scheduler.schedule_pp_opt(0)
    assert second_a_output.scheduled_new_reqs == []
    assert second_a_output.scheduled_cached_reqs.req_ids == ["a"]
    assert second_a_output.num_scheduled_tokens == {"a": 1}
    in_flight_ids.add(0)

    for microbatch_id, output in first_wave_outputs.items():
        if microbatch_id == 0:
            continue
        scheduler.update_from_output_pp_opt(output, _model_output_for(output))
        in_flight_ids.remove(microbatch_id)
    scheduler.update_from_output_pp_opt(
        second_a_output,
        _model_output_for(second_a_output),
    )
    in_flight_ids.remove(0)

    assert in_flight_ids == set()
    assert scheduler.microbatch_id_to_request_ids == {1: {"b"}, 2: {"c"}}
    assert scheduler.request_id_to_microbatch_id == {"b": 1, "c": 2}
    assert "a" not in scheduler.requests
    assert requests[1].status == RequestStatus.RUNNING
    assert requests[2].status == RequestStatus.RUNNING
    assert scheduler.select_microbatch_pp_opt(set()) == 1


def test_pp_opt_empty_update_leaves_request_idle_until_more_work_arrives():
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=1)
    request = create_requests(
        1,
        num_tokens=4,
        max_tokens=2,
        req_ids=["r"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule_pp_opt(0)
    scheduler.update_from_output_pp_opt(output, _model_output_for(output, [[]]))

    assert request.status == RequestStatus.RUNNING
    assert scheduler.request_id_to_microbatch_id["r"] == 0
    assert scheduler.microbatch_id_to_request_ids == {0: {"r"}}
    assert request.num_computed_tokens == request.num_tokens_with_spec
    assert scheduler.select_microbatch_pp_opt(set()) is None

    request.spec_token_ids.append(99)
    assert scheduler.select_microbatch_pp_opt(set()) == 0


def test_pp_opt_global_max_num_seqs_applies_across_microbatches():
    scheduler = _create_pp_opt_scheduler(
        max_num_seqs=2,
        pipeline_parallel_size=4,
    )
    for request in create_requests(5, num_tokens=4):
        scheduler.add_request(request)

    scheduler.schedule_pp_opt(0)

    active_request_count = sum(
        microbatch.request_num for microbatch in scheduler.pp_opt_microbatches.values()
    )
    assert active_request_count == 2
    assert len(scheduler.waiting) == 3


def test_pp_opt_kv_allocation_failure_leaves_blocked_request_waiting(monkeypatch):
    scheduler = _create_pp_opt_scheduler(pipeline_parallel_size=2)
    real_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    num_allocate_calls = 0

    def allocate_once_then_fail(*args, **kwargs):
        nonlocal num_allocate_calls
        num_allocate_calls += 1
        if num_allocate_calls == 1:
            return real_allocate_slots(*args, **kwargs)
        return None

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        allocate_once_then_fail,
    )
    requests = create_requests(2, num_tokens=1, req_ids=["fits", "blocked"])
    for request in requests:
        scheduler.add_request(request)

    scheduler.schedule_pp_opt(0)

    active_request_ids = {
        request.request_id
        for microbatch in scheduler.pp_opt_microbatches.values()
        for request in microbatch.requests
    }
    assert active_request_ids == {"fits"}
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting.peek_request().request_id == "blocked"


def test_engine_core_pp_opt_path_calls_scheduler_pp_opt_methods():
    class FakeExecutor:
        def execute_model(self, scheduler_output, non_block: bool):
            future: Future[ModelRunnerOutput] = Future()
            future.set_result(_model_output_for(scheduler_output))
            return future

    calls: list[str] = []
    scheduler = Mock()
    scheduler.has_requests_pp_opt.side_effect = lambda: calls.append("has") or True
    scheduler.select_microbatch_pp_opt.side_effect = lambda in_flight_ids: (
        calls.append(f"select:{sorted(in_flight_ids)}") or 2
    )

    def schedule_pp_opt(microbatch_id: int):
        calls.append(f"schedule:{microbatch_id}")
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"r": 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

    def update_from_output_pp_opt(scheduler_output, model_runner_output):
        calls.append(f"update:{scheduler_output.microbatch_id}")
        return {0: EngineCoreOutputs()}

    scheduler.schedule_pp_opt.side_effect = schedule_pp_opt
    scheduler.update_from_output_pp_opt.side_effect = update_from_output_pp_opt

    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.model_executor = FakeExecutor()
    engine.batch_queue_size = 2
    engine.batch_queue = deque(maxlen=2)
    engine.in_flight_microbatch_ids = set()
    engine.is_ec_consumer = True
    engine.is_pooling_model = True
    engine.aborts_queue = queue.Queue()
    engine.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(
            enable_logging_iteration_details=False,
        ),
    )

    outputs, model_executed = engine.step_with_batch_queue_pp_opt()

    assert set(outputs) == {0}
    assert isinstance(outputs[0], EngineCoreOutputs)
    assert model_executed is True
    assert calls == [
        "has",
        "select:[]",
        "schedule:2",
        "update:2",
    ]
