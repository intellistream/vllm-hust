# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for QoS params validation, monotonic deadline conversion,
KeyedRequestQueue, counter lifecycle and OpenAI API passthrough."""

import math
import time

import pytest

from vllm.entrypoints.openai.engine.protocol import QoSRequestParams
from vllm.v1.core.sched.request_queue import (
    KeyedRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.qos import MAX_EXPECTED_OUTPUT_TOKENS, QoSParams, QoSRuntimeState

# ---------------------------------------------------------------------------
# 1. QoSParams validation
# ---------------------------------------------------------------------------


class TestQoSParamsValidation:
    def test_at_least_one_slo_required(self):
        with pytest.raises(ValueError, match="At least one QoS SLO"):
            QoSParams()

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, math.inf, math.nan])
    def test_non_positive_slo_rejected(self, bad_value):
        with pytest.raises(ValueError, match="finite positive"):
            QoSParams(ttft_slo_ms=bad_value)

    def test_expected_output_tokens_bounds(self):
        with pytest.raises(ValueError, match="expected_output_tokens"):
            QoSParams(ttft_slo_ms=100.0, expected_output_tokens=0)
        with pytest.raises(ValueError, match="expected_output_tokens"):
            QoSParams(
                ttft_slo_ms=100.0,
                expected_output_tokens=MAX_EXPECTED_OUTPUT_TOKENS + 1,
            )

    def test_service_class_length(self):
        with pytest.raises(ValueError, match="service_class"):
            QoSParams(ttft_slo_ms=100.0, service_class="")
        with pytest.raises(ValueError, match="service_class"):
            QoSParams(ttft_slo_ms=100.0, service_class="x" * 65)

    def test_valid_params_accepted(self):
        params = QoSParams(
            ttft_slo_ms=100.0,
            tbt_slo_ms=20.0,
            ttlt_slo_ms=5000.0,
            expected_output_tokens=128,
            service_class="premium",
        )
        assert params.ttft_slo_ms == 100.0
        assert params.service_class == "premium"


# ---------------------------------------------------------------------------
# 2. Monotonic deadline conversion
# ---------------------------------------------------------------------------


class TestQoSRuntimeStateDeadline:
    def test_frontend_age_subtracted(self):
        wall_now = 1000.0
        arrival_time = 999.0  # 1s already spent in frontend
        monotonic_now = 5000.0
        state = QoSRuntimeState.from_params(
            QoSParams(ttft_slo_ms=1000.0),
            arrival_time=arrival_time,
            default_expected_output_tokens=64,
            wall_now=wall_now,
            monotonic_now=monotonic_now,
        )
        # 1000ms SLO minus 1s frontend age = 0s slack from monotonic_now
        assert state.ttft_deadline == pytest.approx(monotonic_now)

    def test_deadline_future_when_no_frontend_delay(self):
        wall_now = 1000.0
        arrival_time = 1000.0
        monotonic_now = 5000.0
        state = QoSRuntimeState.from_params(
            QoSParams(ttft_slo_ms=500.0),
            arrival_time=arrival_time,
            default_expected_output_tokens=64,
            wall_now=wall_now,
            monotonic_now=monotonic_now,
        )
        assert state.ttft_deadline == pytest.approx(monotonic_now + 0.5)

    def test_none_slo_yields_none_deadline(self):
        state = QoSRuntimeState.from_params(
            QoSParams(tbt_slo_ms=20.0),
            arrival_time=time.time(),
            default_expected_output_tokens=64,
        )
        assert state.ttft_deadline is None
        assert state.ttlt_deadline is None
        assert state.tbt_slo_s == 0.02

    def test_expected_output_tokens_fallback(self):
        state = QoSRuntimeState.from_params(
            QoSParams(ttft_slo_ms=100.0),
            arrival_time=time.time(),
            default_expected_output_tokens=256,
        )
        assert state.expected_output_tokens == 256

    def test_next_token_deadline_phases(self):
        state = QoSRuntimeState.from_params(
            QoSParams(ttft_slo_ms=100.0, tbt_slo_ms=20.0, ttlt_slo_ms=5000.0),
            arrival_time=time.time(),
            default_expected_output_tokens=64,
        )
        # Before first token: TTFT is the active deadline.
        assert math.isfinite(state.next_token_deadline())
        assert state.has_active_deadline()

        # After first token: TBT takes over.
        state.first_token_time = time.monotonic()
        state.last_token_time = state.first_token_time
        assert math.isfinite(state.next_token_deadline())

    def test_observe_tokens_records_violations(self):
        state = QoSRuntimeState.from_params(
            QoSParams(ttft_slo_ms=10.0, tbt_slo_ms=10.0),
            arrival_time=time.time(),
            default_expected_output_tokens=64,
            monotonic_now=0.0,
            wall_now=0.0,
        )
        # First token arrives after deadline -> TTFT violation.
        ttft, tbt = state.observe_tokens(1, now=0.02)
        assert ttft == 1
        assert tbt == 0
        # Second token within TBT -> no violation.
        ttft, tbt = state.observe_tokens(1, now=0.03)
        assert ttft == 0
        assert tbt == 0
        # Third token exceeds TBT -> violation.
        ttft, tbt = state.observe_tokens(1, now=0.10)
        assert tbt == 1


# ---------------------------------------------------------------------------
# 3. KeyedRequestQueue
# ---------------------------------------------------------------------------


class _StubRequest:
    """Minimal stand-in exposing the fields KeyedRequestQueue consumes."""

    def __init__(self, rid: str, key_value: tuple) -> None:
        self.request_id = rid
        self._key_value = key_value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubRequest) and self.request_id == other.request_id

    def __hash__(self) -> int:
        return hash(self.request_id)


class TestKeyedRequestQueue:
    def test_orders_by_key(self):
        queue = KeyedRequestQueue(lambda r: r._key_value, preserve_prepend=False)
        queue.add_request(_StubRequest("a", (2,)))
        queue.add_request(_StubRequest("b", (0,)))
        queue.add_request(_StubRequest("c", (1,)))
        order = [queue.pop_request().request_id for _ in range(3)]
        assert order == ["b", "c", "a"]

    def test_peek_does_not_remove(self):
        queue = KeyedRequestQueue(lambda r: r._key_value, preserve_prepend=False)
        queue.add_request(_StubRequest("a", (1,)))
        queue.add_request(_StubRequest("b", (0,)))
        assert queue.peek_request().request_id == "b"
        assert len(queue) == 2

    def test_remove_request(self):
        queue = KeyedRequestQueue(lambda r: r._key_value, preserve_prepend=False)
        target = _StubRequest("a", (1,))
        queue.add_request(target)
        queue.add_request(_StubRequest("b", (0,)))
        queue.remove_request(target)
        assert len(queue) == 1
        assert queue.pop_request().request_id == "b"

    def test_remove_missing_raises(self):
        queue = KeyedRequestQueue(lambda r: r._key_value, preserve_prepend=False)
        queue.add_request(_StubRequest("a", (0,)))
        with pytest.raises(ValueError, match="not in queue"):
            queue.remove_request(_StubRequest("missing", (0,)))

    def test_prepend_respects_preserve_flag(self):
        # FCFS mode: prepend inserts at front.
        queue_fcfs = KeyedRequestQueue(lambda r: r._key_value, preserve_prepend=True)
        queue_fcfs.add_request(_StubRequest("a", (1,)))
        queue_fcfs.prepend_request(_StubRequest("b", (1,)))
        assert queue_fcfs.peek_request().request_id == "b"

        # Priority mode: prepend is same as add (key-driven).
        queue_prio = KeyedRequestQueue(lambda r: r._key_value, preserve_prepend=False)
        queue_prio.add_request(_StubRequest("a", (1,)))
        queue_prio.prepend_request(_StubRequest("b", (0,)))
        assert queue_prio.peek_request().request_id == "b"

    def test_factory_creates_keyed_queue(self):
        queue = create_request_queue(
            SchedulingPolicy.FCFS, key=lambda r: (0,)
        )
        assert isinstance(queue, KeyedRequestQueue)

    def test_factory_falls_back_to_fcfs_without_key(self):
        queue = create_request_queue(SchedulingPolicy.FCFS)
        assert not isinstance(queue, KeyedRequestQueue)


# ---------------------------------------------------------------------------
# 4. Abort / finish counter lifecycle
# ---------------------------------------------------------------------------


class TestQoSCounterLifecycle:
    """Counter must stay consistent across add / abort / finish paths."""

    @pytest.fixture
    def policy(self, dummy_vllm_config):
        from vllm.v1.core.sched.qos_policy import QoSSchedulingPolicy

        return QoSSchedulingPolicy(dummy_vllm_config)

    def test_increment_decrement_balance(self, policy):
        assert policy._qos_request_count == 0
        policy.increment_qos_count()
        policy.increment_qos_count()
        assert policy._qos_request_count == 2
        policy.decrement_qos_count()
        assert policy._qos_request_count == 1

    def test_decrement_clamped_at_zero(self, policy):
        policy.decrement_qos_count()
        assert policy._qos_request_count == 0

    def test_observe_finished_decrements(self, policy, dummy_qos_request):
        policy.increment_qos_count()
        assert policy._qos_request_count == 1
        policy.observe_request_output(
            dummy_qos_request, num_new_tokens=1, finished=True
        )
        assert policy._qos_request_count == 0

    def test_abort_decrement(self, policy, dummy_qos_request):
        policy.increment_qos_count()
        assert policy._qos_request_count == 1
        # Scheduler._free_request path decrements when qos_state is present.
        if dummy_qos_request.qos_state is not None:
            policy.decrement_qos_count()
        assert policy._qos_request_count == 0


# ---------------------------------------------------------------------------
# 5. OpenAI API passthrough (chat / completion / responses share QoSRequestParams)
# ---------------------------------------------------------------------------


class TestQoSRequestParamsAPI:
    def test_requires_at_least_one_slo(self):
        with pytest.raises(ValueError, match="At least one QoS SLO"):
            QoSRequestParams()

    def test_non_positive_rejected(self):
        with pytest.raises(ValueError):
            QoSRequestParams(ttft_slo_ms=0.0)

    def test_to_internal_roundtrip(self):
        api_params = QoSRequestParams(
            ttft_slo_ms=100.0,
            tbt_slo_ms=20.0,
            ttlt_slo_ms=5000.0,
            expected_output_tokens=128,
            service_class="premium",
        )
        internal = api_params.to_internal()
        assert isinstance(internal, QoSParams)
        assert internal.ttft_slo_ms == 100.0
        assert internal.tbt_slo_ms == 20.0
        assert internal.ttlt_slo_ms == 5000.0
        assert internal.expected_output_tokens == 128
        assert internal.service_class == "premium"

    def test_partial_slo_accepted(self):
        api_params = QoSRequestParams(ttft_slo_ms=50.0)
        internal = api_params.to_internal()
        assert internal.ttft_slo_ms == 50.0
        assert internal.tbt_slo_ms is None


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_vllm_config():
    from unittest.mock import MagicMock

    config = MagicMock()
    config.scheduler_config.enable_qos_scheduling = True
    config.scheduler_config.policy = "fcfs"
    config.scheduler_config.max_num_scheduled_tokens = 8192
    config.scheduler_config.max_num_batched_tokens = 8192
    config.scheduler_config.qos_hybrid_alpha = 0.0
    config.scheduler_config.runner_type = "generate"
    config.scheduler_config.is_multimodal_model = False
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.use_ubatching = False
    config.parallel_config.enable_expert_parallel = False
    config.speculative_config = None
    config.kv_transfer_config = None
    config.lora_config = None
    config.model_config.logits_processors = None
    config.model_config.enable_return_routed_experts = False
    config.kv_cache_compression_config = None
    return config


@pytest.fixture
def dummy_qos_request():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    params = SamplingParams(max_tokens=16, temperature=0.0)
    qos = QoSParams(ttft_slo_ms=1000.0)
    return Request(
        request_id="qos-req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
        qos_params=qos,
    )
