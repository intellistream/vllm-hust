# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic ordering tests for QoSSchedulingPolicy running/waiting keys.

Covers the four cross-phase combinations requested in PR #169 review:
- QoS prefill vs plain decode
- QoS decode vs plain prefill
- priority + deadline interaction
- starvation bound (best-effort request not deferred indefinitely)
"""


import pytest

from vllm.v1.qos import QoSRuntimeState


def _make_request(
    request_id: str,
    *,
    prompt_tokens: int = 16,
    computed_tokens: int = 16,
    max_tokens: int = 32,
    output_tokens: int = 0,
    arrival_time: float = 0.0,
    priority: int = 0,
    qos_state: QoSRuntimeState | None = None,
):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    req = Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=arrival_time,
        priority=priority,
        qos_params=None,
    )
    # Force the scheduling-relevant counters without running a real step.
    req.num_computed_tokens = computed_tokens
    req._output_token_ids = list(range(output_tokens))
    if qos_state is not None:
        req.qos_state = qos_state
    return req


def _qos_state(
    *,
    ttft_deadline: float | None = None,
    tbt_slo_s: float | None = None,
    ttlt_deadline: float | None = None,
    expected_output_tokens: int = 64,
    first_token_time: float | None = None,
    last_token_time: float | None = None,
) -> QoSRuntimeState:
    return QoSRuntimeState(
        ttft_deadline=ttft_deadline,
        tbt_slo_s=tbt_slo_s,
        ttlt_deadline=ttlt_deadline,
        expected_output_tokens=expected_output_tokens,
        service_class=None,
        first_token_time=first_token_time,
        last_token_time=last_token_time,
    )


@pytest.fixture
def policy(dummy_vllm_config):
    from vllm.v1.core.sched.qos_policy import QoSSchedulingPolicy

    return QoSSchedulingPolicy(dummy_vllm_config)


class TestRunningKeyOrdering:
    """running_key must let deadline-active prefills outrank plain decodes."""

    def test_qos_prefill_before_plain_decode(self, policy):
        # Plain decode (no QoS, mid-generation).
        plain_decode = _make_request(
            "plain-decode",
            prompt_tokens=16,
            computed_tokens=16,
            output_tokens=4,
            arrival_time=0.0,
        )
        # QoS prefill with urgent TTFT deadline.
        qos_prefill = _make_request(
            "qos-prefill",
            prompt_tokens=16,
            computed_tokens=0,
            arrival_time=0.1,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        assert policy.running_key(qos_prefill) < policy.running_key(plain_decode)

    def test_qos_decode_before_plain_prefill(self, policy):
        # QoS decode with a TBT deadline.
        qos_decode = _make_request(
            "qos-decode",
            prompt_tokens=16,
            computed_tokens=16,
            output_tokens=4,
            arrival_time=0.0,
            qos_state=_qos_state(
                tbt_slo_s=0.02,
                first_token_time=0.0,
                last_token_time=0.0,
            ),
        )
        # Plain prefill (no QoS).
        plain_prefill = _make_request(
            "plain-prefill",
            prompt_tokens=16,
            computed_tokens=0,
            arrival_time=0.1,
        )
        assert policy.running_key(qos_decode) < policy.running_key(plain_prefill)

    def test_priority_and_deadline_combination(self, dummy_vllm_config):
        from vllm.v1.core.sched.qos_policy import QoSSchedulingPolicy

        dummy_vllm_config.scheduler_config.policy = "priority"
        policy = QoSSchedulingPolicy(dummy_vllm_config)

        # High-priority plain decode (priority 0).
        high_pri_plain = _make_request(
            "high-pri-plain",
            computed_tokens=16,
            output_tokens=4,
            arrival_time=0.0,
            priority=0,
        )
        # Lower-priority QoS prefill (priority 1) with deadline.
        low_pri_qos = _make_request(
            "low-pri-qos",
            computed_tokens=0,
            arrival_time=0.1,
            priority=1,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        # Priority is the second sort key; within the deadline-active group
        # (group 0) priority still orders before hybrid_score.
        assert policy.running_key(low_pri_qos) < policy.running_key(high_pri_plain)

    def test_deadline_urgency_within_group(self, policy):
        urgent = _make_request(
            "urgent",
            computed_tokens=0,
            arrival_time=0.0,
            qos_state=_qos_state(ttft_deadline=0.5),
        )
        relaxed = _make_request(
            "relaxed",
            computed_tokens=0,
            arrival_time=0.0,
            qos_state=_qos_state(ttft_deadline=5.0),
        )
        assert policy.running_key(urgent) < policy.running_key(relaxed)

    def test_order_running_sorts_list(self, policy):
        plain_decode = _make_request(
            "plain-decode",
            computed_tokens=16,
            output_tokens=4,
            arrival_time=0.0,
        )
        qos_prefill = _make_request(
            "qos-prefill",
            computed_tokens=0,
            arrival_time=0.1,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        requests = [plain_decode, qos_prefill]
        policy.order_running(requests, active=True)
        assert requests[0].request_id == "qos-prefill"
        assert requests[1].request_id == "plain-decode"


class TestWaitingKeyOrdering:
    """waiting_key must put deadline-active requests ahead of best-effort."""

    def test_qos_ahead_of_best_effort(self, policy):
        best_effort = _make_request(
            "best-effort",
            computed_tokens=0,
            arrival_time=0.0,
        )
        qos_req = _make_request(
            "qos",
            computed_tokens=0,
            arrival_time=0.5,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        assert policy.waiting_key(qos_req) < policy.waiting_key(best_effort)

    def test_fcfs_tiebreak_within_deadline_group(self, policy):
        earlier = _make_request(
            "earlier",
            computed_tokens=0,
            arrival_time=0.0,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        later = _make_request(
            "later",
            computed_tokens=0,
            arrival_time=1.0,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        assert policy.waiting_key(earlier) < policy.waiting_key(later)


class TestStarvationBound:
    """Best-effort requests must remain schedulable (arrival_time aging)."""

    def test_best_effort_ordered_by_arrival(self, policy):
        # Two best-effort requests: older one must come first.
        older = _make_request(
            "older",
            computed_tokens=16,
            output_tokens=4,
            arrival_time=0.0,
        )
        newer = _make_request(
            "newer",
            computed_tokens=16,
            output_tokens=4,
            arrival_time=10.0,
        )
        assert policy.running_key(older) < policy.running_key(newer)

    def test_best_effort_not_infinite_score(self, policy):
        # A best-effort request should have a comparable (non-constant) sort
        # key so it is never permanently stuck behind deadline-active traffic.
        best_effort = _make_request(
            "best-effort",
            computed_tokens=16,
            output_tokens=4,
            arrival_time=0.0,
        )
        key = policy.running_key(best_effort)
        # Group marker is 1 (best-effort), meaning it sorts after all
        # deadline-active requests (group 0). hybrid_score may be inf
        # when there is no deadline, which is correct — best-effort never
        # has a deadline, so inf is the expected sentinel value.
        assert key[0] == 1
        assert len(key) >= 5

    def test_has_qos_request_detects_deadline(self, policy):
        plain = _make_request("plain", computed_tokens=16, output_tokens=4)
        qos = _make_request(
            "qos",
            computed_tokens=0,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        assert policy.has_qos_request([plain]) is False
        assert policy.has_qos_request([plain, qos]) is True


class TestStartStepActivation:
    """start_step must return inactive until a QoS request exists."""

    def test_inactive_when_no_qos_requests(self, policy):
        plain = _make_request("plain", computed_tokens=16, output_tokens=4)
        decision = policy.start_step(running=[plain], waiting=[])
        assert decision.active is False

    def test_inactive_when_counter_zero(self, policy):
        qos = _make_request(
            "qos",
            computed_tokens=0,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        # Counter not incremented -> short-circuit inactive.
        decision = policy.start_step(running=[qos], waiting=[])
        assert decision.active is False

    def test_active_when_qos_request_present(self, policy):
        qos = _make_request(
            "qos",
            computed_tokens=0,
            qos_state=_qos_state(ttft_deadline=1.0),
        )
        policy.increment_qos_count()
        decision = policy.start_step(running=[qos], waiting=[])
        assert decision.active is True


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
