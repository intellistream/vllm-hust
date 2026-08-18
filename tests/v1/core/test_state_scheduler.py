# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.state_scheduler import (
    STATE_SCHEDULER_RECEIPT_KEY,
    StateScheduler,
    build_state_scheduler_decision,
)

pytestmark = pytest.mark.cpu_test


def make_request(
    *,
    internal_id: str = "cmpl-req-1-0",
    arm: str = "workflow_scheduler",
    reuse_tokens: int = 512,
    latency_budget_ms: float = 128.0,
    oracle_score: float = 5.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=internal_id,
        sampling_params=SimpleNamespace(
            extra_args={
                "state_scheduler_arm": arm,
                "state_scheduler_request_id": "req-1",
                "state_scheduler_reuse_tokens": reuse_tokens,
                "state_scheduler_latency_budget_ms": latency_budget_ms,
                "state_scheduler_oracle_score": oracle_score,
            }
        ),
    )


def test_workflow_scheduler_uses_state_benefit_density():
    priority, receipt = build_state_scheduler_decision(
        make_request(), effective_arm="workflow_scheduler"
    )

    assert priority == -4_000_000
    assert receipt["decision_score"] == 4.0
    assert receipt["policy_regret"] == 1.0
    assert receipt["request_id"] == "req-1"
    assert receipt["effective_arm"] == "workflow_scheduler"


def test_higher_reuse_density_gets_earlier_native_priority():
    low, _ = build_state_scheduler_decision(
        make_request(reuse_tokens=128), effective_arm="workflow_scheduler"
    )
    high, _ = build_state_scheduler_decision(
        make_request(reuse_tokens=1024), effective_arm="workflow_scheduler"
    )

    assert high < low


@pytest.mark.parametrize(
    ("arm", "expected_priority"),
    [("lru", 0), ("oracle", -5_000_000)],
)
def test_control_arms_have_distinct_native_actions(arm: str, expected_priority: int):
    priority, receipt = build_state_scheduler_decision(
        make_request(arm=arm), effective_arm=arm
    )

    assert priority == expected_priority
    assert receipt["decision_path"]


def test_arm_mismatch_fails_closed():
    with pytest.raises(ValueError, match="does not match"):
        build_state_scheduler_decision(make_request(arm="lru"), effective_arm="oracle")


def test_request_identity_mismatch_fails_closed():
    with pytest.raises(ValueError, match="identity"):
        build_state_scheduler_decision(
            make_request(internal_id="cmpl-other-0"),
            effective_arm="workflow_scheduler",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_scheduler_reuse_tokens", -1),
        ("state_scheduler_latency_budget_ms", 0),
        ("state_scheduler_oracle_score", float("nan")),
    ],
)
def test_invalid_numeric_metadata_fails_closed(field: str, value: float):
    request = make_request()
    request.sampling_params.extra_args[field] = value
    with pytest.raises(ValueError):
        build_state_scheduler_decision(
            request,
            effective_arm="workflow_scheduler",
        )


def test_native_lifecycle_binds_runtime_stats_and_preserves_response_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = StateScheduler.__new__(StateScheduler)
    scheduler.state_scheduler_policy = "workflow_scheduler"
    scheduler.requests = {}
    scheduler.block_size = 16
    request = make_request()
    request.priority = 0
    request.prefill_stats = SimpleNamespace(
        num_prompt_tokens=64,
        num_computed_tokens=33,
    )

    def native_add(self: Scheduler, native_request: Any) -> None:
        self.requests[native_request.request_id] = native_request

    monkeypatch.setattr(Scheduler, "add_request", native_add)
    monkeypatch.setattr(
        Scheduler,
        "schedule",
        lambda self, throttle_prefills=False: SimpleNamespace(
            num_scheduled_tokens={request.request_id: 1}
        ),
    )
    monkeypatch.setattr(
        Scheduler,
        "_free_request",
        lambda self, native_request, delay_free_blocks=False: {
            "native": "preserved"
        },
    )

    scheduler.add_request(request)
    output = scheduler.schedule()
    response = scheduler._free_request(request)

    assert output.num_scheduled_tokens == {request.request_id: 1}
    assert request.priority == -4_000_000
    assert response is not None
    assert response["native"] == "preserved"
    receipt = response[STATE_SCHEDULER_RECEIPT_KEY]
    assert receipt["recompute_blocks"] == 3
    assert receipt["runtime_metrics_bound"] is True
    assert receipt["decision_path"][-1] == "native-prefix-cache-stats"


def test_native_add_failure_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = StateScheduler.__new__(StateScheduler)
    scheduler.state_scheduler_policy = "workflow_scheduler"
    request = make_request()
    request.priority = 0

    def fail_native_add(self: Scheduler, native_request: Any) -> None:
        raise RuntimeError("native add failed")

    monkeypatch.setattr(Scheduler, "add_request", fail_native_add)
    with pytest.raises(RuntimeError, match="native add failed"):
        scheduler.add_request(request)
