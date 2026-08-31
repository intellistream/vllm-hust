from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vllm.entrypoints.serve.rpc.api_router import (
    qbi_dynamic_mtp_state,
    qbi_prefix_cache_policy_receipts,
    qbi_priority_scheduling_state,
    qbi_reasoning_budget_state,
    qbi_runtime_transition_commit,
    qbi_runtime_transition_rollback,
    qbi_runtime_transition_stage,
    qbi_runtime_transition_state,
    qbi_runtime_transition_verify,
)

pytestmark = pytest.mark.skip_global_cleanup


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        return {"method": method, "args": list(args)}


class _FakeRequest:
    def __init__(self, body=None):
        self._body = body
        self.client = _FakeClient()
        self.app = SimpleNamespace(state=SimpleNamespace(engine_client=self.client))

    async def json(self):
        return self._body


class _WrappedFakeRequest(_FakeRequest):
    def __init__(self, body=None):
        super().__init__(body)
        self.app.state.engine_client = SimpleNamespace(engine_core=self.client)


@pytest.mark.asyncio
async def test_runtime_transition_routes_use_explicit_engine_utilities():
    state_request = _FakeRequest()
    response = await qbi_runtime_transition_state(state_request)
    assert response.status_code == 200
    assert state_request.client.calls == [("get_runtime_transition_state", ())]

    config = {"max_num_scheduled_tokens": 1024, "max_num_running_reqs": 2}
    stage_request = _FakeRequest(
        {"mechanism_name": "scheduler_batching", "config": config}
    )
    await qbi_runtime_transition_stage(stage_request)
    assert stage_request.client.calls == [
        ("stage_runtime_transition", ("scheduler_batching", config))
    ]

    commit_request = _FakeRequest(
        {"mechanism_name": "scheduler_batching", "next_epoch": 1}
    )
    await qbi_runtime_transition_commit(commit_request)
    assert commit_request.client.calls == [
        ("commit_staged_runtime_transition", ("scheduler_batching", 1))
    ]

    verify_request = _FakeRequest(
        {"mechanism_name": "scheduler_batching", "config": config, "epoch": 1}
    )
    await qbi_runtime_transition_verify(verify_request)
    assert verify_request.client.calls == [
        ("verify_runtime_transition", ("scheduler_batching", config, 1))
    ]

    prepared = {"previous": config, "requested": config, "previous_epoch": 0}
    rollback_request = _FakeRequest(
        {
            "mechanism_name": "scheduler_batching",
            "prepared": prepared,
            "previous_epoch": 0,
        }
    )
    await qbi_runtime_transition_rollback(rollback_request)
    assert rollback_request.client.calls == [
        (
            "rollback_runtime_transition",
            ("scheduler_batching", prepared, 0),
        )
    ]


@pytest.mark.asyncio
async def test_prefill_guard_uses_same_safe_epoch_routes():
    config = {"scheduler_reserve_full_isl": False}
    stage_request = _FakeRequest(
        {"mechanism_name": "prefill_admission_guard", "config": config}
    )
    await qbi_runtime_transition_stage(stage_request)
    assert stage_request.client.calls == [
        ("stage_runtime_transition", ("prefill_admission_guard", config))
    ]

    commit_request = _FakeRequest(
        {"mechanism_name": "prefill_admission_guard", "next_epoch": 2}
    )
    await qbi_runtime_transition_commit(commit_request)
    assert commit_request.client.calls == [
        ("commit_staged_runtime_transition", ("prefill_admission_guard", 2))
    ]

    verify_request = _FakeRequest(
        {
            "mechanism_name": "prefill_admission_guard",
            "config": config,
            "epoch": 2,
        }
    )
    await qbi_runtime_transition_verify(verify_request)
    assert verify_request.client.calls == [
        ("verify_runtime_transition", ("prefill_admission_guard", config, 2))
    ]


@pytest.mark.asyncio
async def test_mechanism_telemetry_routes_use_explicit_engine_utilities():
    priority_request = _FakeRequest()
    await qbi_priority_scheduling_state(priority_request, after_sequence=4)
    assert priority_request.client.calls == [("get_priority_scheduling_state", (4,))]

    apc_request = _FakeRequest()
    await qbi_prefix_cache_policy_receipts(apc_request)
    assert apc_request.client.calls == [("get_prefix_cache_policy_receipts", ())]

    mtp_request = _FakeRequest()
    await qbi_dynamic_mtp_state(mtp_request, after_sequence=5)
    assert mtp_request.client.calls == [("get_dynamic_mtp_state", (5,))]

    reasoning_request = _FakeRequest()
    await qbi_reasoning_budget_state(reasoning_request, after_sequence=6)
    assert reasoning_request.client.calls == [("get_reasoning_budget_state", (6,))]


@pytest.mark.asyncio
async def test_runtime_transition_routes_unwrap_openai_engine_client():
    request = _WrappedFakeRequest()
    response = await qbi_runtime_transition_state(request)
    assert response.status_code == 200
    assert request.client.calls == [("get_runtime_transition_state", ())]


@pytest.mark.asyncio
async def test_runtime_transition_route_rejects_unknown_mechanism_and_fields():
    unknown = _FakeRequest(
        {
            "mechanism_name": "arbitrary_utility",
            "config": {"max_num_scheduled_tokens": 1, "max_num_running_reqs": 1},
        }
    )
    with pytest.raises(HTTPException, match="scheduler_batching"):
        await qbi_runtime_transition_stage(unknown)
    assert unknown.client.calls == []

    extra = _FakeRequest(
        {
            "mechanism_name": "scheduler_batching",
            "config": {
                "max_num_scheduled_tokens": 1,
                "max_num_running_reqs": 1,
                "unexpected": 1,
            },
        }
    )
    with pytest.raises(HTTPException, match="unexpected or missing"):
        await qbi_runtime_transition_stage(extra)
    assert extra.client.calls == []

    bool_scheduler = _FakeRequest(
        {
            "mechanism_name": "scheduler_batching",
            "config": {
                "max_num_scheduled_tokens": True,
                "max_num_running_reqs": 1,
            },
        }
    )
    with pytest.raises(HTTPException, match="must be integers"):
        await qbi_runtime_transition_stage(bool_scheduler)

    non_bool_guard = _FakeRequest(
        {
            "mechanism_name": "prefill_admission_guard",
            "config": {"scheduler_reserve_full_isl": 0},
        }
    )
    with pytest.raises(HTTPException, match="must be a boolean"):
        await qbi_runtime_transition_stage(non_bool_guard)
