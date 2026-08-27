from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vllm.entrypoints.serve.dev.rpc.api_router import (
    qbi_runtime_transition_commit,
    qbi_runtime_transition_rollback,
    qbi_runtime_transition_stage,
    qbi_runtime_transition_state,
    qbi_runtime_transition_verify,
)


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
