# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from datetime import datetime, timezone

import pybase64 as base64
import pytest
from fastapi import FastAPI

from vllm.control_bridge.bootstrap import (
    ControlBridgeBootstrapError,
    ManagedControlBridgeRuntime,
    load_control_bridge_host_config,
)
from vllm.control_bridge.contracts import ControlAuthorizationScope
from vllm.entrypoints.serve.utils.server_utils import lifespan


def write_config(tmp_path, **overrides):
    secret = tmp_path / "control.key"
    secret.write_bytes(base64.b64encode(b"k" * 32))
    secret.chmod(0o600)
    payload = {
        "schema_version": "1.0",
        "runtime_id": "runtime-a",
        "epoch": 7,
        "state_version": 11,
        "socket_path": str(tmp_path / "control.sock"),
        "replay_ledger_path": str(tmp_path / "replay.sqlite3"),
        "granted_scopes": ["runtime.read"],
        "key_set": {
            "generation": 1,
            "keys": [
                {
                    "issuer": "ride.example",
                    "key_id": "key-1",
                    "secret_file": str(secret),
                    "not_before": "2026-08-29T00:00:00+00:00",
                }
            ],
        },
    }
    payload.update(overrides)
    config = tmp_path / "control.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    config.chmod(0o600)
    return config, secret


def test_closed_config_loads_separate_protected_key(tmp_path) -> None:
    config_path, _ = write_config(tmp_path)

    config = load_control_bridge_host_config(config_path)

    assert config.runtime_id == "runtime-a"
    assert config.granted_scopes == frozenset({ControlAuthorizationScope.RUNTIME_READ})
    assert config.key_set.generation == 1
    assert (
        config.key_set.resolve(
            "ride.example",
            "key-1",
            now=datetime(2026, 8, 29, 0, 1, tzinfo=timezone.utc),
        )
        == b"k" * 32
    )


@pytest.mark.parametrize(
    "override",
    [
        {"unknown": True},
        {"granted_scopes": ["runtime.read", "runtime.write"]},
        {"socket_path": "relative.sock"},
        {"limits": {"max_in_flight": 65}},
    ],
)
def test_config_rejects_unknown_authority_or_unbounded_limits(
    tmp_path, override
) -> None:
    config_path, _ = write_config(tmp_path, **override)
    with pytest.raises(ControlBridgeBootstrapError):
        load_control_bridge_host_config(config_path)


def test_key_file_rejects_ambient_access(tmp_path) -> None:
    config_path, secret = write_config(tmp_path)
    secret.chmod(0o640)

    with pytest.raises(ControlBridgeBootstrapError, match="no group/world access"):
        load_control_bridge_host_config(config_path)


class HealthyEngine:
    async def check_health(self) -> None:
        return None


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="requires POSIX ownership")
@pytest.mark.asyncio
async def test_managed_runtime_owns_socket_worker_and_ledger_lifecycle(
    tmp_path,
) -> None:
    config_path, _ = write_config(tmp_path)
    config = load_control_bridge_host_config(config_path)
    runtime = ManagedControlBridgeRuntime(config, HealthyEngine())

    await runtime.start()
    try:
        assert config.socket_path.is_socket()
        assert config.replay_ledger_path.is_file()
    finally:
        await runtime.stop()

    assert not config.socket_path.exists()
    assert config.replay_ledger_path.is_file()


class RecordingRuntime:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")


@pytest.mark.asyncio
async def test_fastapi_lifespan_starts_and_stops_configured_bridge() -> None:
    app = FastAPI()
    runtime = RecordingRuntime()
    app.state.control_bridge_runtime = runtime
    app.state.log_stats = False

    async with lifespan(app):
        assert runtime.events == ["start"]

    assert runtime.events == ["start", "stop"]
