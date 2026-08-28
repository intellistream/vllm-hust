# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import hmac
import json
import stat
import threading
from datetime import datetime, timezone

import pytest

from vllm.control_bridge.contracts import (
    ControlAction,
    ControlActionStatus,
    ControlAuthorizationScope,
    ControlReceipt,
)
from vllm.control_bridge.security import PersistentReplayLedger
from vllm.control_bridge.service import LocalControlBridgeService
from vllm.control_bridge.transport import (
    ControlTransportError,
    ControlTransportProtocolError,
    ControlTransportState,
    UnixControlBridgeHost,
    encode_control_request,
    read_control_response,
)

_KEY = b"control-bridge-test-key-at-least-32-bytes"
_NOW = datetime(2026, 8, 29, 0, 1, tzinfo=timezone.utc)


def action_payload() -> dict:
    return {
        "schema_version": "1.0",
        "action_id": "123e4567-e89b-12d3-a456-426614174000",
        "idempotency_key": "probe-1",
        "action_type": "runtime.health_probe",
        "target_runtime_id": "runtime-a",
        "target_epoch": 7,
        "issued_at": "2026-08-29T00:00:00+00:00",
        "expires_at": "2026-08-29T00:05:00+00:00",
        "issuer": "ride.example",
        "authorization_scope": "runtime.read",
        "payload": {"include_diagnostics": False},
        "expected_state_version": 11,
        "trace_id": "trace-1",
        "causation_id": None,
    }


def signed_frame(*, tamper: bool = False) -> bytes:
    wire = json.dumps(action_payload(), sort_keys=True, separators=(",", ":")).encode()
    signature = "sha256=" + hmac.new(_KEY, wire, hashlib.sha256).hexdigest()
    if tamper:
        wire = wire.replace(b"runtime-a", b"runtime-b")
    return encode_control_request(wire, signature)


class RecordingExecutor:
    def __init__(self, *, block: bool = False) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        if not block:
            self.release.set()

    def execute(
        self,
        action: ControlAction,
        *,
        completed_at: datetime,
        health_observation=None,
    ) -> ControlReceipt:
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test executor was not released")
        self.finished.set()
        return ControlReceipt(
            schema_version="1.0",
            action_id=action.action_id,
            runtime_id=action.target_runtime_id,
            observed_epoch=action.target_epoch,
            status=ControlActionStatus.FAILED,
            reason_code="RUNTIME_HEALTH_UNAVAILABLE",
            diagnostic="runtime health observation unavailable",
            mutation_occurred=False,
            resulting_state_version=11,
            completed_at=completed_at,
            trace_id=action.trace_id,
            causation_id=action.causation_id,
        )


def make_service(tmp_path, executor: RecordingExecutor):
    ledger = PersistentReplayLedger(tmp_path / "replay.sqlite3")
    service = LocalControlBridgeService(
        runtime_id="runtime-a",
        epoch=7,
        state_version=11,
        issuer_keys={"ride.example": _KEY},
        granted_scopes=frozenset({ControlAuthorizationScope.RUNTIME_READ}),
        ledger=ledger,
        executor=executor,
    )
    return service, ledger


async def exchange(path, frame: bytes) -> dict:
    reader, writer = await asyncio.open_unix_connection(path)
    writer.write(frame)
    await writer.drain()
    response = await read_control_response(reader)
    writer.close()
    await writer.wait_closed()
    return response


@pytest.mark.asyncio
async def test_same_uid_transport_preserves_exact_bytes_and_cleans_socket(
    tmp_path,
) -> None:
    executor = RecordingExecutor()
    service, ledger = make_service(tmp_path, executor)
    path = tmp_path / "bridge.sock"
    host = UnixControlBridgeHost(
        socket_path=path,
        service=service,
        max_in_flight=2,
        clock=lambda: _NOW,
    )
    try:
        await host.start()
        assert host.state is ControlTransportState.READY
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

        accepted = await exchange(path, signed_frame())
        rejected = await exchange(path, signed_frame(tamper=True))

        assert accepted["kind"] == "receipt"
        assert accepted["receipt"]["reason_code"] == "RUNTIME_HEALTH_UNAVAILABLE"
        assert rejected == {"kind": "error", "code": "AUTHENTICATION_FAILED"}
        assert executor.calls == 1
    finally:
        await host.stop()
        ledger.close()

    assert host.state is ControlTransportState.STOPPED
    assert not path.exists()


@pytest.mark.asyncio
async def test_bounded_ingress_rejects_excess_without_queueing(tmp_path) -> None:
    executor = RecordingExecutor(block=True)
    service, ledger = make_service(tmp_path, executor)
    path = tmp_path / "bridge.sock"
    host = UnixControlBridgeHost(
        socket_path=path,
        service=service,
        max_in_flight=1,
        clock=lambda: _NOW,
    )
    await host.start()
    try:
        first = asyncio.create_task(exchange(path, signed_frame()))
        assert await asyncio.to_thread(executor.started.wait, 3)

        busy = await exchange(path, signed_frame())
        executor.release.set()
        completed = await first

        assert busy == {"kind": "error", "code": "SERVER_BUSY"}
        assert completed["kind"] == "receipt"
        assert executor.calls == 1
    finally:
        executor.release.set()
        await host.stop()
        ledger.close()


@pytest.mark.asyncio
async def test_timed_out_authority_work_keeps_its_in_flight_slot(tmp_path) -> None:
    executor = RecordingExecutor(block=True)
    service, ledger = make_service(tmp_path, executor)
    path = tmp_path / "bridge.sock"
    host = UnixControlBridgeHost(
        socket_path=path,
        service=service,
        max_in_flight=1,
        service_timeout=0.05,
        clock=lambda: _NOW,
    )
    await host.start()
    try:
        pending = await exchange(path, signed_frame())
        busy = await exchange(path, signed_frame())

        assert pending == {"kind": "error", "code": "ACTION_PENDING"}
        assert busy == {"kind": "error", "code": "SERVER_BUSY"}

        executor.release.set()
        assert await asyncio.to_thread(executor.finished.wait, 3)
        for _ in range(50):
            recovered = await exchange(path, signed_frame())
            if recovered["kind"] == "receipt":
                break
            assert recovered == {"kind": "error", "code": "SERVER_BUSY"}
            await asyncio.sleep(0.01)

        assert recovered["kind"] == "receipt"
        assert executor.calls == 1
    finally:
        executor.release.set()
        await host.stop()
        ledger.close()


@pytest.mark.asyncio
async def test_existing_socket_target_is_never_removed(tmp_path) -> None:
    executor = RecordingExecutor()
    service, ledger = make_service(tmp_path, executor)
    path = tmp_path / "bridge.sock"
    path.write_text("owner data", encoding="utf-8")
    host = UnixControlBridgeHost(socket_path=path, service=service)
    try:
        with pytest.raises(ControlTransportError, match="already exists"):
            await host.start()
    finally:
        ledger.close()

    assert path.read_text(encoding="utf-8") == "owner data"


@pytest.mark.asyncio
async def test_insecure_socket_parent_is_rejected(tmp_path) -> None:
    executor = RecordingExecutor()
    service, ledger = make_service(tmp_path, executor)
    insecure = tmp_path / "insecure"
    insecure.mkdir(mode=0o777)
    insecure.chmod(0o777)
    host = UnixControlBridgeHost(
        socket_path=insecure / "bridge.sock",
        service=service,
    )
    try:
        with pytest.raises(ControlTransportError, match="writable"):
            await host.start()
    finally:
        ledger.close()


def test_request_encoder_rejects_oversized_or_non_ascii_inputs() -> None:
    with pytest.raises(ControlTransportProtocolError, match="payload length"):
        encode_control_request(b"x" * (64 * 1024 + 1), "sha256=" + "0" * 64)
    with pytest.raises(ControlTransportProtocolError, match="ASCII"):
        encode_control_request(b"{}", "签名")
