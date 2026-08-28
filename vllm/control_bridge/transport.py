# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded same-UID Unix transport for local control bridge actions."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import struct
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

from vllm.control_bridge.contracts import (
    ControlActionContractError,
    control_receipt_to_dict,
)
from vllm.control_bridge.runtime_health import (
    EngineHealthClient,
    observe_engine_client_health,
)
from vllm.control_bridge.security import ControlActionAuthenticationError
from vllm.control_bridge.service import LocalControlBridgeService

_REQUEST_HEADER = struct.Struct("!5sHI")
_RESPONSE_HEADER = struct.Struct("!I")
_MAGIC = b"VLCB1"
_MAX_SIGNATURE_BYTES = 256
_MAX_PAYLOAD_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 16 * 1024


class ControlTransportError(RuntimeError):
    """Reject an invalid local transport configuration or lifecycle change."""


class ControlTransportProtocolError(ValueError):
    """Reject malformed or oversized framing before authentication."""


class ControlTransportState(str, Enum):
    """Observable lifecycle for the local Unix request host."""

    NEW = "new"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class UnixControlBridgeHost:
    """Accept bounded concurrent ingress into one serialized authority lane."""

    def __init__(
        self,
        *,
        socket_path: str | os.PathLike[str],
        service: LocalControlBridgeService,
        health_client: EngineHealthClient | None = None,
        max_in_flight: int = 8,
        read_timeout: float = 2.0,
        health_timeout: float = 2.0,
        service_timeout: float = 10.0,
        shutdown_timeout: float = 10.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        if min(read_timeout, health_timeout, service_timeout, shutdown_timeout) <= 0:
            raise ValueError("transport timeouts must be positive")
        if not hasattr(socket, "SO_PEERCRED"):
            raise ControlTransportError("SO_PEERCRED is required")
        self._path = Path(socket_path)
        self._service = service
        self._health_client = health_client
        self._max_in_flight = max_in_flight
        self._read_timeout = read_timeout
        self._health_timeout = health_timeout
        self._service_timeout = service_timeout
        self._shutdown_timeout = shutdown_timeout
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._state = ControlTransportState.NEW
        self._server: asyncio.AbstractServer | None = None
        self._authority_lane: ThreadPoolExecutor | None = None
        self._active = 0
        self._active_lock = asyncio.Lock()
        self._handlers: set[asyncio.Task[Any]] = set()
        self._socket_identity: tuple[int, int] | None = None

    @property
    def state(self) -> ControlTransportState:
        return self._state

    async def __aenter__(self) -> UnixControlBridgeHost:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._state not in {
            ControlTransportState.NEW,
            ControlTransportState.STOPPED,
        }:
            raise ControlTransportError(
                f"cannot start transport from {self._state.value!r}"
            )
        self._validate_socket_target()
        authority_lane = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vllm-control-authority",
        )
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self._path))
            metadata = self._path.lstat()
            self._socket_identity = (metadata.st_dev, metadata.st_ino)
            os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)
            listener.listen(self._max_in_flight)
            listener.setblocking(False)
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                sock=listener,
                limit=_REQUEST_HEADER.size + _MAX_SIGNATURE_BYTES + _MAX_PAYLOAD_BYTES,
            )
        except BaseException:
            listener.close()
            authority_lane.shutdown(wait=False, cancel_futures=True)
            self._state = ControlTransportState.FAILED
            self._remove_owned_socket()
            raise
        self._authority_lane = authority_lane
        self._state = ControlTransportState.READY

    async def stop(self) -> None:
        if self._state is ControlTransportState.STOPPED:
            return
        self._state = ControlTransportState.DRAINING
        server = self._server
        if server is not None:
            server.close()
            await server.wait_closed()
        handlers = tuple(self._handlers)
        if handlers:
            _, pending = await asyncio.wait(
                handlers,
                timeout=self._shutdown_timeout,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        authority_lane = self._authority_lane
        if authority_lane is not None:
            await asyncio.to_thread(
                authority_lane.shutdown,
                wait=True,
                cancel_futures=False,
            )
        self._server = None
        self._authority_lane = None
        self._remove_owned_socket()
        self._state = ControlTransportState.STOPPED

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        admitted = False
        try:
            if not self._same_uid_peer(writer):
                await _write_response(writer, _error_response("PEER_DENIED"))
                return
            admitted = await self._try_admit()
            if not admitted:
                await _write_response(writer, _error_response("SERVER_BUSY"))
                return
            try:
                wire_payload, signature = await asyncio.wait_for(
                    _read_request(reader),
                    timeout=self._read_timeout,
                )
            except (asyncio.TimeoutError, ControlTransportProtocolError):
                await _write_response(writer, _error_response("PROTOCOL_ERROR"))
                return

            observation_started_at = self._clock()
            if observation_started_at.tzinfo is None:
                raise ControlTransportError("transport clock must be timezone-aware")
            health_observation = None
            if self._health_client is not None:
                try:
                    health_observation = await asyncio.wait_for(
                        observe_engine_client_health(
                            self._health_client,
                            observed_at=observation_started_at,
                        ),
                        timeout=self._health_timeout,
                    )
                except asyncio.TimeoutError:
                    health_observation = None
            now = self._clock()
            if now.tzinfo is None:
                raise ControlTransportError("transport clock must be timezone-aware")

            authority_lane = self._authority_lane
            if authority_lane is None:
                raise ControlTransportError("authority lane is unavailable")
            loop = asyncio.get_running_loop()
            call = partial(
                self._service.handle,
                wire_payload,
                signature,
                now=now,
                health_observation=health_observation,
            )
            try:
                authority_future = loop.run_in_executor(authority_lane, call)
                receipt = await asyncio.wait_for(
                    asyncio.shield(authority_future),
                    timeout=self._service_timeout,
                )
            except ControlActionAuthenticationError:
                await _write_response(
                    writer,
                    _error_response("AUTHENTICATION_FAILED"),
                )
                return
            except ControlActionContractError:
                await _write_response(writer, _error_response("ACTION_INVALID"))
                return
            except asyncio.TimeoutError:
                # The authority thread may still commit a durable terminal result.
                release_task = asyncio.create_task(
                    self._release_after_authority(authority_future)
                )
                self._handlers.add(release_task)
                release_task.add_done_callback(self._handlers.discard)
                admitted = False
                await _write_response(writer, _error_response("ACTION_PENDING"))
                return
            await _write_response(
                writer,
                {
                    "kind": "receipt",
                    "receipt": control_receipt_to_dict(receipt),
                },
            )
        except (BrokenPipeError, ConnectionError, asyncio.IncompleteReadError):
            return
        except Exception:
            with suppress(BrokenPipeError, ConnectionError):
                await _write_response(writer, _error_response("INTERNAL_ERROR"))
        finally:
            if admitted:
                await self._release()
            writer.close()
            with suppress(BrokenPipeError, ConnectionError):
                await writer.wait_closed()
            if task is not None:
                self._handlers.discard(task)

    async def _try_admit(self) -> bool:
        async with self._active_lock:
            if self._active >= self._max_in_flight:
                return False
            self._active += 1
            return True

    async def _release(self) -> None:
        async with self._active_lock:
            self._active -= 1

    async def _release_after_authority(
        self,
        authority_future: asyncio.Future[Any],
    ) -> None:
        try:
            await authority_future
        except Exception:
            pass
        finally:
            await self._release()

    @staticmethod
    def _same_uid_peer(writer: asyncio.StreamWriter) -> bool:
        peer_socket = writer.get_extra_info("socket")
        if peer_socket is None:
            return False
        credentials = peer_socket.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        _, uid, _ = struct.unpack("3i", credentials)
        return uid == os.geteuid()

    def _validate_socket_target(self) -> None:
        if not self._path.name:
            raise ControlTransportError("socket path must name a file")
        if not self._path.parent.exists():
            raise ControlTransportError("socket parent directory does not exist")
        parent_metadata = self._path.parent.stat()
        if parent_metadata.st_uid != os.geteuid():
            raise ControlTransportError("socket parent is not owned by this user")
        if parent_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ControlTransportError("socket parent is writable by another user")
        if self._path.exists() or self._path.is_symlink():
            raise ControlTransportError("socket path already exists")

    def _remove_owned_socket(self) -> None:
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            metadata = self._path.lstat()
        except FileNotFoundError:
            return
        if (
            stat.S_ISSOCK(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == identity
        ):
            self._path.unlink()


def encode_control_request(wire_payload: bytes, signature: str) -> bytes:
    """Encode exact action bytes without JSON/base64 normalization."""
    if not isinstance(wire_payload, bytes):
        raise ControlTransportProtocolError("payload must be bytes")
    try:
        signature_bytes = signature.encode("ascii")
    except (AttributeError, UnicodeEncodeError) as error:
        raise ControlTransportProtocolError("signature must be ASCII") from error
    if not 0 < len(signature_bytes) <= _MAX_SIGNATURE_BYTES:
        raise ControlTransportProtocolError("signature length is invalid")
    if not 0 < len(wire_payload) <= _MAX_PAYLOAD_BYTES:
        raise ControlTransportProtocolError("payload length is invalid")
    return (
        _REQUEST_HEADER.pack(
            _MAGIC,
            len(signature_bytes),
            len(wire_payload),
        )
        + signature_bytes
        + wire_payload
    )


async def read_control_response(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one bounded response envelope for tests and local clients."""
    raw_length = await reader.readexactly(_RESPONSE_HEADER.size)
    (length,) = _RESPONSE_HEADER.unpack(raw_length)
    if not 0 < length <= _MAX_RESPONSE_BYTES:
        raise ControlTransportProtocolError("response length is invalid")
    body = await reader.readexactly(length)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlTransportProtocolError("response is not JSON") from error
    if not isinstance(payload, dict):
        raise ControlTransportProtocolError("response is not an object")
    return payload


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[bytes, str]:
    raw_header = await reader.readexactly(_REQUEST_HEADER.size)
    magic, signature_length, payload_length = _REQUEST_HEADER.unpack(raw_header)
    if magic != _MAGIC:
        raise ControlTransportProtocolError("request magic is invalid")
    if not 0 < signature_length <= _MAX_SIGNATURE_BYTES:
        raise ControlTransportProtocolError("signature length is invalid")
    if not 0 < payload_length <= _MAX_PAYLOAD_BYTES:
        raise ControlTransportProtocolError("payload length is invalid")
    signature_bytes = await reader.readexactly(signature_length)
    wire_payload = await reader.readexactly(payload_length)
    try:
        signature = signature_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ControlTransportProtocolError("signature must be ASCII") from error
    return wire_payload, signature


async def _write_response(
    writer: asyncio.StreamWriter,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not 0 < len(body) <= _MAX_RESPONSE_BYTES:
        raise ControlTransportProtocolError("response length is invalid")
    writer.write(_RESPONSE_HEADER.pack(len(body)) + body)
    await writer.drain()


def _error_response(code: str) -> dict[str, str]:
    return {"kind": "error", "code": code}
