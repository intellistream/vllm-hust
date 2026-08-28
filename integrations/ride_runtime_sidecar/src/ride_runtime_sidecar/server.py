# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mutually authenticated TLS ingress forwarding exact VLCB1 frames to UDS."""

import asyncio
import hashlib
import ssl
from contextlib import suppress

from ride_runtime_sidecar.config import SidecarConfig
from ride_runtime_sidecar.protocol import (
    SidecarProtocolError,
    error_frame,
    read_request_frame,
    read_response_frame,
)


def build_server_ssl_context(config: SidecarConfig) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(config.server_certificate, config.server_private_key)
    context.load_verify_locations(cafile=config.client_ca_bundle)
    return context


class RemoteControlSidecar:
    def __init__(self, config: SidecarConfig) -> None:
        self._config = config
        self._server: asyncio.AbstractServer | None = None
        self._active = 0
        self._lock = asyncio.Lock()
        self._handlers: set[asyncio.Task[object]] = set()

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("sidecar already started")
        self._server = await asyncio.start_server(
            self._handle,
            self._config.listen_host,
            self._config.listen_port,
            ssl=build_server_ssl_context(self._config),
            limit=66 * 1024,
        )

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("sidecar is not started")
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()
        handlers = tuple(self._handlers)
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        admitted = False
        try:
            if not self._peer_allowed(writer):
                await self._write(writer, error_frame("REMOTE_PEER_DENIED"))
                return
            admitted = await self._admit()
            if not admitted:
                await self._write(writer, error_frame("SIDECAR_BUSY"))
                return
            request = await asyncio.wait_for(
                read_request_frame(reader), self._config.read_timeout_seconds
            )
            local_reader, local_writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self._config.local_socket_path),
                self._config.connect_timeout_seconds,
            )
            try:
                local_writer.write(request)
                await local_writer.drain()
                response = await asyncio.wait_for(
                    read_response_frame(local_reader),
                    self._config.response_timeout_seconds,
                )
            finally:
                local_writer.close()
                with suppress(Exception):
                    await local_writer.wait_closed()
            await self._write(writer, response)
        except (
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            SidecarProtocolError,
        ):
            await self._safe_error(writer, "SIDECAR_PROTOCOL_ERROR")
        except (ConnectionError, OSError):
            await self._safe_error(writer, "LOCAL_HOST_UNAVAILABLE")
        finally:
            if admitted:
                async with self._lock:
                    self._active -= 1
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            if task is not None:
                self._handlers.discard(task)

    def _peer_allowed(self, writer: asyncio.StreamWriter) -> bool:
        ssl_object = writer.get_extra_info("ssl_object")
        if ssl_object is None:
            return False
        certificate = ssl_object.getpeercert(binary_form=True)
        if not certificate:
            return False
        fingerprint = hashlib.sha256(certificate).hexdigest()
        return fingerprint in self._config.allowed_peer_sha256

    async def _admit(self) -> bool:
        async with self._lock:
            if self._active >= self._config.max_in_flight:
                return False
            self._active += 1
            return True

    @staticmethod
    async def _write(writer: asyncio.StreamWriter, frame: bytes) -> None:
        writer.write(frame)
        await writer.drain()

    async def _safe_error(self, writer: asyncio.StreamWriter, code: str) -> None:
        with suppress(Exception):
            await self._write(writer, error_frame(code))
