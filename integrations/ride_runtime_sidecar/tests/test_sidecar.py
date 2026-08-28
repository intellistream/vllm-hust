# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import hashlib
import json
import ssl
import struct
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import tomllib
from ride_runtime_sidecar.config import SidecarConfig, load_config
from ride_runtime_sidecar.protocol import (
    MAGIC,
    MAX_PAYLOAD_BYTES,
    SidecarProtocolError,
    error_frame,
    read_request_frame,
    read_response_frame,
)
from ride_runtime_sidecar.server import RemoteControlSidecar


def feed(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_request_frame_is_forwarded_byte_exact() -> None:
    payload = b'{"exact":"bytes"}'
    signature = b"v1;kid=test;sha256=" + b"a" * 64
    frame = (
        struct.pack("!5sHI", MAGIC, len(signature), len(payload)) + signature + payload
    )
    assert await read_request_frame(feed(frame)) == frame


@pytest.mark.asyncio
async def test_oversized_request_is_rejected_before_body_read() -> None:
    header = struct.pack("!5sHI", MAGIC, 1, MAX_PAYLOAD_BYTES + 1)
    with pytest.raises(SidecarProtocolError):
        await read_request_frame(feed(header))


def test_error_frames_are_generic_and_bounded() -> None:
    frame = error_frame("SIDECAR_BUSY")
    (length,) = struct.unpack("!I", frame[:4])
    assert json.loads(frame[4:]) == {"kind": "error", "code": "SIDECAR_BUSY"}
    assert length == len(frame) - 4


def test_peer_allowlist_uses_exact_certificate_sha256(tmp_path: Path) -> None:
    certificate = b"peer certificate DER"
    config = SidecarConfig(
        listen_host="127.0.0.1",
        listen_port=9443,
        local_socket_path=tmp_path / "bridge.sock",
        server_certificate=tmp_path / "server.pem",
        server_private_key=tmp_path / "server.key",
        client_ca_bundle=tmp_path / "ca.pem",
        allowed_peer_sha256=frozenset({hashlib.sha256(certificate).hexdigest()}),
        max_in_flight=1,
        read_timeout_seconds=1,
        connect_timeout_seconds=1,
        response_timeout_seconds=1,
    )
    ssl_object = Mock()
    ssl_object.getpeercert.return_value = certificate
    writer = Mock()
    writer.get_extra_info.return_value = ssl_object
    assert RemoteControlSidecar(config)._peer_allowed(writer)


def test_config_is_closed_and_private_key_must_be_private(tmp_path: Path) -> None:
    files = {}
    for name in ("server.pem", "server.key", "ca.pem"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = path
    files["server.key"].chmod(0o600)
    payload = {
        "schema_version": "1.0",
        "listen_host": "127.0.0.1",
        "listen_port": 9443,
        "local_socket_path": str(tmp_path / "bridge.sock"),
        "server_certificate": str(files["server.pem"]),
        "server_private_key": str(files["server.key"]),
        "client_ca_bundle": str(files["ca.pem"]),
        "allowed_peer_sha256": ["ab" * 32],
        "max_in_flight": 8,
        "read_timeout_seconds": 2,
        "connect_timeout_seconds": 2,
        "response_timeout_seconds": 10,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_config(path).allowed_peer_sha256 == frozenset({"ab" * 32})
    payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_config(path)


def create_certificates(root: Path) -> tuple[Path, Path, Path, Path, str]:
    ca_key, ca_cert = root / "ca.key", root / "ca.pem"
    server_key, server_csr, server_cert = (
        root / "server.key",
        root / "server.csr",
        root / "server.pem",
    )
    client_key, client_csr, client_cert = (
        root / "client.key",
        root / "client.csr",
        root / "client.pem",
    )

    def run(*args: str) -> None:
        subprocess.run(["openssl", *args], check=True, capture_output=True, text=True)

    run(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(ca_key),
        "-out",
        str(ca_cert),
        "-subj",
        "/CN=test-ca",
        "-days",
        "1",
    )
    for name, key, csr, cert in (
        ("localhost", server_key, server_csr, server_cert),
        ("ride-client", client_key, client_csr, client_cert),
    ):
        run(
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(csr),
            "-subj",
            f"/CN={name}",
        )
        run(
            "x509",
            "-req",
            "-in",
            str(csr),
            "-CA",
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-out",
            str(cert),
            "-days",
            "1",
            "-sha256",
        )
    server_key.chmod(0o600)
    der = ssl.PEM_cert_to_DER_cert(client_cert.read_text(encoding="ascii"))
    fingerprint = hashlib.sha256(der).hexdigest()
    return ca_cert, server_cert, server_key, client_cert, fingerprint


@pytest.mark.asyncio
async def test_mtls_peer_forwards_exact_frame_to_local_uds(tmp_path: Path) -> None:
    ca, server_cert, server_key, client_cert, fingerprint = create_certificates(
        tmp_path
    )
    client_key = tmp_path / "client.key"
    socket_path = tmp_path / "local.sock"
    observed: list[bytes] = []

    async def local_host(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        observed.append(await read_request_frame(reader))
        writer.write(error_frame("LOCAL_OK"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    local_server = await asyncio.start_unix_server(local_host, path=socket_path)
    config = SidecarConfig(
        listen_host="127.0.0.1",
        listen_port=0,
        local_socket_path=socket_path,
        server_certificate=server_cert,
        server_private_key=server_key,
        client_ca_bundle=ca,
        allowed_peer_sha256=frozenset({fingerprint}),
        max_in_flight=2,
        read_timeout_seconds=2,
        connect_timeout_seconds=2,
        response_timeout_seconds=2,
    )
    sidecar = RemoteControlSidecar(config)
    await sidecar.start()
    assert sidecar._server is not None
    port = sidecar._server.sockets[0].getsockname()[1]
    client_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca)
    client_context.minimum_version = ssl.TLSVersion.TLSv1_3
    client_context.load_cert_chain(client_cert, client_key)
    payload = b'{"end_to_end":true}'
    signature = b"v1;kid=test;sha256=" + b"b" * 64
    frame = (
        struct.pack("!5sHI", MAGIC, len(signature), len(payload)) + signature + payload
    )
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            port,
            ssl=client_context,
            server_hostname="localhost",
        )
        writer.write(frame)
        await writer.drain()
        assert await read_response_frame(reader) == error_frame("LOCAL_OK")
        writer.close()
        await writer.wait_closed()
    finally:
        await sidecar.stop()
        local_server.close()
        await local_server.wait_closed()
    assert observed == [frame]


def test_config_schema_is_closed_wheel_package_data() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]
    schema_path = root / "src" / "ride_runtime_sidecar" / "sidecar-config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert package_data["ride_runtime_sidecar"] == ["sidecar-config.schema.json"]
    assert schema["additionalProperties"] is False
