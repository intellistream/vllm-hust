# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed configuration and TLS material validation."""

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

_FIELDS = {
    "schema_version",
    "listen_host",
    "listen_port",
    "local_socket_path",
    "server_certificate",
    "server_private_key",
    "client_ca_bundle",
    "allowed_peer_sha256",
    "max_in_flight",
    "read_timeout_seconds",
    "connect_timeout_seconds",
    "response_timeout_seconds",
}


@dataclass(frozen=True)
class SidecarConfig:
    listen_host: str
    listen_port: int
    local_socket_path: Path
    server_certificate: Path
    server_private_key: Path
    client_ca_bundle: Path
    allowed_peer_sha256: frozenset[str]
    max_in_flight: int
    read_timeout_seconds: float
    connect_timeout_seconds: float
    response_timeout_seconds: float


def load_config(path: str | os.PathLike[str]) -> SidecarConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        raise ValueError("config path must be absolute")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("sidecar config fields must exactly match schema v1")
    if payload["schema_version"] != "1.0":
        raise ValueError("unsupported sidecar config version")
    listen_host = payload["listen_host"]
    if not isinstance(listen_host, str) or not listen_host:
        raise ValueError("listen_host must be non-empty")
    listen_port = payload["listen_port"]
    if (
        not isinstance(listen_port, int)
        or isinstance(listen_port, bool)
        or not 1 <= listen_port <= 65535
    ):
        raise ValueError("listen_port is invalid")
    paths = {
        name: _absolute_regular_file(payload[name], name)
        for name in ("server_certificate", "server_private_key", "client_ca_bundle")
    }
    local_socket_path = Path(payload["local_socket_path"])
    if not local_socket_path.is_absolute():
        raise ValueError("local_socket_path must be absolute")
    key_mode = paths["server_private_key"].stat().st_mode
    if key_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("server_private_key must not grant group/world access")
    fingerprints = payload["allowed_peer_sha256"]
    if not isinstance(fingerprints, list) or not fingerprints:
        raise ValueError("allowed_peer_sha256 must be non-empty")
    normalized = frozenset(_fingerprint(value) for value in fingerprints)
    if len(normalized) != len(fingerprints):
        raise ValueError("allowed peer fingerprints must be unique")
    max_in_flight = payload["max_in_flight"]
    if (
        not isinstance(max_in_flight, int)
        or isinstance(max_in_flight, bool)
        or not 1 <= max_in_flight <= 256
    ):
        raise ValueError("max_in_flight must be between 1 and 256")
    timeouts = []
    for name in (
        "read_timeout_seconds",
        "connect_timeout_seconds",
        "response_timeout_seconds",
    ):
        value = payload[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value <= 60
        ):
            raise ValueError(f"{name} must be in (0, 60]")
        timeouts.append(float(value))
    return SidecarConfig(
        listen_host=listen_host,
        listen_port=listen_port,
        local_socket_path=local_socket_path,
        server_certificate=paths["server_certificate"],
        server_private_key=paths["server_private_key"],
        client_ca_bundle=paths["client_ca_bundle"],
        allowed_peer_sha256=normalized,
        max_in_flight=max_in_flight,
        read_timeout_seconds=timeouts[0],
        connect_timeout_seconds=timeouts[1],
        response_timeout_seconds=timeouts[2],
    )


def _absolute_regular_file(value: object, name: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a path")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an absolute regular file")
    return path


def _fingerprint(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("peer fingerprint must be a string")
    normalized = value.lower().replace(":", "")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError("peer fingerprint must be SHA-256 hex")
    return normalized
