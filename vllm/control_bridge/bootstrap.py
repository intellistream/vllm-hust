# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit lifecycle wiring for the local control-bridge host.

The configuration is deliberately local and file-backed. It does not claim to
be a production remote-control-plane transport or secret-management backend.
"""

from __future__ import annotations

import asyncio
import binascii
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pybase64 as base64

from vllm.control_bridge.contracts import ControlAuthorizationScope
from vllm.control_bridge.executor import (
    ProcessIsolatedControlBridgeExecutor,
    materialize_process_isolated_control_bridge,
)
from vllm.control_bridge.keys import ControlHmacKey, ControlKeySet
from vllm.control_bridge.security import PersistentReplayLedger
from vllm.control_bridge.service import LocalControlBridgeService
from vllm.control_bridge.transport import UnixControlBridgeHost
from vllm.engine.protocol import EngineClient
from vllm.plugins.contracts import (
    ComponentIsolation,
    ComponentPermission,
    DomainContract,
    ExecutionPlane,
    ExtensionComponentDescriptor,
)
from vllm.plugins.snapshot import ResolvedExtensionComponent

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "runtime_id",
    "epoch",
    "state_version",
    "socket_path",
    "replay_ledger_path",
    "granted_scopes",
    "key_set",
    "limits",
}
_KEY_SET_FIELDS = {"generation", "keys"}
_KEY_FIELDS = {
    "issuer",
    "key_id",
    "secret_file",
    "not_before",
    "not_after",
    "revoked_at",
}
_LIMIT_FIELDS = {
    "max_in_flight",
    "read_timeout",
    "health_timeout",
    "service_timeout",
    "shutdown_timeout",
    "executor_startup_timeout",
    "executor_request_timeout",
    "executor_shutdown_timeout",
}


class ControlBridgeBootstrapError(ValueError):
    """Reject unsafe, ambiguous, or unsupported host configuration."""


@dataclass(frozen=True, slots=True)
class ControlBridgeHostLimits:
    max_in_flight: int = 8
    read_timeout: float = 2.0
    health_timeout: float = 2.0
    service_timeout: float = 10.0
    shutdown_timeout: float = 10.0
    executor_startup_timeout: float = 30.0
    executor_request_timeout: float = 5.0
    executor_shutdown_timeout: float = 2.0


@dataclass(frozen=True, slots=True)
class ControlBridgeHostConfig:
    runtime_id: str
    epoch: int
    state_version: int
    socket_path: Path
    replay_ledger_path: Path
    granted_scopes: frozenset[ControlAuthorizationScope]
    key_set: ControlKeySet
    limits: ControlBridgeHostLimits


def load_control_bridge_host_config(
    config_path: str | os.PathLike[str],
) -> ControlBridgeHostConfig:
    """Load one closed configuration without importing extension code."""
    path = Path(config_path)
    _validate_config_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlBridgeBootstrapError(
            "control bridge config is unreadable"
        ) from error
    root = _closed_object(payload, _TOP_LEVEL_FIELDS, "config")
    required = _TOP_LEVEL_FIELDS - {"limits"}
    _require_fields(root, required, "config")
    if root["schema_version"] != "1.0":
        raise ControlBridgeBootstrapError("unsupported control bridge schema_version")

    runtime_id = _nonempty_string(root["runtime_id"], "runtime_id")
    epoch = _nonnegative_integer(root["epoch"], "epoch")
    state_version = _nonnegative_integer(root["state_version"], "state_version")
    socket_path = _absolute_path(root["socket_path"], "socket_path")
    ledger_path = _absolute_path(root["replay_ledger_path"], "replay_ledger_path")
    if socket_path == ledger_path:
        raise ControlBridgeBootstrapError("socket and replay ledger paths must differ")

    scopes_raw = root["granted_scopes"]
    if not isinstance(scopes_raw, list) or not scopes_raw:
        raise ControlBridgeBootstrapError("granted_scopes must be a non-empty array")
    try:
        scopes = frozenset(ControlAuthorizationScope(item) for item in scopes_raw)
    except (TypeError, ValueError) as error:
        raise ControlBridgeBootstrapError(
            "granted_scopes contains an unknown scope"
        ) from error
    if len(scopes) != len(scopes_raw) or scopes != {
        ControlAuthorizationScope.RUNTIME_READ
    }:
        raise ControlBridgeBootstrapError(
            "v1 local host grants exactly the runtime.read scope"
        )

    key_set = _parse_key_set(root["key_set"])
    limits = _parse_limits(root.get("limits", {}))
    return ControlBridgeHostConfig(
        runtime_id=runtime_id,
        epoch=epoch,
        state_version=state_version,
        socket_path=socket_path,
        replay_ledger_path=ledger_path,
        granted_scopes=scopes,
        key_set=key_set,
        limits=limits,
    )


class ManagedControlBridgeRuntime:
    """Own and close the local bridge resources in dependency order."""

    def __init__(self, config: ControlBridgeHostConfig, engine: EngineClient) -> None:
        self._config = config
        self._engine = engine
        self._ledger: PersistentReplayLedger | None = None
        self._executor: ProcessIsolatedControlBridgeExecutor | None = None
        self._host: UnixControlBridgeHost | None = None

    @property
    def socket_path(self) -> Path:
        return self._config.socket_path

    async def start(self) -> None:
        if self._host is not None:
            raise RuntimeError("control bridge runtime is already started")
        limits = self._config.limits
        ledger: PersistentReplayLedger | None = None
        executor: ProcessIsolatedControlBridgeExecutor | None = None
        host: UnixControlBridgeHost | None = None
        try:
            ledger = PersistentReplayLedger(self._config.replay_ledger_path)
            executor = materialize_process_isolated_control_bridge(
                _core_bridge_component(),
                runtime_id=self._config.runtime_id,
                epoch=self._config.epoch,
                state_version=self._config.state_version,
                startup_timeout=limits.executor_startup_timeout,
                request_timeout=limits.executor_request_timeout,
                shutdown_timeout=limits.executor_shutdown_timeout,
            )
            await asyncio.to_thread(executor.start)
            service = LocalControlBridgeService(
                runtime_id=self._config.runtime_id,
                epoch=self._config.epoch,
                state_version=self._config.state_version,
                issuer_keys=self._config.key_set,
                granted_scopes=self._config.granted_scopes,
                ledger=ledger,
                executor=executor,
            )
            host = UnixControlBridgeHost(
                socket_path=self._config.socket_path,
                service=service,
                health_client=self._engine,
                max_in_flight=limits.max_in_flight,
                read_timeout=limits.read_timeout,
                health_timeout=limits.health_timeout,
                service_timeout=limits.service_timeout,
                shutdown_timeout=limits.shutdown_timeout,
            )
            await host.start()
        except BaseException:
            if host is not None:
                await host.stop()
            if executor is not None:
                await asyncio.to_thread(executor.close)
            if ledger is not None:
                ledger.close()
            raise
        self._ledger = ledger
        self._executor = executor
        self._host = host

    async def stop(self) -> None:
        host, executor, ledger = self._host, self._executor, self._ledger
        self._host = None
        self._executor = None
        self._ledger = None
        if host is not None:
            await host.stop()
        if executor is not None:
            await asyncio.to_thread(executor.close)
        if ledger is not None:
            ledger.close()


def _core_bridge_component() -> ResolvedExtensionComponent:
    return ResolvedExtensionComponent(
        bundle_id="vllm-core-control-bridge",
        bundle_version="1.0.0",
        component=ExtensionComponentDescriptor(
            component_id="core-health-probe",
            contracts=(
                DomainContract.CONTROL_ACTION_V1,
                DomainContract.CONTROL_RECEIPT_V1,
            ),
            execution_planes=(ExecutionPlane.BRIDGE,),
            isolation=ComponentIsolation.PROCESS_ISOLATED,
            implementation_ref=(
                "vllm.control_bridge.executor:core_health_probe_worker"
            ),
            permissions=(ComponentPermission.IPC,),
        ),
    )


def _parse_key_set(value: Any) -> ControlKeySet:
    item = _closed_object(value, _KEY_SET_FIELDS, "key_set")
    _require_fields(item, _KEY_SET_FIELDS, "key_set")
    generation = _nonnegative_integer(item["generation"], "key_set.generation")
    raw_keys = item["keys"]
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ControlBridgeBootstrapError("key_set.keys must be a non-empty array")
    keys: list[ControlHmacKey] = []
    for index, raw_key in enumerate(raw_keys):
        location = f"key_set.keys[{index}]"
        key = _closed_object(raw_key, _KEY_FIELDS, location)
        _require_fields(key, _KEY_FIELDS - {"not_after", "revoked_at"}, location)
        secret_path = _absolute_path(key["secret_file"], f"{location}.secret_file")
        keys.append(
            ControlHmacKey(
                issuer=_nonempty_string(key["issuer"], f"{location}.issuer"),
                key_id=_nonempty_string(key["key_id"], f"{location}.key_id"),
                secret=_read_secret(secret_path),
                not_before=_timestamp(key["not_before"], f"{location}.not_before"),
                not_after=_optional_timestamp(
                    key.get("not_after"), f"{location}.not_after"
                ),
                revoked_at=_optional_timestamp(
                    key.get("revoked_at"), f"{location}.revoked_at"
                ),
            )
        )
    return ControlKeySet(generation=generation, keys=tuple(keys))


def _parse_limits(value: Any) -> ControlBridgeHostLimits:
    item = _closed_object(value, _LIMIT_FIELDS, "limits")
    defaults = ControlBridgeHostLimits()
    values: dict[str, int | float] = {}
    for name in _LIMIT_FIELDS:
        raw = item.get(name, getattr(defaults, name))
        if name == "max_in_flight":
            if (
                not isinstance(raw, int)
                or isinstance(raw, bool)
                or raw <= 0
                or raw > 64
            ):
                raise ControlBridgeBootstrapError("max_in_flight must be in [1, 64]")
        elif (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or raw <= 0
            or raw > 300
        ):
            raise ControlBridgeBootstrapError(f"limits.{name} must be in (0, 300]")
        values[name] = raw
    return ControlBridgeHostLimits(
        max_in_flight=int(values["max_in_flight"]),
        read_timeout=float(values["read_timeout"]),
        health_timeout=float(values["health_timeout"]),
        service_timeout=float(values["service_timeout"]),
        shutdown_timeout=float(values["shutdown_timeout"]),
        executor_startup_timeout=float(values["executor_startup_timeout"]),
        executor_request_timeout=float(values["executor_request_timeout"]),
        executor_shutdown_timeout=float(values["executor_shutdown_timeout"]),
    )


def _validate_config_file(path: Path) -> None:
    if not path.is_absolute():
        raise ControlBridgeBootstrapError("control bridge config path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ControlBridgeBootstrapError(
            "control bridge config is unavailable"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ControlBridgeBootstrapError(
            "control bridge config must be a regular file"
        )
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise ControlBridgeBootstrapError(
            "control bridge config must be host-owned and not group/world writable"
        )


def _read_secret(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ControlBridgeBootstrapError("control key file is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ControlBridgeBootstrapError("control key file must be a regular file")
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise ControlBridgeBootstrapError(
            "control key file must be host-owned with no group/world access"
        )
    if metadata.st_size > 4096:
        raise ControlBridgeBootstrapError("control key file is too large")
    try:
        encoded = path.read_bytes().strip()
        secret = base64.b64decode(encoded, validate=True)
    except (OSError, binascii.Error) as error:
        raise ControlBridgeBootstrapError(
            "control key file must contain canonical base64"
        ) from error
    if base64.b64encode(secret) != encoded:
        raise ControlBridgeBootstrapError("control key file is not canonical base64")
    if len(secret) < 32:
        raise ControlBridgeBootstrapError("control key must contain at least 32 bytes")
    return secret


def _closed_object(value: Any, allowed: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ControlBridgeBootstrapError(f"{location} must be an object")
    unknown = value.keys() - allowed
    if unknown:
        raise ControlBridgeBootstrapError(
            f"{location} contains unknown fields: {sorted(unknown)}"
        )
    return value


def _require_fields(value: dict[str, Any], required: set[str], location: str) -> None:
    missing = required - value.keys()
    if missing:
        raise ControlBridgeBootstrapError(
            f"{location} is missing fields: {sorted(missing)}"
        )


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlBridgeBootstrapError(f"{location} must be a non-empty string")
    return value


def _nonnegative_integer(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ControlBridgeBootstrapError(f"{location} must be non-negative integer")
    return value


def _absolute_path(value: Any, location: str) -> Path:
    path = Path(_nonempty_string(value, location))
    if not path.is_absolute():
        raise ControlBridgeBootstrapError(f"{location} must be absolute")
    return path


def _timestamp(value: Any, location: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_nonempty_string(value, location))
    except ValueError as error:
        raise ControlBridgeBootstrapError(f"{location} is not ISO 8601") from error
    if parsed.tzinfo is None:
        raise ControlBridgeBootstrapError(f"{location} must include a timezone")
    return parsed


def _optional_timestamp(value: Any, location: str) -> datetime | None:
    return None if value is None else _timestamp(value, location)
