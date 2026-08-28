# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Core-owned, process-isolated executor for read-only control actions.

The child process executes no bundle-supplied code, and the HMAC key mapping is
never sent over IPC. This is a failure-isolation boundary, not an
operating-system security sandbox.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from datetime import datetime
from enum import Enum
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from vllm.control_bridge.contracts import (
    ControlAction,
    ControlActionContractError,
    ControlActionStatus,
    ControlActionType,
    ControlReceipt,
    control_action_to_dict,
    control_receipt_to_dict,
    parse_control_action,
    parse_control_receipt,
)
from vllm.control_bridge.runtime_health import (
    RuntimeHealthObservation,
    RuntimeHealthState,
    health_observation_to_dict,
    parse_health_observation,
)
from vllm.plugins.contracts import (
    ComponentIsolation,
    ComponentPermission,
    DomainContract,
    ExecutionPlane,
)
from vllm.plugins.snapshot import ResolvedExtensionComponent

_CORE_HEALTH_PROBE_IMPLEMENTATION = (
    "vllm.control_bridge.executor:core_health_probe_worker"
)
_MAX_HEALTH_OBSERVATION_AGE_SECONDS = 5.0


class ControlBridgeExecutorError(RuntimeError):
    """Reject an invalid materialization or failed executor operation."""


class ControlBridgeBackpressureError(ControlBridgeExecutorError):
    """Reject work when the single bounded request slot is occupied."""


class ControlBridgeExecutorState(str, Enum):
    """Observable local lifecycle states for the fixed bridge worker."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class ProcessIsolatedControlBridgeExecutor:
    """Execute one bounded read-only request at a time in a spawned process."""

    def __init__(
        self,
        *,
        runtime_id: str,
        epoch: int,
        state_version: int,
        startup_timeout: float = 30.0,
        request_timeout: float = 5.0,
        shutdown_timeout: float = 2.0,
    ) -> None:
        if not runtime_id:
            raise ValueError("runtime_id must be non-empty")
        if epoch < 0 or state_version < 0:
            raise ValueError("epoch and state_version must be non-negative")
        if min(startup_timeout, request_timeout, shutdown_timeout) <= 0:
            raise ValueError("executor timeouts must be positive")
        self._runtime_id = runtime_id
        self._epoch = epoch
        self._state_version = state_version
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._shutdown_timeout = shutdown_timeout
        self._context = multiprocessing.get_context("spawn")
        self._state = ControlBridgeExecutorState.NEW
        self._parent_connection: Connection | None = None
        self._process: BaseProcess | None = None
        self._request_slot = threading.Lock()

    @property
    def state(self) -> ControlBridgeExecutorState:
        return self._state

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.is_alive() else None

    def __enter__(self) -> ProcessIsolatedControlBridgeExecutor:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def start(self) -> None:
        """Spawn the fixed core worker and require an explicit ready handshake."""
        if self._state not in {
            ControlBridgeExecutorState.NEW,
            ControlBridgeExecutorState.STOPPED,
            ControlBridgeExecutorState.FAILED,
        }:
            raise ControlBridgeExecutorError(
                f"cannot start executor from state {self._state.value!r}"
            )
        self._cleanup_process()
        self._state = ControlBridgeExecutorState.STARTING
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=core_health_probe_worker,
            args=(
                child,
                self._runtime_id,
                self._epoch,
                self._state_version,
            ),
            name="vllm-control-bridge",
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            process.close()
            self._state = ControlBridgeExecutorState.FAILED
            raise
        child.close()
        self._parent_connection = parent
        self._process = process
        try:
            if not parent.poll(self._startup_timeout):
                raise ControlBridgeExecutorError("bridge worker startup timed out")
            try:
                message = parent.recv()
            except EOFError as error:
                raise ControlBridgeExecutorError(
                    "bridge worker exited during startup"
                ) from error
            if message != {"kind": "ready"}:
                raise ControlBridgeExecutorError("bridge worker handshake failed")
        except BaseException:
            self._state = ControlBridgeExecutorState.FAILED
            self._terminate_process()
            raise
        self._state = ControlBridgeExecutorState.READY

    def execute(
        self,
        action: ControlAction,
        *,
        completed_at: datetime,
        health_observation: RuntimeHealthObservation | None = None,
    ) -> ControlReceipt:
        """Execute a previously authenticated, admitted, and reserved action."""
        if self._state is not ControlBridgeExecutorState.READY:
            raise ControlBridgeExecutorError(
                f"executor is not ready: {self._state.value!r}"
            )
        if completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        self._validate_action_preconditions(action)
        self._validate_health_observation(health_observation, completed_at)
        if not self._request_slot.acquire(blocking=False):
            raise ControlBridgeBackpressureError("bridge request slot is occupied")
        try:
            connection = self._require_live_connection()
            connection.send(
                {
                    "kind": "execute",
                    "action": control_action_to_dict(action),
                    "completed_at": completed_at.isoformat(),
                    "health_observation": (
                        None
                        if health_observation is None
                        else health_observation_to_dict(health_observation)
                    ),
                }
            )
            if not connection.poll(self._request_timeout):
                self._fail_worker()
                raise ControlBridgeExecutorError("bridge request timed out")
            try:
                response = connection.recv()
            except EOFError as error:
                self._fail_worker()
                raise ControlBridgeExecutorError("bridge worker exited") from error
            if not isinstance(response, dict) or response.get("kind") != "receipt":
                self._fail_worker()
                raise ControlBridgeExecutorError("bridge worker response is invalid")
            try:
                receipt = parse_control_receipt(response["receipt"])
            except (KeyError, ControlActionContractError) as error:
                self._fail_worker()
                raise ControlBridgeExecutorError(
                    "bridge worker returned an invalid receipt"
                ) from error
            self._validate_receipt_correlation(action, receipt)
            return receipt
        finally:
            self._request_slot.release()

    def close(self) -> None:
        """Drain the request slot, request shutdown, then terminate on timeout."""
        if self._state is ControlBridgeExecutorState.STOPPED:
            return
        with self._request_slot:
            process = self._process
            connection = self._parent_connection
            if process is not None and process.is_alive() and connection is not None:
                self._state = ControlBridgeExecutorState.DRAINING
                try:
                    connection.send({"kind": "shutdown"})
                    if connection.poll(self._shutdown_timeout):
                        connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._terminate_process()
            self._state = ControlBridgeExecutorState.STOPPED

    def restart(self) -> None:
        """Explicitly replace the child after a failed or drained generation."""
        self.close()
        self.start()

    def _require_live_connection(self) -> Connection:
        process = self._process
        connection = self._parent_connection
        if process is None or connection is None or not process.is_alive():
            self._fail_worker()
            raise ControlBridgeExecutorError("bridge worker is not alive")
        return connection

    def _validate_receipt_correlation(
        self, action: ControlAction, receipt: ControlReceipt
    ) -> None:
        if (
            receipt.action_id != action.action_id
            or receipt.runtime_id != self._runtime_id
            or receipt.observed_epoch != self._epoch
            or receipt.trace_id != action.trace_id
            or receipt.causation_id != action.causation_id
        ):
            self._fail_worker()
            raise ControlBridgeExecutorError("bridge receipt correlation failed")

    def _validate_action_preconditions(self, action: ControlAction) -> None:
        if action.target_runtime_id != self._runtime_id:
            raise ControlBridgeExecutorError("action targets another runtime")
        if action.target_epoch != self._epoch:
            raise ControlBridgeExecutorError("action targets another runtime epoch")
        if (
            action.expected_state_version is not None
            and action.expected_state_version != self._state_version
        ):
            raise ControlBridgeExecutorError("action state precondition is stale")

    def _validate_health_observation(
        self,
        observation: RuntimeHealthObservation | None,
        completed_at: datetime,
    ) -> None:
        if observation is None:
            return
        age = (completed_at - observation.observed_at).total_seconds()
        if age < 0:
            raise ControlBridgeExecutorError("health observation is from the future")
        if age > _MAX_HEALTH_OBSERVATION_AGE_SECONDS:
            raise ControlBridgeExecutorError("health observation is stale")

    def _fail_worker(self) -> None:
        self._state = ControlBridgeExecutorState.FAILED
        self._terminate_process()

    def _terminate_process(self) -> None:
        process = self._process
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(self._shutdown_timeout)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(self._shutdown_timeout)
        self._cleanup_process()

    def _cleanup_process(self) -> None:
        connection = self._parent_connection
        if connection is not None:
            connection.close()
        process = self._process
        if process is not None:
            process.close()
        self._parent_connection = None
        self._process = None


def materialize_process_isolated_control_bridge(
    resolved: ResolvedExtensionComponent,
    *,
    runtime_id: str,
    epoch: int,
    state_version: int,
    startup_timeout: float = 30.0,
    request_timeout: float = 5.0,
    shutdown_timeout: float = 2.0,
) -> ProcessIsolatedControlBridgeExecutor:
    """Materialize only the fixed, least-authority v1 bridge component."""
    component = resolved.component
    expected_contracts = {
        DomainContract.CONTROL_ACTION_V1,
        DomainContract.CONTROL_RECEIPT_V1,
    }
    if set(component.contracts) != expected_contracts:
        raise ControlBridgeExecutorError(
            "control bridge must implement exactly action and receipt v1"
        )
    if component.execution_planes != (ExecutionPlane.BRIDGE,):
        raise ControlBridgeExecutorError("control bridge must run only in bridge plane")
    if component.isolation is not ComponentIsolation.PROCESS_ISOLATED:
        raise ControlBridgeExecutorError("control bridge must be process isolated")
    if component.permissions != (ComponentPermission.IPC,):
        raise ControlBridgeExecutorError(
            "core health-probe bridge must request exactly IPC permission"
        )
    if component.implementation_ref != _CORE_HEALTH_PROBE_IMPLEMENTATION:
        raise ControlBridgeExecutorError(
            "bundle-supplied bridge implementations are not executable in v1"
        )
    return ProcessIsolatedControlBridgeExecutor(
        runtime_id=runtime_id,
        epoch=epoch,
        state_version=state_version,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        shutdown_timeout=shutdown_timeout,
    )


def core_health_probe_worker(
    connection: Connection,
    runtime_id: str,
    epoch: int,
    state_version: int,
) -> None:
    """Fixed child entry point; never import a bundle implementation reference."""
    connection.send({"kind": "ready"})
    try:
        while True:
            message = connection.recv()
            if not isinstance(message, dict):
                raise ControlBridgeExecutorError("worker request is not an object")
            kind = message.get("kind")
            if kind == "shutdown":
                connection.send({"kind": "stopped"})
                return
            if kind != "execute":
                raise ControlBridgeExecutorError("worker request kind is invalid")
            action = parse_control_action(message["action"])
            completed_at = datetime.fromisoformat(message["completed_at"])
            if completed_at.tzinfo is None:
                raise ControlBridgeExecutorError(
                    "worker completion timestamp has no timezone"
                )
            if action.action_type is not ControlActionType.RUNTIME_HEALTH_PROBE:
                raise ControlBridgeExecutorError("worker action is not read-only")
            raw_observation = message.get("health_observation")
            observation = (
                None
                if raw_observation is None
                else parse_health_observation(raw_observation)
            )
            status = ControlActionStatus.FAILED
            reason_code = "RUNTIME_HEALTH_UNAVAILABLE"
            diagnostic = "runtime health observation unavailable"
            if observation is not None:
                if observation.observed_at > completed_at:
                    raise ControlBridgeExecutorError(
                        "health observation is newer than completion"
                    )
                if (
                    completed_at - observation.observed_at
                ).total_seconds() > _MAX_HEALTH_OBSERVATION_AGE_SECONDS:
                    raise ControlBridgeExecutorError("health observation is stale")
                if observation.state is RuntimeHealthState.HEALTHY:
                    status = ControlActionStatus.APPLIED
                    reason_code = "RUNTIME_HEALTHY"
                    diagnostic = "runtime health check passed"
                elif observation.state is RuntimeHealthState.UNHEALTHY:
                    reason_code = "RUNTIME_UNHEALTHY"
                    diagnostic = "runtime health check reported engine dead"
            if action.payload.include_diagnostics:
                source = "none" if observation is None else observation.source
                diagnostic += f"; source={source}; worker_pid={os.getpid()}"
            receipt = ControlReceipt(
                schema_version="1.0",
                action_id=action.action_id,
                runtime_id=runtime_id,
                observed_epoch=epoch,
                status=status,
                reason_code=reason_code,
                diagnostic=diagnostic,
                mutation_occurred=False,
                resulting_state_version=state_version,
                completed_at=completed_at,
                trace_id=action.trace_id,
                causation_id=action.causation_id,
            )
            connection.send(
                {"kind": "receipt", "receipt": control_receipt_to_dict(receipt)}
            )
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()
