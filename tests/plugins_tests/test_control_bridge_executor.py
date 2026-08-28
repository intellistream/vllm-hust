# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from vllm.control_bridge.contracts import ControlActionStatus, parse_control_action
from vllm.control_bridge.executor import (
    ControlBridgeExecutorError,
    ControlBridgeExecutorState,
    materialize_process_isolated_control_bridge,
)
from vllm.control_bridge.runtime_health import (
    RuntimeHealthObservation,
    RuntimeHealthState,
)
from vllm.plugins.contracts import (
    ComponentIsolation,
    ComponentPermission,
    DomainContract,
    ExecutionPlane,
    ExtensionComponentDescriptor,
)
from vllm.plugins.snapshot import ResolvedExtensionComponent


def action(*, include_diagnostics: bool = False):
    return parse_control_action(
        {
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
            "payload": {"include_diagnostics": include_diagnostics},
            "expected_state_version": 11,
            "trace_id": "trace-1",
            "causation_id": None,
        }
    )


def resolved_component(**overrides) -> ResolvedExtensionComponent:
    values = {
        "component_id": "ride-runtime-bridge",
        "contracts": (
            DomainContract.CONTROL_ACTION_V1,
            DomainContract.CONTROL_RECEIPT_V1,
        ),
        "execution_planes": (ExecutionPlane.BRIDGE,),
        "isolation": ComponentIsolation.PROCESS_ISOLATED,
        "implementation_ref": ("vllm.control_bridge.executor:core_health_probe_worker"),
        "permissions": (ComponentPermission.IPC,),
    }
    values.update(overrides)
    return ResolvedExtensionComponent(
        bundle_id="ride-bridge",
        bundle_version="1.0.0",
        component=ExtensionComponentDescriptor(**values),
    )


def materialize(**component_overrides):
    return materialize_process_isolated_control_bridge(
        resolved_component(**component_overrides),
        runtime_id="runtime-a",
        epoch=7,
        state_version=11,
    )


def test_fixed_worker_executes_read_only_probe_in_another_process() -> None:
    executor = materialize()
    with executor:
        parent_pid = __import__("os").getpid()
        receipt = executor.execute(
            action(include_diagnostics=True),
            completed_at=datetime(2026, 8, 29, 0, 2, tzinfo=timezone.utc),
            health_observation=RuntimeHealthObservation(
                state=RuntimeHealthState.HEALTHY,
                observed_at=datetime(2026, 8, 29, 0, 2, tzinfo=timezone.utc),
            ),
        )

        assert executor.state is ControlBridgeExecutorState.READY
        assert executor.worker_pid != parent_pid
        assert receipt.status is ControlActionStatus.APPLIED
        assert receipt.reason_code == "RUNTIME_HEALTHY"
        assert receipt.mutation_occurred is False
        assert receipt.resulting_state_version == 11
        assert f"worker_pid={executor.worker_pid}" in receipt.diagnostic
        unavailable = executor.execute(
            action(),
            completed_at=datetime(2026, 8, 29, 0, 2, tzinfo=timezone.utc),
        )
        assert unavailable.status is ControlActionStatus.FAILED
        assert unavailable.reason_code == "RUNTIME_HEALTH_UNAVAILABLE"
    assert executor.state is ControlBridgeExecutorState.STOPPED


def test_explicit_restart_replaces_worker_generation() -> None:
    executor = materialize()
    executor.start()
    first_pid = executor.worker_pid

    executor.restart()

    assert executor.state is ControlBridgeExecutorState.READY
    assert executor.worker_pid is not None
    assert executor.worker_pid != first_pid
    executor.close()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"isolation": ComponentIsolation.TRUSTED_IN_PROCESS}, "process isolated"),
        ({"permissions": ()}, "exactly IPC"),
        ({"permissions": (ComponentPermission.NETWORK_EGRESS,)}, "exactly IPC"),
        ({"execution_planes": (ExecutionPlane.API,)}, "only in bridge"),
        ({"contracts": (DomainContract.CONTROL_ACTION_V1,)}, "exactly action"),
        ({"implementation_ref": "external.module:worker"}, "not executable"),
    ],
)
def test_materializer_rejects_ambient_or_bundle_supplied_authority(
    override, message
) -> None:
    with pytest.raises(ControlBridgeExecutorError, match=message):
        materialize(**override)


def test_executor_requires_ready_state_and_aware_timestamp() -> None:
    executor = materialize()
    with pytest.raises(ControlBridgeExecutorError, match="not ready"):
        executor.execute(
            action(),
            completed_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        )
    executor.start()
    with pytest.raises(ValueError, match="timezone-aware"):
        executor.execute(action(), completed_at=datetime(2026, 8, 29))
    with pytest.raises(ControlBridgeExecutorError, match="observation is stale"):
        executor.execute(
            action(),
            completed_at=datetime(2026, 8, 29, 0, 2, tzinfo=timezone.utc),
            health_observation=RuntimeHealthObservation(
                state=RuntimeHealthState.HEALTHY,
                observed_at=datetime(2026, 8, 29, 0, 1, tzinfo=timezone.utc),
            ),
        )
    assert executor.state is ControlBridgeExecutorState.READY
    executor.close()


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"target_runtime_id": "runtime-b"}, "another runtime"),
        ({"target_epoch": 8}, "another runtime epoch"),
        ({"expected_state_version": 12}, "precondition is stale"),
    ],
)
def test_executor_rechecks_runtime_owned_preconditions(changed, message) -> None:
    executor = materialize()
    executor.start()
    with pytest.raises(ControlBridgeExecutorError, match=message):
        executor.execute(
            replace(action(), **changed),
            completed_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        )
    assert executor.state is ControlBridgeExecutorState.READY
    executor.close()
