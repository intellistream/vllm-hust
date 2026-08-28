# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.plugins.contracts import (
    ComponentIsolation,
    ComponentPermission,
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)


def test_bundle_can_place_kv_components_in_distinct_processes() -> None:
    scheduler = ExtensionComponentDescriptor(
        component_id="scheduler-connector",
        contracts=(DomainContract.KV_CONNECTOR_SCHEDULER_V1,),
        execution_planes=(ExecutionPlane.SCHEDULER,),
        isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
        implementation_ref="example.connector:SchedulerConnector",
    )
    worker = ExtensionComponentDescriptor(
        component_id="worker-connector",
        contracts=(DomainContract.KV_CONNECTOR_WORKER_V1,),
        execution_planes=(ExecutionPlane.WORKER, ExecutionPlane.DEVICE),
        isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
        implementation_ref="example.connector:WorkerConnector",
    )

    bundle = ExtensionBundleDescriptor(
        bundle_id="org.example.kv-system-adapter",
        bundle_version="1.0.0",
        host_api_range=">=1,<2",
        components=(scheduler, worker),
    )

    assert bundle.components == (scheduler, worker)


def test_control_plane_is_represented_by_a_local_bridge() -> None:
    bridge = ExtensionComponentDescriptor(
        component_id="control-plane-bridge",
        contracts=(
            DomainContract.CONTROL_ACTION_V1,
            DomainContract.CONTROL_RECEIPT_V1,
        ),
        execution_planes=(ExecutionPlane.BRIDGE,),
        isolation=ComponentIsolation.PROCESS_ISOLATED,
        implementation_ref="example.control:Bridge",
        permissions=(ComponentPermission.NETWORK_EGRESS,),
    )

    assert DomainContract.CONTROL_ACTION_V1 in bridge.contracts
    assert DomainContract.CONTROL_RECEIPT_V1 in bridge.contracts


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("component_id", "Invalid ID", "component_id"),
        ("contracts", (), "domain contract"),
        ("execution_planes", (), "execution plane"),
        ("implementation_ref", " ", "implementation_ref"),
    ],
)
def test_component_descriptor_rejects_ambiguous_identity_or_placement(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments = {
        "component_id": "valid-component",
        "contracts": (DomainContract.SCHEDULER_POLICY_V1,),
        "execution_planes": (ExecutionPlane.SCHEDULER,),
        "isolation": ComponentIsolation.TRUSTED_IN_PROCESS,
        "implementation_ref": "example.policy:Policy",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        ExtensionComponentDescriptor(**arguments)  # type: ignore[arg-type]


def test_bundle_rejects_duplicate_component_ids() -> None:
    component = ExtensionComponentDescriptor(
        component_id="policy",
        contracts=(DomainContract.SCHEDULER_POLICY_V1,),
        execution_planes=(ExecutionPlane.SCHEDULER,),
        isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
        implementation_ref="example.policy:Policy",
    )

    with pytest.raises(ValueError, match="component ids must be unique"):
        ExtensionBundleDescriptor(
            bundle_id="org.example.policy",
            bundle_version="1.0.0",
            host_api_range=">=1,<2",
            components=(component, component),
        )
