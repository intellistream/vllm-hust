# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.plugins.contracts import (
    ComponentIsolation,
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)
from vllm.plugins.kv_connector_selection import (
    KVConnectorComposition,
    KVConnectorSelectionError,
    parse_kv_connector_selection,
    resolve_kv_connector_selection,
)
from vllm.plugins.snapshot import ExtensionStartupSnapshot


def make_snapshot() -> ExtensionStartupSnapshot:
    split = ExtensionBundleDescriptor(
        bundle_id="org.example.split",
        bundle_version="1.0.0",
        host_api_range=">=1,<2",
        components=(
            ExtensionComponentDescriptor(
                component_id="scheduler",
                contracts=(DomainContract.KV_CONNECTOR_SCHEDULER_V1,),
                execution_planes=(ExecutionPlane.SCHEDULER,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="must_not_import:SchedulerConnector",
            ),
            ExtensionComponentDescriptor(
                component_id="worker",
                contracts=(DomainContract.KV_CONNECTOR_WORKER_V1,),
                execution_planes=(ExecutionPlane.WORKER, ExecutionPlane.DEVICE),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="must_not_import:WorkerConnector",
            ),
        ),
    )
    combined = ExtensionBundleDescriptor(
        bundle_id="org.example.combined",
        bundle_version="1.0.0",
        host_api_range=">=1,<2",
        components=(
            ExtensionComponentDescriptor(
                component_id="connector",
                contracts=(
                    DomainContract.KV_CONNECTOR_SCHEDULER_V1,
                    DomainContract.KV_CONNECTOR_WORKER_V1,
                ),
                execution_planes=(ExecutionPlane.SCHEDULER, ExecutionPlane.WORKER),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="must_not_import:CombinedConnector",
            ),
        ),
    )
    return ExtensionStartupSnapshot.build((split, combined))


def test_single_selection_resolves_split_role_components() -> None:
    profile = parse_kv_connector_selection(
        {
            "schema_version": "1.0",
            "composition": "single",
            "connectors": [
                {
                    "connector_id": "primary",
                    "scheduler_component": "org.example.split/scheduler",
                    "worker_component": "org.example.split/worker",
                }
            ],
        }
    )

    resolved = resolve_kv_connector_selection(profile, make_snapshot())

    assert resolved.composition is KVConnectorComposition.SINGLE
    assert resolved.connectors[0].scheduler_component.qualified_id == (
        "org.example.split/scheduler"
    )
    assert resolved.connectors[0].worker_component.qualified_id == (
        "org.example.split/worker"
    )


def test_ordered_multi_preserves_explicit_order() -> None:
    profile = parse_kv_connector_selection(
        {
            "schema_version": "1.0",
            "composition": "ordered_multi",
            "connectors": [
                {
                    "connector_id": "first",
                    "scheduler_component": "org.example.combined/connector",
                    "worker_component": "org.example.combined/connector",
                },
                {
                    "connector_id": "second",
                    "scheduler_component": "org.example.split/scheduler",
                    "worker_component": "org.example.split/worker",
                },
            ],
        }
    )

    resolved = resolve_kv_connector_selection(profile, make_snapshot())

    assert [connector.connector_id for connector in resolved.connectors] == [
        "first",
        "second",
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "schema_version": "1.0",
                "composition": "single",
                "connectors": [],
            },
            "at least one",
        ),
        (
            {
                "schema_version": "1.0",
                "composition": "ordered_multi",
                "connectors": [
                    {
                        "connector_id": "only",
                        "scheduler_component": "org.example.split/scheduler",
                        "worker_component": "org.example.split/worker",
                    }
                ],
            },
            "at least two",
        ),
        (
            {
                "schema_version": "1.0",
                "composition": "single",
                "connectors": [
                    {
                        "connector_id": "primary",
                        "scheduler_component": "org.example.split/scheduler",
                        "worker_component": "org.example.split/worker",
                        "configuration": {},
                    }
                ],
            },
            "unknown fields",
        ),
    ],
)
def test_parser_rejects_ambiguous_or_free_form_topology(payload, message) -> None:
    with pytest.raises(KVConnectorSelectionError, match=message):
        parse_kv_connector_selection(payload)


def test_resolution_rejects_scheduler_worker_crossing() -> None:
    profile = parse_kv_connector_selection(
        {
            "schema_version": "1.0",
            "composition": "single",
            "connectors": [
                {
                    "connector_id": "crossed",
                    "scheduler_component": "org.example.split/worker",
                    "worker_component": "org.example.split/scheduler",
                }
            ],
        }
    )

    with pytest.raises(KVConnectorSelectionError, match="does not implement"):
        resolve_kv_connector_selection(profile, make_snapshot())


def test_resolution_error_names_available_provider() -> None:
    profile = parse_kv_connector_selection(
        {
            "schema_version": "1.0",
            "composition": "single",
            "connectors": [
                {
                    "connector_id": "missing",
                    "scheduler_component": "org.example.missing/scheduler",
                    "worker_component": "org.example.split/worker",
                }
            ],
        }
    )

    with pytest.raises(
        KVConnectorSelectionError, match="org.example.split/scheduler"
    ):
        resolve_kv_connector_selection(profile, make_snapshot())
