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


def capabilities(
    *,
    supports_hma: bool = False,
    requires_piecewise: bool = False,
    required_layout: str | None = None,
) -> dict:
    return {
        "scheduler_capabilities": {"supports_hma": supports_hma},
        "worker_capabilities": {
            "supports_hma": supports_hma,
            "requires_piecewise_for_cudagraph": requires_piecewise,
            "required_kv_cache_layout": required_layout,
        },
    }


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
            ExtensionComponentDescriptor(
                component_id="telemetry",
                contracts=(DomainContract.KV_CONNECTOR_TELEMETRY_V1,),
                execution_planes=(ExecutionPlane.API,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="must_not_import:TelemetryCodec",
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
                    DomainContract.KV_CONNECTOR_TELEMETRY_V1,
                ),
                execution_planes=(
                    ExecutionPlane.API,
                    ExecutionPlane.SCHEDULER,
                    ExecutionPlane.WORKER,
                ),
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
                    "telemetry_component": "org.example.split/telemetry",
                    **capabilities(),
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
    assert resolved.connectors[0].telemetry_component.qualified_id == (
        "org.example.split/telemetry"
    )
    assert resolved.supports_hma is False
    assert resolved.requires_piecewise_for_cudagraph is False
    assert resolved.required_kv_cache_layout is None


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
                    "telemetry_component": "org.example.combined/connector",
                    **capabilities(
                        supports_hma=True,
                        requires_piecewise=True,
                        required_layout="NHD",
                    ),
                },
                {
                    "connector_id": "second",
                    "scheduler_component": "org.example.split/scheduler",
                    "worker_component": "org.example.split/worker",
                    "telemetry_component": "org.example.split/telemetry",
                    **capabilities(supports_hma=True, required_layout="NHD"),
                },
            ],
        }
    )

    resolved = resolve_kv_connector_selection(profile, make_snapshot())

    assert [connector.connector_id for connector in resolved.connectors] == [
        "first",
        "second",
    ]
    assert resolved.supports_hma is True
    assert resolved.requires_piecewise_for_cudagraph is True
    assert resolved.required_kv_cache_layout == "NHD"


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
                        "telemetry_component": "org.example.split/telemetry",
                        **capabilities(),
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
                        "telemetry_component": "org.example.split/telemetry",
                    }
                ],
            },
            "scheduler_capabilities must be an object",
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
                        "telemetry_component": "org.example.split/telemetry",
                        **capabilities(),
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
                    "telemetry_component": "org.example.split/telemetry",
                    **capabilities(),
                }
            ],
        }
    )

    with pytest.raises(KVConnectorSelectionError, match="does not implement"):
        resolve_kv_connector_selection(profile, make_snapshot())


def test_resolution_rejects_worker_as_telemetry_component() -> None:
    profile = parse_kv_connector_selection(
        {
            "schema_version": "1.0",
            "composition": "single",
            "connectors": [
                {
                    "connector_id": "wrong-telemetry-role",
                    "scheduler_component": "org.example.split/scheduler",
                    "worker_component": "org.example.split/worker",
                    "telemetry_component": "org.example.split/worker",
                    **capabilities(),
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
                    "telemetry_component": "org.example.split/telemetry",
                    **capabilities(),
                }
            ],
        }
    )

    with pytest.raises(KVConnectorSelectionError, match="org.example.split/scheduler"):
        resolve_kv_connector_selection(profile, make_snapshot())


def test_ordered_multi_rejects_conflicting_cache_layouts() -> None:
    payload = {
        "schema_version": "1.0",
        "composition": "ordered_multi",
        "connectors": [
            {
                "connector_id": "first",
                "scheduler_component": "org.example.combined/connector",
                "worker_component": "org.example.combined/connector",
                "telemetry_component": "org.example.combined/connector",
                **capabilities(required_layout="NHD"),
            },
            {
                "connector_id": "second",
                "scheduler_component": "org.example.split/scheduler",
                "worker_component": "org.example.split/worker",
                "telemetry_component": "org.example.split/telemetry",
                **capabilities(required_layout="HND"),
            },
        ],
    }

    with pytest.raises(KVConnectorSelectionError, match="conflicting"):
        parse_kv_connector_selection(payload)
