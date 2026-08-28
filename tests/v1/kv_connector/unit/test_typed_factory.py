# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector import factory
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.plugins.contracts import (
    ComponentIsolation,
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)
from vllm.plugins.snapshot import ExtensionStartupSnapshot
from vllm.plugins.startup import ExtensionStartupResolution


class SchedulerConnector(ExampleConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        self.constructed_role = role


class WorkerConnector(ExampleConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        self.constructed_role = role


class TelemetryCodec:
    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return None

    @classmethod
    def build_prom_metrics(cls, *args, **kwargs):
        return None


def make_resolution() -> ExtensionStartupResolution:
    bundle = ExtensionBundleDescriptor(
        bundle_id="org.example.kv",
        bundle_version="1.0.0",
        host_api_range=">=1,<2",
        components=(
            ExtensionComponentDescriptor(
                component_id="scheduler",
                contracts=(DomainContract.KV_CONNECTOR_SCHEDULER_V1,),
                execution_planes=(ExecutionPlane.SCHEDULER,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="scheduler_module:SchedulerConnector",
            ),
            ExtensionComponentDescriptor(
                component_id="worker",
                contracts=(DomainContract.KV_CONNECTOR_WORKER_V1,),
                execution_planes=(ExecutionPlane.WORKER,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="worker_module:WorkerConnector",
            ),
            ExtensionComponentDescriptor(
                component_id="telemetry",
                contracts=(DomainContract.KV_CONNECTOR_TELEMETRY_V1,),
                execution_planes=(ExecutionPlane.API,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="telemetry_module:TelemetryCodec",
            ),
        ),
    )
    return ExtensionStartupResolution(
        snapshot=ExtensionStartupSnapshot.build((bundle,)),
        disabled_bundle_ids=(),
    )


def typed_config() -> KVTransferConfig:
    return KVTransferConfig(
        kv_role="kv_both",
        kv_connector_selection={
            "schema_version": "1.0",
            "composition": "single",
            "connectors": [
                {
                    "connector_id": "primary",
                    "scheduler_component": "org.example.kv/scheduler",
                    "worker_component": "org.example.kv/worker",
                    "telemetry_component": "org.example.kv/telemetry",
                    "scheduler_capabilities": {"supports_hma": False},
                    "worker_capabilities": {
                        "supports_hma": False,
                        "requires_piecewise_for_cudagraph": False,
                        "required_kv_cache_layout": None,
                    },
                }
            ],
        },
    )


@pytest.fixture(autouse=True)
def configured_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        make_resolution,
    )


def test_configuration_checks_do_not_import_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factory.importlib,
        "import_module",
        lambda name: pytest.fail(f"configuration imported {name}"),
    )
    config = typed_config()

    assert KVConnectorFactory.supports_hma_config(config) is False
    assert KVConnectorFactory.requires_piecewise_config(config) is False
    assert (
        KVConnectorFactory.required_kv_cache_layout(
            SimpleNamespace(kv_transfer_config=config)
        )
        is None
    )


def test_each_process_imports_only_its_selected_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []
    modules = {
        "scheduler_module": SimpleNamespace(SchedulerConnector=SchedulerConnector),
        "worker_module": SimpleNamespace(WorkerConnector=WorkerConnector),
        "telemetry_module": SimpleNamespace(TelemetryCodec=TelemetryCodec),
    }

    def import_module(name: str):
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(factory.importlib, "import_module", import_module)
    config = typed_config()
    pair = KVConnectorFactory.resolve_typed_selection(config).connectors[0]

    assert (
        KVConnectorFactory.get_typed_connector_class(pair, KVConnectorRole.SCHEDULER)
        is SchedulerConnector
    )
    assert imported == ["scheduler_module"]

    imported.clear()
    assert (
        KVConnectorFactory.get_typed_connector_class(pair, KVConnectorRole.WORKER)
        is WorkerConnector
    )
    assert imported == ["worker_module"]

    imported.clear()
    assert KVConnectorFactory.get_telemetry_class(config) is TelemetryCodec
    assert imported == ["telemetry_module"]


@pytest.mark.parametrize(
    ("role", "expected_type", "expected_module"),
    [
        (KVConnectorRole.SCHEDULER, SchedulerConnector, "scheduler_module"),
        (KVConnectorRole.WORKER, WorkerConnector, "worker_module"),
    ],
)
def test_factory_constructs_the_role_specific_typed_connector(
    monkeypatch: pytest.MonkeyPatch,
    role: KVConnectorRole,
    expected_type: type[ExampleConnector],
    expected_module: str,
) -> None:
    imported = []
    modules = {
        "scheduler_module": SimpleNamespace(SchedulerConnector=SchedulerConnector),
        "worker_module": SimpleNamespace(WorkerConnector=WorkerConnector),
    }

    def import_module(name: str):
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(factory.importlib, "import_module", import_module)
    config = typed_config()
    vllm_config = SimpleNamespace(
        kv_transfer_config=config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )

    connector = KVConnectorFactory.create_connector(
        vllm_config,
        role,
        SimpleNamespace(),
    )

    assert isinstance(connector, expected_type)
    assert connector.constructed_role is role
    assert imported == [expected_module]


def test_declaration_mismatch_fails_closed() -> None:
    config = typed_config()
    pair = KVConnectorFactory.resolve_typed_selection(config).connectors[0]
    pair = type(pair)(
        connector_id=pair.connector_id,
        scheduler_component=pair.scheduler_component,
        worker_component=pair.worker_component,
        telemetry_component=pair.telemetry_component,
        scheduler_capabilities=type(pair.scheduler_capabilities)(supports_hma=True),
        worker_capabilities=pair.worker_capabilities,
    )

    with pytest.raises(factory.KVConnectorMaterializationError, match="mismatch"):
        KVConnectorFactory.verify_typed_connector_declarations(
            SchedulerConnector,
            pair,
            KVConnectorRole.SCHEDULER,
            SimpleNamespace(kv_transfer_config=config),
        )
