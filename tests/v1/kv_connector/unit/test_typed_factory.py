# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector import factory
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
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
        self.extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config


class WorkerConnector(ExampleConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        self.constructed_role = role
        self.extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config


@dataclass
class ExampleStats(KVConnectorStats):
    def reset(self):
        self.data.clear()

    def aggregate(self, other):
        self.data.update(other.data)
        return self

    def reduce(self):
        return self.data

    def is_empty(self):
        return not self.data


class TelemetryCodec:
    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return ExampleStats(data=data or {})

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
            ExtensionComponentDescriptor(
                component_id="scheduler-two",
                contracts=(DomainContract.KV_CONNECTOR_SCHEDULER_V1,),
                execution_planes=(ExecutionPlane.SCHEDULER,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="scheduler_module:SchedulerConnector",
            ),
            ExtensionComponentDescriptor(
                component_id="worker-two",
                contracts=(DomainContract.KV_CONNECTOR_WORKER_V1,),
                execution_planes=(ExecutionPlane.WORKER,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref="worker_module:WorkerConnector",
            ),
            ExtensionComponentDescriptor(
                component_id="telemetry-two",
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


def typed_ordered_config(*, include_child_configs: bool = True) -> KVTransferConfig:
    extra_config = (
        {
            "typed_connectors": {
                "first": {"secret_marker": "first-only"},
                "second": {"secret_marker": "second-only"},
            }
        }
        if include_child_configs
        else {}
    )
    return KVTransferConfig(
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
        kv_connector_selection={
            "schema_version": "1.0",
            "composition": "ordered_multi",
            "connectors": [
                {
                    "connector_id": "first",
                    "scheduler_component": "org.example.kv/scheduler",
                    "worker_component": "org.example.kv/worker",
                    "telemetry_component": "org.example.kv/telemetry",
                    "scheduler_capabilities": {"supports_hma": False},
                    "worker_capabilities": {
                        "supports_hma": False,
                        "requires_piecewise_for_cudagraph": False,
                        "required_kv_cache_layout": None,
                    },
                },
                {
                    "connector_id": "second",
                    "scheduler_component": "org.example.kv/scheduler-two",
                    "worker_component": "org.example.kv/worker-two",
                    "telemetry_component": "org.example.kv/telemetry-two",
                    "scheduler_capabilities": {"supports_hma": False},
                    "worker_capabilities": {
                        "supports_hma": False,
                        "requires_piecewise_for_cudagraph": False,
                        "required_kv_cache_layout": None,
                    },
                },
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


def test_ordered_multi_requires_exact_namespaced_child_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factory.importlib,
        "import_module",
        lambda name: pytest.fail(f"invalid config imported {name}"),
    )

    with pytest.raises(
        factory.KVConnectorMaterializationError, match="typed_connectors"
    ):
        KVConnectorFactory.supports_hma_config(
            typed_ordered_config(include_child_configs=False)
        )


def test_typed_runtime_config_schema_is_closed_at_the_host_boundary() -> None:
    schema_path = (
        Path(factory.__file__).parents[3]
        / "plugins"
        / "kv_connector_runtime_config.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    validator.validate({"typed_connectors": {"first": {"provider_owned": "value"}}})
    errors = list(
        validator.iter_errors(
            {"typed_connectors": {"first": {}}, "ambiguous_global": True}
        )
    )
    assert errors


def test_legacy_module_path_remains_available_without_typed_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []

    def import_module(name: str):
        imported.append(name)
        return SimpleNamespace(SchedulerConnector=SchedulerConnector)

    monkeypatch.setattr(factory.importlib, "import_module", import_module)
    legacy_config = KVTransferConfig(
        kv_connector="SchedulerConnector",
        kv_connector_module_path="external_connector_module",
        kv_role="kv_both",
    )

    assert KVConnectorFactory.get_connector_class(legacy_config) is SchedulerConnector
    assert imported == ["external_connector_module"]


def test_typed_import_failure_does_not_fall_back_and_legacy_rollback_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []

    def fail_typed_import(name: str):
        imported.append(name)
        raise ImportError(name)

    monkeypatch.setattr(factory.importlib, "import_module", fail_typed_import)
    typed = typed_config()
    typed_vllm_config = SimpleNamespace(
        kv_transfer_config=typed,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )

    with pytest.raises(
        factory.KVConnectorMaterializationError, match="failed to import"
    ):
        KVConnectorFactory.create_connector(
            typed_vllm_config,
            KVConnectorRole.SCHEDULER,
            SimpleNamespace(),
        )
    assert imported == ["scheduler_module"]

    imported.clear()

    def import_legacy(name: str):
        imported.append(name)
        return SimpleNamespace(SchedulerConnector=SchedulerConnector)

    monkeypatch.setattr(factory.importlib, "import_module", import_legacy)
    legacy_config = KVTransferConfig(
        kv_connector="SchedulerConnector",
        kv_connector_module_path="external_connector_module",
        kv_role="kv_both",
    )

    assert KVConnectorFactory.get_connector_class(legacy_config) is SchedulerConnector
    assert imported == ["external_connector_module"]


@pytest.mark.parametrize(
    ("role", "expected_type", "expected_module"),
    [
        (KVConnectorRole.SCHEDULER, SchedulerConnector, "scheduler_module"),
        (KVConnectorRole.WORKER, WorkerConnector, "worker_module"),
    ],
)
def test_ordered_multi_constructs_children_in_order_with_isolated_config(
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
    config = typed_ordered_config()
    vllm_config = SimpleNamespace(
        kv_transfer_config=config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )

    connector = KVConnectorFactory.create_connector(
        vllm_config,
        role,
        SimpleNamespace(),
    )

    assert connector._connector_ids == ["first", "second"]
    assert [child.extra_config for child in connector._connectors] == [
        {"secret_marker": "first-only"},
        {"secret_marker": "second-only"},
    ]
    assert all(isinstance(child, expected_type) for child in connector._connectors)
    assert imported == [expected_module, expected_module]


def test_ordered_multi_telemetry_imports_only_api_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []

    def import_module(name: str):
        imported.append(name)
        assert name == "telemetry_module"
        return SimpleNamespace(TelemetryCodec=TelemetryCodec)

    monkeypatch.setattr(factory.importlib, "import_module", import_module)
    provider = KVConnectorFactory.get_telemetry_class(typed_ordered_config())

    stats = provider.build_kv_connector_stats(None)
    assert stats.data == {}
    assert imported == ["telemetry_module", "telemetry_module"]
    stats = provider.build_kv_connector_stats({"first": {"data": {"logical_id": 1}}})
    assert list(stats.data) == ["first"]
    assert stats.data["first"].data == {"logical_id": 1}
    with pytest.raises(
        factory.KVConnectorMaterializationError, match="unknown connector IDs"
    ):
        provider.build_kv_connector_stats({"unknown": {"data": {}}})
