# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory,
    KVConnectorMaterializationError,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.plugins.contracts import ComponentPermission
from vllm.plugins.startup import resolve_extension_startup

ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ROOT / "vllm" / "plugins" / "builtin_kv_bundles"


def configured_bundles():
    return resolve_extension_startup(
        (BUNDLES / "mooncake.bundle.json", BUNDLES / "lmcache.bundle.json"),
        allowed_permissions=tuple(ComponentPermission),
    )


@pytest.fixture(autouse=True)
def startup_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        configured_bundles,
    )


def typed_config(
    *,
    connector_id: str,
    bundle_id: str,
    component_prefix: str,
    supports_hma: bool,
    required_layout: str | None = None,
) -> KVTransferConfig:
    return KVTransferConfig(
        kv_role="kv_both",
        kv_connector_selection={
            "schema_version": "1.0",
            "composition": "single",
            "connectors": [
                {
                    "connector_id": connector_id,
                    "scheduler_component": (
                        f"{bundle_id}/{component_prefix}-scheduler"
                    ),
                    "worker_component": f"{bundle_id}/{component_prefix}-worker",
                    "telemetry_component": (
                        f"{bundle_id}/{component_prefix}-telemetry"
                    ),
                    "scheduler_capabilities": {"supports_hma": supports_hma},
                    "worker_capabilities": {
                        "supports_hma": supports_hma,
                        "requires_piecewise_for_cudagraph": False,
                        "required_kv_cache_layout": required_layout,
                    },
                }
            ],
        },
    )


CASES = (
    (
        "MooncakeConnector",
        typed_config(
            connector_id="mooncake-direct",
            bundle_id="vllm-core.mooncake-bridges",
            component_prefix="direct",
            supports_hma=True,
            required_layout="HND",
        ),
    ),
    (
        "MooncakeStoreConnector",
        typed_config(
            connector_id="mooncake-store",
            bundle_id="vllm-core.mooncake-bridges",
            component_prefix="store",
            supports_hma=True,
        ),
    ),
    (
        "LMCacheConnectorV1",
        typed_config(
            connector_id="lmcache-v1",
            bundle_id="vllm-core.lmcache-bridges",
            component_prefix="v1",
            supports_hma=False,
        ),
    ),
    (
        "LMCacheMPConnector",
        typed_config(
            connector_id="lmcache-multiprocess",
            bundle_id="vllm-core.lmcache-bridges",
            component_prefix="multiprocess",
            supports_hma=False,
        ),
    ),
)


@pytest.mark.parametrize(("legacy_name", "typed"), CASES)
def test_typed_roles_resolve_to_the_legacy_implementation_or_same_missing_dependency(
    legacy_name: str,
    typed: KVTransferConfig,
) -> None:
    pair = KVConnectorFactory.resolve_typed_selection(typed).connectors[0]
    try:
        legacy_class = KVConnectorFactory.get_connector_class_by_name(legacy_name)
    except ModuleNotFoundError as legacy_error:
        with pytest.raises(KVConnectorMaterializationError) as typed_error:
            KVConnectorFactory.get_typed_connector_class(
                pair, KVConnectorRole.SCHEDULER
            )
        assert isinstance(typed_error.value.__cause__, ModuleNotFoundError)
        assert typed_error.value.__cause__.name == legacy_error.name
        return

    assert (
        KVConnectorFactory.get_typed_connector_class(pair, KVConnectorRole.SCHEDULER)
        is legacy_class
    )
    assert (
        KVConnectorFactory.get_typed_connector_class(pair, KVConnectorRole.WORKER)
        is legacy_class
    )
    rollback = KVTransferConfig(kv_connector=legacy_name, kv_role="kv_both")
    assert KVConnectorFactory.get_connector_class(rollback) is legacy_class


@pytest.mark.parametrize(("legacy_name", "typed"), CASES[:3])
def test_typed_telemetry_matches_the_legacy_stats_codec(
    legacy_name: str,
    typed: KVTransferConfig,
) -> None:
    legacy_class = KVConnectorFactory.get_connector_class_by_name(legacy_name)
    telemetry_class = KVConnectorFactory.get_telemetry_class(typed)
    data: dict[str, list[object]] = {"test": []}

    legacy_stats = legacy_class.build_kv_connector_stats(data=data)
    typed_stats = telemetry_class.build_kv_connector_stats(data=data)
    if legacy_stats is None:
        assert typed_stats is None
    else:
        assert type(typed_stats) is type(legacy_stats)
        assert typed_stats.data == legacy_stats.data
