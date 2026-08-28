# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import sys
from pathlib import Path

import pytest

from vllm.plugins.contracts import ComponentPermission
from vllm.plugins.manifest import load_extension_bundle_manifest
from vllm.plugins.startup import (
    ExtensionBundleAdmissionError,
    resolve_extension_startup,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = ROOT / "vllm" / "plugins" / "builtin_kv_bundles"
MOONCAKE = BUNDLE_DIR / "mooncake.bundle.json"
LMCACHE = BUNDLE_DIR / "lmcache.bundle.json"


@pytest.mark.parametrize("manifest", [MOONCAKE, LMCACHE])
def test_builtin_kv_bundle_is_closed_and_requires_permissions(
    manifest: Path,
) -> None:
    bundle = load_extension_bundle_manifest(manifest)
    assert len(bundle.components) == 6
    with pytest.raises(ExtensionBundleAdmissionError):
        resolve_extension_startup((manifest,), allowed_permissions=())
    resolution = resolve_extension_startup(
        (manifest,), allowed_permissions=tuple(ComponentPermission)
    )
    assert len(resolution.snapshot.components) == 6


def test_mooncake_telemetry_does_not_import_connector_implementations() -> None:
    connector_modules = {
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector",
    }
    for module_name in connector_modules:
        sys.modules.pop(module_name, None)

    telemetry = importlib.import_module(
        "vllm.plugins.builtin_kv_bundles.mooncake_telemetry"
    )

    assert connector_modules.isdisjoint(sys.modules)
    assert telemetry.MooncakeDirectTelemetryProvider.build_kv_connector_stats()
    assert telemetry.MooncakeStoreTelemetryProvider.build_kv_connector_stats()


def test_lmcache_telemetry_does_not_import_lmcache_or_worker_modules() -> None:
    connector_modules = {
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
    }
    for module_name in connector_modules:
        sys.modules.pop(module_name, None)
    telemetry = importlib.import_module(
        "vllm.plugins.builtin_kv_bundles.lmcache_telemetry"
    )

    assert connector_modules.isdisjoint(sys.modules)
    assert telemetry.LMCacheTelemetryProvider.build_kv_connector_stats() is None


def test_builtin_bundle_json_is_wheel_package_data() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"plugins/builtin_kv_bundles/*.json"' in setup_source
