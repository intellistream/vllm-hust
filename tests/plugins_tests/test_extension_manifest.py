# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from importlib import resources
from pathlib import Path

import pytest

from vllm.plugins.contracts import (
    ComponentIsolation,
    ComponentPermission,
    DomainContract,
    ExecutionPlane,
)
from vllm.plugins.manifest import (
    load_extension_bundle_manifest,
    parse_extension_bundle_manifest,
)
from vllm.plugins.snapshot import ExtensionStartupSnapshot


def valid_manifest() -> dict:
    return {
        "schema_version": "1.0",
        "bundle_id": "org.example.kv-adapter",
        "bundle_version": "1.0.0",
        "host_api_range": ">=1,<2",
        "components": [
            {
                "component_id": "scheduler-connector",
                "contracts": ["vllm.kv_connector.scheduler.v1"],
                "execution_planes": ["scheduler"],
                "isolation": "trusted_in_process",
                "implementation_ref": "module_that_must_not_load:Scheduler",
                "permissions": [],
            },
            {
                "component_id": "worker-connector",
                "contracts": ["vllm.kv_connector.worker.v1"],
                "execution_planes": ["worker", "device"],
                "isolation": "trusted_in_process",
                "implementation_ref": "module_that_must_not_load:Worker",
                "permissions": [],
            },
        ],
    }


def test_manifest_is_validated_before_implementation_import(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(valid_manifest()), encoding="utf-8")

    bundle = load_extension_bundle_manifest(path)

    assert bundle.bundle_id == "org.example.kv-adapter"
    assert "module_that_must_not_load" not in sys.modules


def test_packaged_schema_matches_runtime_vocabulary() -> None:
    schema = json.loads(
        resources.files("vllm.plugins")
        .joinpath("manifest.schema.json")
        .read_text(encoding="utf-8")
    )
    component = schema["$defs"]["component"]["properties"]

    assert set(component["contracts"]["items"]["enum"]) == {
        contract.value for contract in DomainContract
    }
    assert set(component["execution_planes"]["items"]["enum"]) == {
        plane.value for plane in ExecutionPlane
    }
    assert set(component["isolation"]["enum"]) == {
        isolation.value for isolation in ComponentIsolation
    }
    assert set(component["permissions"]["items"]["enum"]) == {
        permission.value for permission in ComponentPermission
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update(extra=True), "unknown fields"),
        (
            lambda manifest: manifest.update(bundle_version="latest"),
            "semantic version",
        ),
        (
            lambda manifest: manifest["components"][0].update(extra=True),
            "unknown fields",
        ),
        (
            lambda manifest: manifest["components"][0].update(
                contracts=["vllm.unknown.v1"]
            ),
            "unsupported value",
        ),
        (
            lambda manifest: manifest["components"][0].update(
                isolation="subprocess"
            ),
            "isolation is unsupported",
        ),
        (
            lambda manifest: manifest["components"][0].update(
                permissions=["ambient_root"]
            ),
            "unsupported value",
        ),
    ],
)
def test_manifest_rejects_unknown_or_ambiguous_fields(
    mutation,
    message: str,
) -> None:
    manifest = valid_manifest()
    mutation(manifest)

    with pytest.raises(ValueError, match=message):
        parse_extension_bundle_manifest(manifest)


def test_startup_snapshot_preserves_scheduler_worker_separation() -> None:
    bundle = parse_extension_bundle_manifest(valid_manifest())
    snapshot = ExtensionStartupSnapshot.build((bundle,))

    scheduler = snapshot.components_for(
        DomainContract.KV_CONNECTOR_SCHEDULER_V1,
        ExecutionPlane.SCHEDULER,
    )
    worker = snapshot.components_for(
        DomainContract.KV_CONNECTOR_WORKER_V1,
        ExecutionPlane.WORKER,
    )

    assert [item.qualified_id for item in scheduler] == [
        "org.example.kv-adapter/scheduler-connector"
    ]
    assert [item.qualified_id for item in worker] == [
        "org.example.kv-adapter/worker-connector"
    ]


def test_startup_snapshot_rejects_duplicate_bundle_ids() -> None:
    bundle = parse_extension_bundle_manifest(valid_manifest())

    with pytest.raises(ValueError, match="bundle ids must be unique"):
        ExtensionStartupSnapshot.build((bundle, bundle))
