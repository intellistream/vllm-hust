# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.plugins.contracts import (
    ComponentIsolation,
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)
from vllm.plugins.snapshot import ExtensionStartupSnapshot
from vllm.plugins.startup import ExtensionStartupResolution
from vllm.v1.core.sched import victim_selector
from vllm.v1.core.sched.victim_selector import (
    VictimSelectorMaterializationError,
    get_victim_selector,
)


def make_resolution(*component_ids: str) -> ExtensionStartupResolution:
    bundle = ExtensionBundleDescriptor(
        bundle_id="org.example.policies",
        bundle_version="1.0.0",
        host_api_range=">=1,<2",
        components=tuple(
            ExtensionComponentDescriptor(
                component_id=component_id,
                contracts=(DomainContract.SCHEDULER_POLICY_V1,),
                execution_planes=(ExecutionPlane.SCHEDULER,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref=f"example_policy:{component_id}",
            )
            for component_id in component_ids
        ),
    )
    return ExtensionStartupResolution(
        snapshot=ExtensionStartupSnapshot.build((bundle,)),
        disabled_bundle_ids=(),
    )


class ExampleSelector:
    @classmethod
    def from_vllm_config(cls, vllm_config):
        return cls()

    def pick_victim(self, running, policy, *, kv_utilization=None, now_s=None):
        return running[-1]

    def emit_observability_log(self, logger, scheduler_name):
        pass

    def export_metrics(self):
        return {}


def test_typed_provider_materializes_before_legacy_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: make_resolution("primary"),
    )
    monkeypatch.setattr(
        victim_selector,
        "import_module",
        lambda name: SimpleNamespace(primary=ExampleSelector),
    )
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda **kwargs: pytest.fail("legacy discovery must not run"),
    )

    selector = get_victim_selector(SimpleNamespace(additional_config={}))

    assert isinstance(selector, ExampleSelector)


def test_multiple_typed_providers_require_qualified_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: make_resolution("first", "second"),
    )

    with pytest.raises(VictimSelectorMaterializationError, match="multiple admitted"):
        get_victim_selector(SimpleNamespace(additional_config={}))


def test_qualified_selection_resolves_typed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: make_resolution("first", "second"),
    )
    module = SimpleNamespace(first=ExampleSelector, second=ExampleSelector)
    monkeypatch.setattr(victim_selector, "import_module", lambda name: module)
    config = SimpleNamespace(
        additional_config={
            "victim_selector_component": "org.example.policies/second"
        }
    )

    assert isinstance(get_victim_selector(config), ExampleSelector)


def test_typed_import_failure_does_not_fall_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: make_resolution("primary"),
    )

    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(victim_selector, "import_module", fail_import)

    with pytest.raises(VictimSelectorMaterializationError, match="failed to import"):
        get_victim_selector(SimpleNamespace(additional_config={}))
