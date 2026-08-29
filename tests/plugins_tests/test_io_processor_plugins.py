# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.inputs import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.contracts import (
    ComponentIsolation,
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)
from vllm.plugins.io_processors import get_io_processor
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.plugins.materialization import ExtensionComponentMaterializationError
from vllm.plugins.snapshot import ExtensionStartupSnapshot
from vllm.plugins.startup import ExtensionStartupResolution
from vllm.renderers import BaseRenderer


class DummyIOProcessor(IOProcessor):
    """Minimal IOProcessor used as the target of the mocked plugin entry point."""

    def pre_process(
        self,
        prompt: object,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> object:
        raise NotImplementedError


def typed_io_resolution(
    implementation_ref: str,
) -> ExtensionStartupResolution:
    bundle = ExtensionBundleDescriptor(
        bundle_id="test-io",
        bundle_version="1.0.0",
        host_api_range=">=1.0,<2.0",
        components=(
            ExtensionComponentDescriptor(
                component_id="processor",
                contracts=(DomainContract.IO_PROCESSOR_V1,),
                execution_planes=(ExecutionPlane.API,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref=implementation_ref,
            ),
        ),
    )
    return ExtensionStartupResolution(
        snapshot=ExtensionStartupSnapshot.build((bundle,)),
        disabled_bundle_ids=(),
    )


@pytest.fixture
def my_plugin_entry_points():
    """Patch importlib.metadata.entry_points to expose a single 'my_plugin'
    entry point backed by DummyIOProcessor, exercising the full plugin-loading
    code path: entry_points → plugin.load() → func() →
    resolve_obj_by_qualname → IOProcessor.__init__."""
    qualname = f"{DummyIOProcessor.__module__}.{DummyIOProcessor.__qualname__}"
    ep = MagicMock()
    ep.name = "my_plugin"
    ep.value = qualname
    ep.load.return_value = lambda: qualname
    with patch("importlib.metadata.entry_points", return_value=[ep]):
        yield


def test_loading_missing_plugin():
    vllm_config = MagicMock(spec=VllmConfig)
    renderer = MagicMock(spec=BaseRenderer)
    with pytest.raises(ValueError):
        get_io_processor(
            vllm_config, renderer=renderer, plugin_from_init="wrong_plugin"
        )


def test_loading_plugin(my_plugin_entry_points):
    # Plugin name supplied via plugin_from_init.
    vllm_config = MagicMock(spec=VllmConfig)
    renderer = MagicMock(spec=BaseRenderer)

    result = get_io_processor(
        vllm_config, renderer=renderer, plugin_from_init="my_plugin"
    )

    assert isinstance(result, DummyIOProcessor)


def test_loading_missing_plugin_from_model_config():
    # Build a mock VllmConfig whose hf_config advertises a plugin name,
    # exercising the model-config code path without loading a real model.
    mock_hf_config = MagicMock()
    mock_hf_config.to_dict.return_value = {"io_processor_plugin": "wrong_plugin"}

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config.hf_config = mock_hf_config

    renderer = MagicMock(spec=BaseRenderer)
    with pytest.raises(ValueError):
        get_io_processor(vllm_config, renderer=renderer)


def test_loading_plugin_from_model_config(my_plugin_entry_points):
    # Plugin name supplied via the model's hf_config.
    mock_hf_config = MagicMock()
    mock_hf_config.to_dict.return_value = {"io_processor_plugin": "my_plugin"}

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config.hf_config = mock_hf_config

    renderer = MagicMock(spec=BaseRenderer)

    result = get_io_processor(vllm_config, renderer=renderer)

    assert isinstance(result, DummyIOProcessor)


def test_loading_explicit_typed_io_processor(monkeypatch: pytest.MonkeyPatch):
    resolution = typed_io_resolution(
        f"{DummyIOProcessor.__module__}:{DummyIOProcessor.__qualname__}"
    )
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: resolution,
    )
    legacy_loader = MagicMock()
    monkeypatch.setattr(
        "vllm.plugins.io_processors.load_plugins_by_group",
        legacy_loader,
    )

    result = get_io_processor(
        MagicMock(spec=VllmConfig),
        renderer=MagicMock(spec=BaseRenderer),
        plugin_from_init="test-io/processor",
    )

    assert isinstance(result, DummyIOProcessor)
    legacy_loader.assert_not_called()


def test_unknown_typed_io_processor_fails_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    resolution = typed_io_resolution(
        f"{DummyIOProcessor.__module__}:{DummyIOProcessor.__qualname__}"
    )
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: resolution,
    )
    legacy_loader = MagicMock()
    monkeypatch.setattr(
        "vllm.plugins.io_processors.load_plugins_by_group",
        legacy_loader,
    )

    with pytest.raises(
        ExtensionComponentMaterializationError,
        match="selects no admitted API-plane IO processor",
    ):
        get_io_processor(
            MagicMock(spec=VllmConfig),
            renderer=MagicMock(spec=BaseRenderer),
            plugin_from_init="unknown/processor",
        )
    legacy_loader.assert_not_called()


def test_typed_io_processor_protocol_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    resolution = typed_io_resolution("builtins:object")
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: resolution,
    )

    with pytest.raises(
        ExtensionComponentMaterializationError,
        match="must resolve to an IOProcessor subclass",
    ):
        get_io_processor(
            MagicMock(spec=VllmConfig),
            renderer=MagicMock(spec=BaseRenderer),
            plugin_from_init="test-io/processor",
        )
