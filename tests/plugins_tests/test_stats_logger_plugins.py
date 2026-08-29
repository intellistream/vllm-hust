# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
from dummy_stat_logger.dummy_stat_logger import DummyStatLogger

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.plugins.contracts import (
    ComponentIsolation,
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)
from vllm.plugins.materialization import ExtensionComponentMaterializationError
from vllm.plugins.snapshot import ExtensionStartupSnapshot
from vllm.plugins.startup import ExtensionStartupResolution
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import load_stat_logger_plugin_factories


def typed_stat_logger_resolution(
    implementation_ref: str,
) -> ExtensionStartupResolution:
    bundle = ExtensionBundleDescriptor(
        bundle_id="test-observability",
        bundle_version="1.0.0",
        host_api_range=">=1.0,<2.0",
        components=(
            ExtensionComponentDescriptor(
                component_id="stats",
                contracts=(DomainContract.STAT_LOGGER_V1,),
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


def test_stat_logger_plugin_is_discovered(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "dummy_stat_logger")

        factories = load_stat_logger_plugin_factories()
        assert len(factories) == 1, f"Expected 1 factory, got {len(factories)}"
        assert factories[0] is DummyStatLogger, (
            f"Expected DummyStatLogger class, got {factories[0]}"
        )

        # instantiate and confirm the right type
        vllm_config = MagicMock(spec=VllmConfig)
        instance = factories[0](vllm_config)
        assert isinstance(instance, DummyStatLogger)


def test_no_plugins_loaded_if_env_empty(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "")

        factories = load_stat_logger_plugin_factories()
        assert factories == []


def test_invalid_stat_logger_plugin_raises(monkeypatch: pytest.MonkeyPatch):
    def fake_plugin_loader(group: str):
        assert group == "vllm.stat_logger_plugins"
        return {"bad": object()}

    with monkeypatch.context() as m:
        m.setattr(
            "vllm.v1.metrics.loggers.load_plugins_by_group",
            fake_plugin_loader,
        )
        with pytest.raises(
            TypeError,
            match="Stat logger plugin 'bad' must be a subclass of StatLoggerBase",
        ):
            load_stat_logger_plugin_factories()


def test_typed_stat_logger_composes_with_distinct_legacy_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    resolution = typed_stat_logger_resolution(
        "dummy_stat_logger.dummy_stat_logger:DummyStatLogger"
    )
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: resolution,
    )

    class OtherDummyStatLogger(DummyStatLogger):
        pass

    monkeypatch.setattr(
        "vllm.v1.metrics.loggers.load_plugins_by_group",
        lambda group: {"other": OtherDummyStatLogger},
    )

    assert load_stat_logger_plugin_factories() == [
        DummyStatLogger,
        OtherDummyStatLogger,
    ]


def test_typed_and_legacy_stat_logger_migration_is_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
):
    resolution = typed_stat_logger_resolution(
        "dummy_stat_logger.dummy_stat_logger:DummyStatLogger"
    )
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: resolution,
    )
    monkeypatch.setattr(
        "vllm.v1.metrics.loggers.load_plugins_by_group",
        lambda group: {"legacy-dummy": DummyStatLogger},
    )

    assert load_stat_logger_plugin_factories() == [DummyStatLogger]


def test_invalid_typed_stat_logger_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    resolution = typed_stat_logger_resolution("builtins:object")
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: resolution,
    )
    monkeypatch.setattr(
        "vllm.v1.metrics.loggers.load_plugins_by_group",
        lambda group: {},
    )

    with pytest.raises(
        ExtensionComponentMaterializationError,
        match="must resolve to a StatLoggerBase subclass",
    ):
        load_stat_logger_plugin_factories()


@pytest.mark.asyncio
async def test_stat_logger_plugin_integration_with_engine(
    monkeypatch: pytest.MonkeyPatch,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "dummy_stat_logger")

        engine_args = AsyncEngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,  # reduce test time
            disable_log_stats=True,  # disable default loggers
        )

        engine = AsyncLLM.from_engine_args(engine_args=engine_args)

        assert len(engine.logger_manager.stat_loggers) == 2
        assert len(engine.logger_manager.stat_loggers[0].per_engine_stat_loggers) == 1
        assert isinstance(
            engine.logger_manager.stat_loggers[0].per_engine_stat_loggers[0],
            DummyStatLogger,
        )

        engine.shutdown()
