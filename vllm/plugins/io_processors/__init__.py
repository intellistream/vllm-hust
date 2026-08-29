# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

from vllm.config import VllmConfig
from vllm.plugins import IO_PROCESSOR_PLUGINS_GROUP, load_plugins_by_group
from vllm.plugins.contracts import DomainContract, ExecutionPlane
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.plugins.materialization import (
    ExtensionComponentMaterializationError,
    import_component_implementation,
)
from vllm.renderers import BaseRenderer
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


def _materialize_typed_io_processor(
    component_id: str,
    vllm_config: VllmConfig,
    renderer: BaseRenderer,
) -> IOProcessor:
    """Materialize one explicitly qualified API-plane IO processor."""
    from vllm.plugins.startup import get_configured_extension_startup

    providers = get_configured_extension_startup().snapshot.components_for(
        DomainContract.IO_PROCESSOR_V1,
        ExecutionPlane.API,
    )
    matches = tuple(
        provider for provider in providers if provider.qualified_id == component_id
    )
    if len(matches) != 1:
        available = sorted(provider.qualified_id for provider in providers)
        raise ExtensionComponentMaterializationError(
            "io_processor_plugin selects no admitted API-plane IO processor "
            f"component: {component_id!r}; available components: {available}"
        )

    implementation = import_component_implementation(matches[0], domain="IO processor")
    if not isinstance(implementation, type) or not issubclass(
        implementation, IOProcessor
    ):
        raise ExtensionComponentMaterializationError(
            f"IO processor component {component_id!r} must resolve to an "
            f"IOProcessor subclass (got {implementation!r})"
        )
    try:
        return implementation(vllm_config, renderer)
    except Exception as error:
        raise ExtensionComponentMaterializationError(
            f"IO processor component {component_id!r} failed to initialize"
        ) from error


def has_io_processor(
    vllm_config: VllmConfig,
    plugin_from_init: str | None = None,
):
    if plugin_from_init:
        model_plugin = plugin_from_init
    else:
        # A plugin can be specified via the model config
        # Retrieve the model specific plugin if available
        # This is using a custom field in the hf_config for the model
        hf_config = vllm_config.model_config.hf_config.to_dict()
        config_plugin = hf_config.get("io_processor_plugin")
        model_plugin = config_plugin

    return model_plugin is not None


def get_io_processor(
    vllm_config: VllmConfig,
    renderer: BaseRenderer,
    plugin_from_init: str | None = None,
) -> IOProcessor | None:
    # Input.Output processors are loaded as plugins under the
    # 'vllm.io_processor_plugins' group. Similar to platform
    # plugins, these plugins register a function that returns the class
    # name for the processor to install.

    if plugin_from_init:
        model_plugin = plugin_from_init
    else:
        # A plugin can be specified via the model config
        # Retrieve the model specific plugin if available
        # This is using a custom field in the hf_config for the model
        hf_config = vllm_config.model_config.hf_config.to_dict()
        config_plugin = hf_config.get("io_processor_plugin")
        model_plugin = config_plugin

    if model_plugin is None:
        logger.debug("No IOProcessor plugins requested by the model")
        return None

    # Qualified bundle/component identities opt into the typed materializer.
    # Unqualified names retain the legacy entry-point path unchanged.
    if "/" in model_plugin:
        return _materialize_typed_io_processor(
            model_plugin,
            vllm_config,
            renderer,
        )

    logger.debug("IOProcessor plugin to be loaded %s", model_plugin)

    # Load all installed plugin in the group
    multimodal_data_processor_plugins = load_plugins_by_group(
        IO_PROCESSOR_PLUGINS_GROUP
    )

    loadable_plugins = {}
    for name, func in multimodal_data_processor_plugins.items():
        try:
            assert callable(func)
            processor_cls_qualname = func()
            if processor_cls_qualname is not None:
                loadable_plugins[name] = processor_cls_qualname
        except Exception:
            logger.warning("Failed to load plugin %s.", name, exc_info=True)

    num_available_plugins = len(loadable_plugins.keys())
    if num_available_plugins == 0:
        raise ValueError(
            f"No IOProcessor plugins installed but one is required ({model_plugin})."
        )

    if model_plugin not in loadable_plugins:
        raise ValueError(
            f"The model requires the '{model_plugin}' IO Processor plugin "
            "but it is not installed. "
            f"Available plugins: {list(loadable_plugins.keys())}"
        )

    activated_plugin_cls = resolve_obj_by_qualname(loadable_plugins[model_plugin])

    return activated_plugin_cls(vllm_config, renderer)
