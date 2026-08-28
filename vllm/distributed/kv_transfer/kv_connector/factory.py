# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase,
    KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorRole,
    supports_hma,
)
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.plugins.kv_connector_selection import (
        ResolvedKVConnectorPair,
        ResolvedKVConnectorSelection,
    )
    from vllm.plugins.snapshot import ResolvedExtensionComponent
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class KVConnectorMaterializationError(RuntimeError):
    """Fail closed after a typed KV connector topology was selected."""


class KVConnectorFactory:
    """Materialize scheduler and worker implementations of the KV contract.

    A connector is a domain-specific adapter between vLLM and a KV state
    system. Registration or packaging may be provided by an extension bundle,
    but the external state system itself is not a vLLM plugin component.
    """

    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> KVConnectorBase:
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        if kv_transfer_config.kv_connector_selection is None:
            connector_cls = cls.get_connector_class(kv_transfer_config)
        else:
            resolved = cls.resolve_typed_selection(kv_transfer_config)
            if len(resolved.connectors) != 1:
                raise KVConnectorMaterializationError(
                    "typed ordered_multi construction is not implemented; "
                    "continue using the legacy MultiConnector compatibility path"
                )
            resolved_pair = resolved.connectors[0]
            connector_cls = cls.get_typed_connector_class(resolved_pair, role)
            cls.verify_typed_connector_declarations(
                connector_cls, resolved_pair, role, config
            )

        # check if the connector supports HMA
        hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
        if hma_enabled and not cls.supports_hma_config(kv_transfer_config):
            raise ValueError(
                f"Connector {connector_cls.__name__} does not support HMA but "
                f"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."
            )

        logger.info(
            "Creating v1 connector with name: %s and engine_id: %s",
            connector_cls.__name__,
            kv_transfer_config.engine_id,
        )
        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        return connector_cls(config, role, kv_cache_config)

    @classmethod
    def resolve_typed_selection(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> "ResolvedKVConnectorSelection":
        """Resolve a typed topology against the immutable startup snapshot."""
        profile = kv_transfer_config.kv_connector_selection
        if profile is None:
            raise KVConnectorMaterializationError(
                "typed KV connector selection is not configured"
            )
        from vllm.plugins.kv_connector_selection import (
            KVConnectorSelectionError,
            resolve_kv_connector_selection,
        )
        from vllm.plugins.startup import get_configured_extension_startup

        try:
            snapshot = get_configured_extension_startup().snapshot
            return resolve_kv_connector_selection(profile, snapshot)
        except KVConnectorSelectionError as error:
            raise KVConnectorMaterializationError(
                "typed KV connector selection could not be resolved"
            ) from error

    @classmethod
    def get_typed_connector_class(
        cls,
        pair: "ResolvedKVConnectorPair",
        role: KVConnectorRole,
    ) -> type[KVConnectorBaseType]:
        """Import only the connector component owned by the current process."""
        component = (
            pair.scheduler_component
            if role is KVConnectorRole.SCHEDULER
            else pair.worker_component
        )
        implementation = cls._load_typed_implementation(component, role.name.lower())
        if not isinstance(implementation, type) or not issubclass(
            implementation, KVConnectorBase
        ):
            raise KVConnectorMaterializationError(
                f"typed KV component {component.qualified_id!r} must resolve to "
                "a KVConnectorBase class"
            )
        cls._validate_constructor(implementation, component.qualified_id)
        return cast(type[KVConnectorBaseType], implementation)

    @classmethod
    def get_telemetry_class(cls, kv_transfer_config: "KVTransferConfig") -> type[Any]:
        """Import the API-plane stats codec without importing a worker component."""
        if kv_transfer_config.kv_connector_selection is None:
            return cls.get_connector_class(kv_transfer_config)
        resolved = cls.resolve_typed_selection(kv_transfer_config)
        if len(resolved.connectors) != 1:
            raise KVConnectorMaterializationError(
                "typed ordered_multi telemetry is not implemented"
            )
        component = resolved.connectors[0].telemetry_component
        implementation = cls._load_typed_implementation(component, "telemetry")
        if not isinstance(implementation, type):
            raise KVConnectorMaterializationError(
                f"typed KV telemetry component {component.qualified_id!r} must "
                "resolve to a class"
            )
        for method_name in ("build_kv_connector_stats", "build_prom_metrics"):
            if not callable(getattr(implementation, method_name, None)):
                raise KVConnectorMaterializationError(
                    f"typed KV telemetry component {component.qualified_id!r} "
                    f"does not expose callable {method_name}"
                )
        return implementation

    @classmethod
    def _load_typed_implementation(
        cls, component: "ResolvedExtensionComponent", role_name: str
    ) -> Any:
        implementation_ref = component.component.implementation_ref
        module_name, separator, attribute_path = implementation_ref.partition(":")
        if not separator or not module_name or not attribute_path:
            raise KVConnectorMaterializationError(
                f"typed KV {role_name} component {component.qualified_id!r} must "
                "use module:attribute implementation_ref syntax"
            )
        try:
            implementation: Any = importlib.import_module(module_name)
            for attribute in attribute_path.split("."):
                if not attribute:
                    raise AttributeError("empty attribute segment")
                implementation = getattr(implementation, attribute)
            return implementation
        except Exception as error:
            raise KVConnectorMaterializationError(
                f"failed to import typed KV {role_name} component "
                f"{component.qualified_id!r}"
            ) from error

    @classmethod
    def _validate_constructor(
        cls, connector_cls: type[Any], component_name: str
    ) -> None:
        if not supports_kw(connector_cls, "kv_cache_config"):
            raise KVConnectorMaterializationError(
                f"typed KV component {component_name!r} must accept "
                "kv_cache_config as the third constructor argument"
            )

    @classmethod
    def verify_typed_connector_declarations(
        cls,
        connector_cls: type[KVConnectorBaseType],
        pair: "ResolvedKVConnectorPair",
        role: KVConnectorRole,
        config: "VllmConfig",
    ) -> None:
        """Verify declarations only after importing in their owning process."""
        actual_hma = supports_hma(connector_cls)
        if role is KVConnectorRole.SCHEDULER:
            scheduler_capabilities = pair.scheduler_capabilities
            if actual_hma is not scheduler_capabilities.supports_hma:
                raise KVConnectorMaterializationError(
                    "typed KV scheduler component HMA declaration mismatch: "
                    f"declared={scheduler_capabilities.supports_hma}, "
                    f"actual={actual_hma}"
                )
            return
        worker_capabilities = pair.worker_capabilities
        if actual_hma is not worker_capabilities.supports_hma:
            raise KVConnectorMaterializationError(
                "typed KV worker component HMA declaration mismatch: "
                f"declared={worker_capabilities.supports_hma}, actual={actual_hma}"
            )
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise KVConnectorMaterializationError(
                "kv_transfer_config disappeared during typed KV verification"
            )
        extra_config = kv_transfer_config.kv_connector_extra_config
        actual_piecewise = connector_cls.requires_piecewise_for_cudagraph(extra_config)
        if actual_piecewise is not worker_capabilities.requires_piecewise_for_cudagraph:
            raise KVConnectorMaterializationError(
                "typed KV worker piecewise declaration mismatch: "
                f"declared={worker_capabilities.requires_piecewise_for_cudagraph}, "
                f"actual={actual_piecewise}"
            )
        actual_layout = connector_cls.get_required_kvcache_layout(config)
        declared_layout = worker_capabilities.required_kv_cache_layout
        declared_layout_value = declared_layout.value if declared_layout else None
        if actual_layout != declared_layout_value:
            raise KVConnectorMaterializationError(
                "typed KV worker cache-layout declaration mismatch: "
                f"declared={declared_layout_value!r}, actual={actual_layout!r}"
            )

    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBaseType]:
        """Get a registered connector class by name.

        Raises ValueError if the connector is not registered.

        Args:
            connector_name: Name of the registered connector.

        Returns:
            The connector class.
        """
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()

    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBaseType]:
        if kv_transfer_config.kv_connector_selection is not None:
            raise KVConnectorMaterializationError(
                "typed KV connector classes are role-specific; use "
                "get_typed_connector_class"
            )
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in KVTransferConfig")
        connector_module_path = kv_transfer_config.kv_connector_module_path
        if connector_module_path is not None and not connector_module_path:
            raise ValueError("kv_connector_module_path cannot be an empty string.")
        if connector_module_path:
            # External module path takes priority over internal registry.
            connector_module = importlib.import_module(connector_module_path)
            try:
                connector_cls = getattr(connector_module, connector_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class {connector_name} not found in {connector_module_path}"
                ) from e
            connector_cls = cast(type[KVConnectorBaseType], connector_cls)
            if not supports_kw(connector_cls, "kv_cache_config"):
                msg = (
                    f"Connector {connector_cls.__name__} uses deprecated "
                    "2-argument constructor signature. External v1 KV "
                    "connectors must accept kv_cache_config as the third "
                    "constructor argument and pass it to super().__init__()."
                )
                logger.error(msg)
                raise ValueError(msg)
        elif connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        return connector_cls

    @classmethod
    def supports_hma_config(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        """Return whether this KV transfer config supports HMA.

        MultiConnector is a special case: the wrapper class implements
        SupportsHMA, but effective support depends on every configured child.
        """
        if kv_transfer_config.kv_connector_selection is not None:
            return cls.resolve_typed_selection(kv_transfer_config).supports_hma
        connector_cls = cls.get_connector_class(kv_transfer_config)
        if kv_transfer_config.kv_connector != "MultiConnector":
            return supports_hma(connector_cls)

        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        return MultiConnector.all_children_support_hma(kv_transfer_config)

    @classmethod
    def requires_piecewise_config(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        """Read typed declarations without importing a worker implementation."""
        if kv_transfer_config.kv_connector_selection is not None:
            return cls.resolve_typed_selection(
                kv_transfer_config
            ).requires_piecewise_for_cudagraph
        connector_cls = cls.get_connector_class(kv_transfer_config)
        return connector_cls.requires_piecewise_for_cudagraph(
            kv_transfer_config.kv_connector_extra_config
        )

    @classmethod
    def required_kv_cache_layout(cls, config: "VllmConfig") -> str | None:
        """Read a typed layout declaration without importing worker code."""
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            return None
        if kv_transfer_config.kv_connector_selection is not None:
            layout = cls.resolve_typed_selection(
                kv_transfer_config
            ).required_kv_cache_layout
            return layout.value if layout is not None else None
        connector_cls = cls.get_connector_class(kv_transfer_config)
        return connector_cls.get_required_kvcache_layout(config)


# Register built-in connector adapters here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

KVConnectorFactory.register_connector(
    "ExampleConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.example_connector",
    "ExampleConnector",
)

KVConnectorFactory.register_connector(
    "ExampleHiddenStatesConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector",
    "ExampleHiddenStatesConnector",
)

KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1",
)

KVConnectorFactory.register_connector(
    "LMCacheMPConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
    "LMCacheMPConnector",
)

KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlConnector",
)

KVConnectorFactory.register_connector(
    "NixlPullConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlPullConnector",
)

KVConnectorFactory.register_connector(
    "NixlPushConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlPushConnector",
)

KVConnectorFactory.register_connector(
    "MultiConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
    "MultiConnector",
)

KVConnectorFactory.register_connector(
    "MoRIIOConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector",
    "MoRIIOConnector",
)

KVConnectorFactory.register_connector(
    "OffloadingConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
    "OffloadingConnector",
)

KVConnectorFactory.register_connector(
    "DecodeBenchConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector",
    "DecodeBenchConnector",
)

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
    "MooncakeConnector",
)
KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector",
    "MooncakeStoreConnector",
)
KVConnectorFactory.register_connector(
    "FlexKVConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector",
    "FlexKVConnectorV1",
)
KVConnectorFactory.register_connector(
    "SimpleCPUOffloadConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector",
    "SimpleCPUOffloadConnector",
)
KVConnectorFactory.register_connector(
    "HF3FSKVConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector",
    "HF3FSKVConnector",
)
