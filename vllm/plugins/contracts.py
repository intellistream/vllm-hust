# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed identities shared by extension bundle and domain materializers.

This module deliberately models runtime components, not external systems or
repositories. A Mooncake, LMCache, or PegaFlow deployment is an external KV
state system; the component materialized inside vLLM is its scheduler or worker
connector. Likewise, an external control plane is represented only by a local
bridge component.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_BUNDLE_VERSION = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
EXTENSION_HOST_API_VERSION = "1.0"


class DomainContract(str, Enum):
    """Stable identities for independently versioned runtime contracts."""

    KV_CONNECTOR_SCHEDULER_V1 = "vllm.kv_connector.scheduler.v1"
    KV_CONNECTOR_WORKER_V1 = "vllm.kv_connector.worker.v1"
    SCHEDULER_POLICY_V1 = "vllm.scheduler.policy.v1"
    PLATFORM_V1 = "vllm.platform.v1"
    OPERATOR_V1 = "vllm.operator.v1"
    MODEL_RUNNER_V1 = "vllm.model_runner.v1"
    IO_PROCESSOR_V1 = "vllm.io_processor.v1"
    STAT_LOGGER_V1 = "vllm.stat_logger.v1"
    TELEMETRY_EXPORTER_V1 = "vllm.telemetry.exporter.v1"
    CONTROL_ACTION_V1 = "vllm.control.action.v1"
    CONTROL_RECEIPT_V1 = "vllm.control.receipt.v1"


class ExecutionPlane(str, Enum):
    """Process or device boundary in which a component executes."""

    API = "api"
    SCHEDULER = "scheduler"
    WORKER = "worker"
    NATIVE = "native"
    DEVICE = "device"
    BRIDGE = "bridge"


class ComponentIsolation(str, Enum):
    """Failure-isolation mode, without implying a security sandbox."""

    TRUSTED_IN_PROCESS = "trusted_in_process"
    PROCESS_ISOLATED = "process_isolated"
    SANDBOXED_PROCESS = "sandboxed_process"


class ComponentPermission(str, Enum):
    """Auditable capabilities requested by an extension component.

    Declarations are admission inputs, not proof of enforcement. In particular,
    ``trusted_in_process`` components still run with the host process authority.
    """

    DEVICE_ACCESS = "device_access"
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    IPC = "ipc"
    NETWORK_EGRESS = "network_egress"
    SHARED_MEMORY = "shared_memory"
    SUBPROCESS = "subprocess"


@dataclass(frozen=True, slots=True)
class ExtensionComponentDescriptor:
    """Describe one implementation of one or more typed domain contracts.

    The descriptor is immutable so a validated startup snapshot cannot change
    underneath scheduler or worker materialization.
    """

    component_id: str
    contracts: tuple[DomainContract, ...]
    execution_planes: tuple[ExecutionPlane, ...]
    isolation: ComponentIsolation
    implementation_ref: str
    permissions: tuple[ComponentPermission, ...] = ()

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.component_id):
            raise ValueError(
                "component_id must use lowercase letters, digits, dots, or hyphens"
            )
        if not self.contracts:
            raise ValueError("a component must implement at least one domain contract")
        if len(self.contracts) != len(set(self.contracts)):
            raise ValueError("component contracts must be unique")
        if not self.execution_planes:
            raise ValueError("a component must declare an execution plane")
        if len(self.execution_planes) != len(set(self.execution_planes)):
            raise ValueError("component execution planes must be unique")
        if not self.implementation_ref.strip():
            raise ValueError("implementation_ref must not be empty")
        if len(self.permissions) != len(set(self.permissions)):
            raise ValueError("component permissions must be unique")
        if not all(
            isinstance(permission, ComponentPermission)
            for permission in self.permissions
        ):
            raise ValueError("component permissions must use known identities")


@dataclass(frozen=True, slots=True)
class ExtensionBundleDescriptor:
    """Describe a delivery unit containing independently placed components."""

    bundle_id: str
    bundle_version: str
    host_api_range: str
    components: tuple[ExtensionComponentDescriptor, ...]

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.bundle_id):
            raise ValueError(
                "bundle_id must use lowercase letters, digits, dots, or hyphens"
            )
        if not _BUNDLE_VERSION.fullmatch(self.bundle_version):
            raise ValueError("bundle_version must be a semantic version")
        if not self.host_api_range.strip():
            raise ValueError("host_api_range must not be empty")
        if not self.components:
            raise ValueError("a bundle must contain at least one component")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component ids must be unique within a bundle")
