# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime-owned contracts for an external control-plane bridge."""

from vllm.control_bridge.bootstrap import (
    ControlBridgeBootstrapError,
    ControlBridgeHostConfig,
    ControlBridgeHostLimits,
    ManagedControlBridgeRuntime,
    load_control_bridge_host_config,
)
from vllm.control_bridge.contracts import (
    ControlAction,
    ControlActionAdmissionContext,
    ControlActionAdmissionDecision,
    ControlActionContractError,
    ControlActionStatus,
    ControlReceipt,
    control_action_to_dict,
    control_receipt_to_dict,
    evaluate_control_action_admission,
    parse_control_action,
    parse_control_receipt,
)
from vllm.control_bridge.executor import (
    ControlBridgeBackpressureError,
    ControlBridgeExecutorError,
    ControlBridgeExecutorState,
    ProcessIsolatedControlBridgeExecutor,
    materialize_process_isolated_control_bridge,
)
from vllm.control_bridge.keys import (
    ControlHmacKey,
    ControlKeyConfigurationError,
    ControlKeySet,
    ReloadableControlKeyStore,
    build_control_action_signature,
)
from vllm.control_bridge.runtime_health import (
    RuntimeHealthObservation,
    RuntimeHealthState,
    observe_engine_client_health,
)
from vllm.control_bridge.security import (
    ControlActionAuthenticationError,
    PersistentReplayLedger,
    ReplayDisposition,
    ReplayLedgerError,
    ReplayReservation,
    authenticate_control_action,
)
from vllm.control_bridge.service import LocalControlBridgeService
from vllm.control_bridge.transport import (
    ControlTransportError,
    ControlTransportProtocolError,
    ControlTransportState,
    UnixControlBridgeHost,
    encode_control_request,
    read_control_response,
)

__all__ = [
    "ControlBridgeBootstrapError",
    "ControlBridgeHostConfig",
    "ControlBridgeHostLimits",
    "ControlAction",
    "ControlActionAdmissionContext",
    "ControlActionAdmissionDecision",
    "ControlActionAuthenticationError",
    "ControlActionContractError",
    "ControlActionStatus",
    "ControlBridgeBackpressureError",
    "ControlBridgeExecutorError",
    "ControlBridgeExecutorState",
    "ControlHmacKey",
    "ControlKeyConfigurationError",
    "ControlKeySet",
    "ControlTransportError",
    "ControlTransportProtocolError",
    "ControlTransportState",
    "ControlReceipt",
    "LocalControlBridgeService",
    "ManagedControlBridgeRuntime",
    "PersistentReplayLedger",
    "ProcessIsolatedControlBridgeExecutor",
    "ReplayDisposition",
    "ReplayLedgerError",
    "ReplayReservation",
    "ReloadableControlKeyStore",
    "RuntimeHealthObservation",
    "RuntimeHealthState",
    "UnixControlBridgeHost",
    "authenticate_control_action",
    "build_control_action_signature",
    "control_action_to_dict",
    "control_receipt_to_dict",
    "encode_control_request",
    "evaluate_control_action_admission",
    "materialize_process_isolated_control_bridge",
    "load_control_bridge_host_config",
    "observe_engine_client_health",
    "parse_control_action",
    "parse_control_receipt",
    "read_control_response",
]
