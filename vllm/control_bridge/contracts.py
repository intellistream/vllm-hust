# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Side-effect-free control action parsing and admission decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

import regex as re


class ControlActionContractError(ValueError):
    """Reject a malformed control action before any runtime mutation."""


class ControlActionType(str, Enum):
    """Action types implemented by the current non-mutating contract phase."""

    RUNTIME_HEALTH_PROBE = "runtime.health_probe"


class ControlAuthorizationScope(str, Enum):
    """Closed authorization scopes understood by the runtime."""

    RUNTIME_READ = "runtime.read"


class ControlActionStatus(str, Enum):
    """Stable admission or terminal receipt statuses."""

    ACCEPTED = "accepted"
    APPLIED = "applied"
    REJECTED = "rejected"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HealthProbePayload:
    """Closed payload for the first read-only action."""

    include_diagnostics: bool


@dataclass(frozen=True, slots=True)
class ControlAction:
    """Authenticated action envelope owned and versioned by the runtime."""

    schema_version: str
    action_id: str
    idempotency_key: str
    action_type: ControlActionType
    target_runtime_id: str
    target_epoch: int
    issued_at: datetime
    expires_at: datetime
    issuer: str
    authorization_scope: ControlAuthorizationScope
    payload: HealthProbePayload
    expected_state_version: int | None
    trace_id: str
    causation_id: str | None


@dataclass(frozen=True, slots=True)
class ControlReceipt:
    """Receipt envelope returned for every admitted or rejected action."""

    schema_version: str
    action_id: str
    runtime_id: str
    observed_epoch: int
    status: ControlActionStatus
    reason_code: str
    diagnostic: str
    mutation_occurred: bool
    resulting_state_version: int | None
    completed_at: datetime
    trace_id: str
    causation_id: str | None


@dataclass(frozen=True, slots=True)
class ControlActionAdmissionContext:
    """Host facts supplied to the pure admission gate."""

    runtime_id: str
    epoch: int
    state_version: int
    trusted_issuers: frozenset[str]
    granted_scopes: frozenset[ControlAuthorizationScope]
    idempotency_ledger: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ControlActionAdmissionDecision:
    """A non-mutating admission result suitable for a future receipt."""

    status: ControlActionStatus
    reason_code: str
    diagnostic: str
    mutation_occurred: bool = False


_ACTION_FIELDS = {
    "schema_version",
    "action_id",
    "idempotency_key",
    "action_type",
    "target_runtime_id",
    "target_epoch",
    "issued_at",
    "expires_at",
    "issuer",
    "authorization_scope",
    "payload",
    "expected_state_version",
    "trace_id",
    "causation_id",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "action_id",
    "runtime_id",
    "observed_epoch",
    "status",
    "reason_code",
    "diagnostic",
    "mutation_occurred",
    "resulting_state_version",
    "completed_at",
    "trace_id",
    "causation_id",
}
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def parse_control_action(payload: Mapping[str, Any]) -> ControlAction:
    """Parse the closed v1 action envelope without executing an action."""
    unknown = set(payload) - _ACTION_FIELDS
    missing = _ACTION_FIELDS - set(payload)
    if unknown:
        raise ControlActionContractError(
            f"control action contains unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise ControlActionContractError(
            f"control action is missing required fields: {sorted(missing)}"
        )
    schema_version = _required_string(payload, "schema_version")
    if schema_version != "1.0":
        raise ControlActionContractError(
            f"unsupported control action schema: {schema_version!r}"
        )
    action_id = _required_string(payload, "action_id")
    try:
        parsed_action_id = UUID(action_id)
    except ValueError as error:
        raise ControlActionContractError("action_id must be a UUID") from error
    if str(parsed_action_id) != action_id:
        raise ControlActionContractError("action_id must use canonical UUID syntax")
    action_type = _enum_value(payload, "action_type", ControlActionType)
    scope = _enum_value(payload, "authorization_scope", ControlAuthorizationScope)
    issued_at = _datetime_value(payload, "issued_at")
    expires_at = _datetime_value(payload, "expires_at")
    if expires_at <= issued_at:
        raise ControlActionContractError("expires_at must be later than issued_at")
    target_epoch = _nonnegative_int(payload, "target_epoch")
    expected_state_version_value = payload["expected_state_version"]
    expected_state_version = None
    if expected_state_version_value is not None:
        expected_state_version = _nonnegative_int(payload, "expected_state_version")
    causation_id_value = payload["causation_id"]
    if causation_id_value is not None and not isinstance(causation_id_value, str):
        raise ControlActionContractError("causation_id must be null or a string")
    raw_action_payload = payload["payload"]
    if not isinstance(raw_action_payload, Mapping):
        raise ControlActionContractError("payload must be an object")
    action_payload = _parse_health_probe_payload(raw_action_payload, action_type)
    idempotency_key = _required_string(payload, "idempotency_key")
    if len(idempotency_key) > 256:
        raise ControlActionContractError(
            "idempotency_key must contain at most 256 characters"
        )
    return ControlAction(
        schema_version=schema_version,
        action_id=action_id,
        idempotency_key=idempotency_key,
        action_type=action_type,
        target_runtime_id=_required_string(payload, "target_runtime_id"),
        target_epoch=target_epoch,
        issued_at=issued_at,
        expires_at=expires_at,
        issuer=_required_string(payload, "issuer"),
        authorization_scope=scope,
        payload=action_payload,
        expected_state_version=expected_state_version,
        trace_id=_required_string(payload, "trace_id"),
        causation_id=causation_id_value,
    )


def parse_control_receipt(payload: Mapping[str, Any]) -> ControlReceipt:
    """Parse the closed receipt envelope before persistence or transmission."""
    unknown = set(payload) - _RECEIPT_FIELDS
    missing = _RECEIPT_FIELDS - set(payload)
    if unknown:
        raise ControlActionContractError(
            f"control receipt contains unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise ControlActionContractError(
            f"control receipt is missing required fields: {sorted(missing)}"
        )
    schema_version = _required_string(payload, "schema_version")
    if schema_version != "1.0":
        raise ControlActionContractError(
            f"unsupported control receipt schema: {schema_version!r}"
        )
    action_id = _required_string(payload, "action_id")
    try:
        parsed_action_id = UUID(action_id)
    except ValueError as error:
        raise ControlActionContractError("receipt action_id must be a UUID") from error
    if str(parsed_action_id) != action_id:
        raise ControlActionContractError(
            "receipt action_id must use canonical UUID syntax"
        )
    reason_code = _required_string(payload, "reason_code")
    if not _REASON_CODE.fullmatch(reason_code):
        raise ControlActionContractError(
            "reason_code must use uppercase letters, digits, and underscores"
        )
    diagnostic = payload["diagnostic"]
    if not isinstance(diagnostic, str) or len(diagnostic) > 1024:
        raise ControlActionContractError(
            "diagnostic must be a string of at most 1024 characters"
        )
    mutation_occurred = payload["mutation_occurred"]
    if not isinstance(mutation_occurred, bool):
        raise ControlActionContractError("mutation_occurred must be a boolean")
    resulting_state_value = payload["resulting_state_version"]
    resulting_state_version = None
    if resulting_state_value is not None:
        resulting_state_version = _nonnegative_int(payload, "resulting_state_version")
    causation_id = payload["causation_id"]
    if causation_id is not None and not isinstance(causation_id, str):
        raise ControlActionContractError("causation_id must be null or a string")
    return ControlReceipt(
        schema_version=schema_version,
        action_id=action_id,
        runtime_id=_required_string(payload, "runtime_id"),
        observed_epoch=_nonnegative_int(payload, "observed_epoch"),
        status=_enum_value(payload, "status", ControlActionStatus),
        reason_code=reason_code,
        diagnostic=diagnostic,
        mutation_occurred=mutation_occurred,
        resulting_state_version=resulting_state_version,
        completed_at=_datetime_value(payload, "completed_at"),
        trace_id=_required_string(payload, "trace_id"),
        causation_id=causation_id,
    )


def control_action_to_dict(action: ControlAction) -> dict[str, Any]:
    """Serialize an action for stable semantic fingerprinting."""
    return {
        "schema_version": action.schema_version,
        "action_id": action.action_id,
        "idempotency_key": action.idempotency_key,
        "action_type": action.action_type.value,
        "target_runtime_id": action.target_runtime_id,
        "target_epoch": action.target_epoch,
        "issued_at": action.issued_at.isoformat(),
        "expires_at": action.expires_at.isoformat(),
        "issuer": action.issuer,
        "authorization_scope": action.authorization_scope.value,
        "payload": {
            "include_diagnostics": action.payload.include_diagnostics,
        },
        "expected_state_version": action.expected_state_version,
        "trace_id": action.trace_id,
        "causation_id": action.causation_id,
    }


def control_receipt_to_dict(receipt: ControlReceipt) -> dict[str, Any]:
    """Serialize a validated receipt without lossy enum or datetime objects."""
    return {
        "schema_version": receipt.schema_version,
        "action_id": receipt.action_id,
        "runtime_id": receipt.runtime_id,
        "observed_epoch": receipt.observed_epoch,
        "status": receipt.status.value,
        "reason_code": receipt.reason_code,
        "diagnostic": receipt.diagnostic,
        "mutation_occurred": receipt.mutation_occurred,
        "resulting_state_version": receipt.resulting_state_version,
        "completed_at": receipt.completed_at.isoformat(),
        "trace_id": receipt.trace_id,
        "causation_id": receipt.causation_id,
    }


def evaluate_control_action_admission(
    action: ControlAction,
    context: ControlActionAdmissionContext,
    *,
    now: datetime,
) -> ControlActionAdmissionDecision:
    """Evaluate authorization and replay facts without mutating the ledger."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if action.target_runtime_id != context.runtime_id:
        return _reject("TARGET_MISMATCH", "action targets another runtime")
    if action.issuer not in context.trusted_issuers:
        return _reject("UNTRUSTED_ISSUER", "issuer is not trusted")
    if action.authorization_scope not in context.granted_scopes:
        return _reject("SCOPE_DENIED", "authorization scope is not granted")
    if now >= action.expires_at:
        return ControlActionAdmissionDecision(
            status=ControlActionStatus.EXPIRED,
            reason_code="ACTION_EXPIRED",
            diagnostic="action deadline has passed",
        )
    if action.target_epoch != context.epoch:
        return _reject("STALE_EPOCH", "target epoch does not match runtime epoch")
    if (
        action.expected_state_version is not None
        and action.expected_state_version != context.state_version
    ):
        return _reject(
            "PRECONDITION_FAILED",
            "expected state version does not match runtime state",
        )
    previous_action_id = context.idempotency_ledger.get(action.idempotency_key)
    if previous_action_id is not None:
        if previous_action_id == action.action_id:
            return ControlActionAdmissionDecision(
                status=ControlActionStatus.DUPLICATE,
                reason_code="DUPLICATE_ACTION",
                diagnostic="action was already admitted under this idempotency key",
            )
        return _reject(
            "IDEMPOTENCY_CONFLICT",
            "idempotency key is already bound to another action",
        )
    return ControlActionAdmissionDecision(
        status=ControlActionStatus.ACCEPTED,
        reason_code="ADMISSION_ACCEPTED",
        diagnostic="action passed the side-effect-free admission gate",
    )


def _reject(reason_code: str, diagnostic: str) -> ControlActionAdmissionDecision:
    return ControlActionAdmissionDecision(
        status=ControlActionStatus.REJECTED,
        reason_code=reason_code,
        diagnostic=diagnostic,
    )


def _parse_health_probe_payload(
    payload: Mapping[str, Any], action_type: ControlActionType
) -> HealthProbePayload:
    if action_type is not ControlActionType.RUNTIME_HEALTH_PROBE:
        raise ControlActionContractError(
            f"unsupported action type: {action_type.value}"
        )
    unknown = set(payload) - {"include_diagnostics"}
    if unknown:
        raise ControlActionContractError(
            f"health probe payload contains unknown fields: {sorted(unknown)}"
        )
    value = payload.get("include_diagnostics")
    if not isinstance(value, bool):
        raise ControlActionContractError("include_diagnostics must be a boolean")
    return HealthProbePayload(include_diagnostics=value)


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value:
        raise ControlActionContractError(f"{field} must be a non-empty string")
    return value


def _nonnegative_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ControlActionContractError(f"{field} must be a non-negative integer")
    return value


def _datetime_value(payload: Mapping[str, Any], field: str) -> datetime:
    value = _required_string(payload, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ControlActionContractError(
            f"{field} must be an RFC 3339 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise ControlActionContractError(f"{field} must include a timezone")
    return parsed


def _enum_value(payload: Mapping[str, Any], field: str, enum_type: type[Enum]):
    value = _required_string(payload, field)
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = [item.value for item in enum_type]
        raise ControlActionContractError(
            f"{field} must be one of {allowed}, got {value!r}"
        ) from error
