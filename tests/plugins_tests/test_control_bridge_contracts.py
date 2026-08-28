# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from datetime import datetime, timezone
from importlib import resources

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from vllm.control_bridge.contracts import (
    ControlActionAdmissionContext,
    ControlActionContractError,
    ControlActionStatus,
    ControlAuthorizationScope,
    evaluate_control_action_admission,
    parse_control_action,
)


def valid_action() -> dict:
    return {
        "schema_version": "1.0",
        "action_id": "123e4567-e89b-12d3-a456-426614174000",
        "idempotency_key": "probe-1",
        "action_type": "runtime.health_probe",
        "target_runtime_id": "runtime-a",
        "target_epoch": 7,
        "issued_at": "2026-08-29T00:00:00+00:00",
        "expires_at": "2026-08-29T00:05:00+00:00",
        "issuer": "ride.example",
        "authorization_scope": "runtime.read",
        "payload": {"include_diagnostics": False},
        "expected_state_version": 11,
        "trace_id": "trace-1",
        "causation_id": None,
    }


def context(**overrides) -> ControlActionAdmissionContext:
    values = {
        "runtime_id": "runtime-a",
        "epoch": 7,
        "state_version": 11,
        "trusted_issuers": frozenset({"ride.example"}),
        "granted_scopes": frozenset({ControlAuthorizationScope.RUNTIME_READ}),
        "idempotency_ledger": {},
    }
    values.update(overrides)
    return ControlActionAdmissionContext(**values)


def now() -> datetime:
    return datetime(2026, 8, 29, 0, 1, tzinfo=timezone.utc)


def test_valid_read_only_action_is_admitted_without_mutation() -> None:
    decision = evaluate_control_action_admission(
        parse_control_action(valid_action()), context(), now=now()
    )

    assert decision.status is ControlActionStatus.ACCEPTED
    assert decision.reason_code == "ADMISSION_ACCEPTED"
    assert decision.mutation_occurred is False


def test_packaged_action_and_receipt_schemas_validate_wire_examples() -> None:
    packaged = resources.files("vllm.plugins")
    action_schema = json.loads(
        packaged.joinpath("control_action.schema.json").read_text(encoding="utf-8")
    )
    receipt_schema = json.loads(
        packaged.joinpath("control_receipt.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(action_schema, format_checker=FormatChecker()).validate(
        valid_action()
    )
    Draft202012Validator(receipt_schema, format_checker=FormatChecker()).validate(
        {
            "schema_version": "1.0",
            "action_id": "123e4567-e89b-12d3-a456-426614174000",
            "runtime_id": "runtime-a",
            "observed_epoch": 7,
            "status": "accepted",
            "reason_code": "ADMISSION_ACCEPTED",
            "diagnostic": "side-effect-free admission passed",
            "mutation_occurred": False,
            "resulting_state_version": 11,
            "completed_at": "2026-08-29T00:01:00+00:00",
            "trace_id": "trace-1",
            "causation_id": None,
        }
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(extra=True), "unknown fields"),
        (lambda payload: payload.pop("issuer"), "missing required"),
        (lambda payload: payload.update(schema_version="2.0"), "unsupported"),
        (lambda payload: payload.update(action_id="not-a-uuid"), "UUID"),
        (lambda payload: payload.update(target_epoch=True), "non-negative integer"),
        (
            lambda payload: payload["payload"].update(command="mutate"),
            "unknown fields",
        ),
        (
            lambda payload: payload.update(expires_at=payload["issued_at"]),
            "later than",
        ),
    ],
)
def test_parser_rejects_ambiguous_or_malformed_actions(mutation, message) -> None:
    payload = valid_action()
    mutation(payload)

    with pytest.raises(ControlActionContractError, match=message):
        parse_control_action(payload)


@pytest.mark.parametrize(
    ("context_override", "action_override", "status", "reason"),
    [
        ({"runtime_id": "runtime-b"}, {}, "rejected", "TARGET_MISMATCH"),
        ({"trusted_issuers": frozenset()}, {}, "rejected", "UNTRUSTED_ISSUER"),
        ({"granted_scopes": frozenset()}, {}, "rejected", "SCOPE_DENIED"),
        ({"epoch": 8}, {}, "rejected", "STALE_EPOCH"),
        ({"state_version": 12}, {}, "rejected", "PRECONDITION_FAILED"),
        (
            {"idempotency_ledger": {"probe-1": "another-action"}},
            {},
            "rejected",
            "IDEMPOTENCY_CONFLICT",
        ),
        (
            {"idempotency_ledger": {"probe-1": "123e4567-e89b-12d3-a456-426614174000"}},
            {},
            "duplicate",
            "DUPLICATE_ACTION",
        ),
    ],
)
def test_admission_rejects_without_partial_mutation(
    context_override, action_override, status, reason
) -> None:
    payload = valid_action()
    payload.update(action_override)
    decision = evaluate_control_action_admission(
        parse_control_action(payload), context(**context_override), now=now()
    )

    assert decision.status.value == status
    assert decision.reason_code == reason
    assert decision.mutation_occurred is False


def test_expired_action_has_distinct_status() -> None:
    action = parse_control_action(valid_action())
    decision = evaluate_control_action_admission(
        action,
        context(),
        now=datetime(2026, 8, 29, 0, 6, tzinfo=timezone.utc),
    )

    assert decision.status is ControlActionStatus.EXPIRED
    assert decision.mutation_occurred is False
