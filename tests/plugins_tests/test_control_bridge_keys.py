# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest

from vllm.control_bridge.keys import (
    ControlHmacKey,
    ControlKeyConfigurationError,
    ControlKeySet,
    ReloadableControlKeyStore,
    build_control_action_signature,
)
from vllm.control_bridge.security import (
    ControlActionAuthenticationError,
    authenticate_control_action,
)

_OLD = b"old-control-key-material-at-least-32-bytes"
_NEW = b"new-control-key-material-at-least-32-bytes"


def at(minute: int) -> datetime:
    return datetime(2026, 8, 29, 0, minute, tzinfo=timezone.utc)


def wire() -> bytes:
    return json.dumps(
        {
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
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def key(
    key_id: str,
    secret: bytes,
    *,
    not_before: datetime,
    not_after: datetime | None = None,
    revoked_at: datetime | None = None,
) -> ControlHmacKey:
    return ControlHmacKey(
        issuer="ride.example",
        key_id=key_id,
        secret=secret,
        not_before=not_before,
        not_after=not_after,
        revoked_at=revoked_at,
    )


def test_versioned_signature_selects_exact_issuer_key() -> None:
    payload = wire()
    keys = ControlKeySet(
        generation=1,
        keys=(key("old-1", _OLD, not_before=at(0), not_after=at(3)),),
    )
    signature = build_control_action_signature(payload, key_id="old-1", secret=_OLD)

    action = authenticate_control_action(payload, signature, keys, now=at(1))

    assert action.issuer == "ride.example"
    assert "old-control-key" not in repr(keys)


def test_rotation_overlap_accepts_both_exact_key_ids() -> None:
    payload = wire()
    keys = ControlKeySet(
        generation=2,
        keys=(
            key("old-1", _OLD, not_before=at(0), not_after=at(3)),
            key("new-2", _NEW, not_before=at(1)),
        ),
    )

    old_action = authenticate_control_action(
        payload,
        build_control_action_signature(payload, key_id="old-1", secret=_OLD),
        keys,
        now=at(2),
    )
    new_action = authenticate_control_action(
        payload,
        build_control_action_signature(payload, key_id="new-2", secret=_NEW),
        keys,
        now=at(2),
    )

    assert old_action == new_action


@pytest.mark.parametrize(
    ("record", "when"),
    [
        (key("future", _OLD, not_before=at(2)), at(1)),
        (key("expired", _OLD, not_before=at(0), not_after=at(1)), at(1)),
        (key("revoked", _OLD, not_before=at(0), revoked_at=at(1)), at(1)),
    ],
)
def test_inactive_expired_or_revoked_keys_fail_closed(record, when) -> None:
    payload = wire()
    keys = ControlKeySet(generation=1, keys=(record,))
    signature = build_control_action_signature(
        payload, key_id=record.key_id, secret=record.secret
    )

    with pytest.raises(ControlActionAuthenticationError, match="failed"):
        authenticate_control_action(payload, signature, keys, now=when)


def test_unknown_key_id_and_legacy_downgrade_do_not_fallback() -> None:
    payload = wire()
    keys = ControlKeySet(
        generation=1,
        keys=(key("known", _OLD, not_before=at(0)),),
    )
    unknown = build_control_action_signature(payload, key_id="unknown", secret=_OLD)
    legacy = "sha256=" + hmac.new(_OLD, payload, hashlib.sha256).hexdigest()

    with pytest.raises(ControlActionAuthenticationError):
        authenticate_control_action(payload, unknown, keys, now=at(1))
    with pytest.raises(ControlActionAuthenticationError, match="format"):
        authenticate_control_action(payload, legacy, keys, now=at(1))


def test_reloadable_store_requires_monotonic_generation() -> None:
    first = ControlKeySet(
        generation=3,
        keys=(key("old", _OLD, not_before=at(0)),),
    )
    replacement = ControlKeySet(
        generation=4,
        keys=(key("new", _NEW, not_before=at(0)),),
    )
    store = ReloadableControlKeyStore(first)

    store.replace(replacement)

    assert store.snapshot() is replacement
    with pytest.raises(ControlKeyConfigurationError, match="must increase"):
        store.replace(first)
