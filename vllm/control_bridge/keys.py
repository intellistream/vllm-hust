# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Versioned HMAC key lifecycle for control action authentication."""

from __future__ import annotations

import hashlib
import hmac
import threading
from dataclasses import dataclass, field
from datetime import datetime

import regex as re

_KEY_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MINIMUM_SECRET_BYTES = 32


class ControlKeyConfigurationError(ValueError):
    """Reject an invalid key set or unsafe key generation transition."""


class ControlKeyResolutionError(ValueError):
    """Reject a key that is unknown, inactive, expired, or revoked."""


@dataclass(frozen=True, slots=True)
class ControlHmacKey:
    """One issuer-scoped HMAC key with an explicit validity window."""

    issuer: str
    key_id: str
    secret: bytes = field(repr=False)
    not_before: datetime
    not_after: datetime | None = None
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.issuer:
            raise ControlKeyConfigurationError("key issuer must be non-empty")
        if not _KEY_ID.fullmatch(self.key_id):
            raise ControlKeyConfigurationError("key_id is not canonical")
        if (
            not isinstance(self.secret, bytes)
            or len(self.secret) < _MINIMUM_SECRET_BYTES
        ):
            raise ControlKeyConfigurationError(
                "HMAC key secret must contain at least 32 bytes"
            )
        _require_aware(self.not_before, "not_before")
        if self.not_after is not None:
            _require_aware(self.not_after, "not_after")
            if self.not_after <= self.not_before:
                raise ControlKeyConfigurationError(
                    "not_after must be later than not_before"
                )
        if self.revoked_at is not None:
            _require_aware(self.revoked_at, "revoked_at")
            if self.revoked_at < self.not_before:
                raise ControlKeyConfigurationError(
                    "revoked_at must not precede not_before"
                )

    def is_active_at(self, now: datetime) -> bool:
        _require_aware(now, "now")
        return (
            now >= self.not_before
            and (self.not_after is None or now < self.not_after)
            and (self.revoked_at is None or now < self.revoked_at)
        )


@dataclass(frozen=True, slots=True)
class ControlKeySet:
    """Immutable key generation atomically selected by the host."""

    generation: int
    keys: tuple[ControlHmacKey, ...]
    _by_identity: dict[tuple[str, str], ControlHmacKey] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation < 0
        ):
            raise ControlKeyConfigurationError(
                "key-set generation must be a non-negative integer"
            )
        if not self.keys:
            raise ControlKeyConfigurationError("key set must not be empty")
        identities = [(key.issuer, key.key_id) for key in self.keys]
        if len(identities) != len(set(identities)):
            raise ControlKeyConfigurationError("issuer and key_id pairs must be unique")
        object.__setattr__(
            self,
            "_by_identity",
            dict(zip(identities, self.keys)),
        )

    @property
    def issuers(self) -> frozenset[str]:
        return frozenset(key.issuer for key in self.keys)

    def resolve(self, issuer: str, key_id: str, *, now: datetime) -> bytes:
        key = self._by_identity.get((issuer, key_id))
        if key is None or not key.is_active_at(now):
            raise ControlKeyResolutionError("control authentication key is unavailable")
        return key.secret


class ReloadableControlKeyStore:
    """Atomically replace immutable key generations without rollback."""

    def __init__(self, initial: ControlKeySet) -> None:
        self._lock = threading.Lock()
        self._current = initial

    def snapshot(self) -> ControlKeySet:
        with self._lock:
            return self._current

    def replace(self, replacement: ControlKeySet) -> None:
        with self._lock:
            if replacement.generation <= self._current.generation:
                raise ControlKeyConfigurationError(
                    "replacement key generation must increase"
                )
            self._current = replacement


def build_control_action_signature(
    wire_payload: bytes,
    *,
    key_id: str,
    secret: bytes,
) -> str:
    """Build the canonical v1 signature over exact transmitted bytes."""
    if not isinstance(wire_payload, bytes) or not wire_payload:
        raise ValueError("wire payload must be non-empty bytes")
    if not _KEY_ID.fullmatch(key_id):
        raise ValueError("key_id is not canonical")
    if not isinstance(secret, bytes) or len(secret) < _MINIMUM_SECRET_BYTES:
        raise ValueError("HMAC secret must contain at least 32 bytes")
    digest = hmac.new(secret, wire_payload, hashlib.sha256).hexdigest()
    return f"v1;kid={key_id};sha256={digest}"


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None:
        raise ControlKeyConfigurationError(f"{field_name} must be timezone-aware")
