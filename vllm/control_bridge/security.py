# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Authentication and durable replay primitives for the control bridge.

This module deliberately does not execute control actions. It authenticates an
exact wire payload and durably reserves an idempotency key so a future executor
can recover its receipt state after restart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import regex as re

from vllm.control_bridge.contracts import (
    ControlAction,
    ControlActionContractError,
    ControlActionStatus,
    ControlReceipt,
    control_action_to_dict,
    control_receipt_to_dict,
    parse_control_action,
    parse_control_receipt,
)

_SIGNATURE = re.compile(r"^sha256=([0-9a-f]{64})$")
_MINIMUM_HMAC_KEY_BYTES = 32
_DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024
_TERMINAL_STATUSES = frozenset(
    {
        ControlActionStatus.APPLIED,
        ControlActionStatus.REJECTED,
        ControlActionStatus.EXPIRED,
        ControlActionStatus.DUPLICATE,
        ControlActionStatus.FAILED,
    }
)


class ControlActionAuthenticationError(ValueError):
    """Reject an unauthenticated wire payload before contract admission."""


class ReplayLedgerError(RuntimeError):
    """Reject a conflicting or invalid durable replay transition."""


class ReplayDisposition(str, Enum):
    """Stable outcomes from an atomic replay reservation."""

    RESERVED = "reserved"
    DUPLICATE_IN_PROGRESS = "duplicate_in_progress"
    DUPLICATE_TERMINAL = "duplicate_terminal"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class ReplayReservation:
    """Result of binding an idempotency key to an action ID."""

    disposition: ReplayDisposition
    action_id: str
    action: ControlAction | None = None
    receipt: ControlReceipt | None = None


def authenticate_control_action(
    wire_payload: bytes,
    signature: str,
    issuer_keys: Mapping[str, bytes],
    *,
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
) -> ControlAction:
    """Verify an HMAC over exact wire bytes, then parse the strict action.

    The untrusted ``issuer`` field is used only to select a candidate key. The
    action contract is not parsed or admitted until its signature is verified.
    Producers must sign exactly the UTF-8 JSON bytes they transmit.
    """
    if not isinstance(wire_payload, bytes):
        raise ControlActionAuthenticationError("wire payload must be bytes")
    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive")
    if not wire_payload or len(wire_payload) > max_payload_bytes:
        raise ControlActionAuthenticationError("wire payload size is invalid")
    if not isinstance(signature, str):
        raise ControlActionAuthenticationError("signature format is invalid")
    signature_match = _SIGNATURE.fullmatch(signature)
    if signature_match is None:
        raise ControlActionAuthenticationError("signature format is invalid")

    try:
        decoded = wire_payload.decode("utf-8")
        payload = json.loads(decoded, object_pairs_hook=_reject_duplicate_fields)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControlActionAuthenticationError(
            "wire payload is not strict JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ControlActionAuthenticationError("wire payload must be a JSON object")
    issuer = payload.get("issuer")
    if not isinstance(issuer, str) or not issuer:
        raise ControlActionAuthenticationError("wire payload has no usable issuer")
    key = issuer_keys.get(issuer)
    if key is None:
        raise ControlActionAuthenticationError("issuer has no authentication key")
    if not isinstance(key, bytes) or len(key) < _MINIMUM_HMAC_KEY_BYTES:
        raise ControlActionAuthenticationError("issuer authentication key is invalid")

    expected = hmac.new(key, wire_payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_match.group(1)):
        raise ControlActionAuthenticationError("control action signature is invalid")
    try:
        return parse_control_action(payload)
    except ControlActionContractError:
        raise


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


class PersistentReplayLedger:
    """SQLite-backed idempotency and terminal-receipt recovery ledger."""

    def __init__(self, database_path: str | os.PathLike[str]) -> None:
        self._path = Path(database_path)
        self._prepare_database_path()
        self._connection = sqlite3.connect(
            self._path, isolation_level=None, timeout=5.0
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS control_action_replay (
                idempotency_key TEXT PRIMARY KEY,
                action_id TEXT NOT NULL UNIQUE,
                action_fingerprint TEXT NOT NULL,
                action_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('in_progress', 'terminal')),
                receipt_json TEXT,
                CHECK (
                    (state = 'in_progress' AND receipt_json IS NULL) OR
                    (state = 'terminal' AND receipt_json IS NOT NULL)
                )
            )
            """
        )
        os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)

    def __enter__(self) -> PersistentReplayLedger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def reserve(self, action: ControlAction) -> ReplayReservation:
        """Atomically bind the action's key, or describe the prior binding."""
        serialized_action = _serialize_action(action)
        fingerprint = hashlib.sha256(serialized_action.encode("utf-8")).hexdigest()
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """SELECT action_id, state, receipt_json, action_fingerprint,
                          action_json
                   FROM control_action_replay WHERE idempotency_key = ?""",
                (action.idempotency_key,),
            ).fetchone()
            if row is not None:
                if row[0] != action.action_id or row[3] != fingerprint:
                    result = ReplayReservation(ReplayDisposition.CONFLICT, row[0])
                else:
                    result = self._reservation_from_row(row)
                connection.execute("COMMIT")
                return result

            action_row = connection.execute(
                """SELECT idempotency_key, action_fingerprint
                   FROM control_action_replay
                   WHERE action_id = ?""",
                (action.action_id,),
            ).fetchone()
            if action_row is not None:
                connection.execute("COMMIT")
                return ReplayReservation(ReplayDisposition.CONFLICT, action.action_id)

            connection.execute(
                """INSERT INTO control_action_replay
                   (idempotency_key, action_id, action_fingerprint,
                    action_json, state, receipt_json)
                   VALUES (?, ?, ?, ?, 'in_progress', NULL)""",
                (
                    action.idempotency_key,
                    action.action_id,
                    fingerprint,
                    serialized_action,
                ),
            )
            connection.execute("COMMIT")
            return ReplayReservation(
                ReplayDisposition.RESERVED, action.action_id, action=action
            )
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def complete(self, idempotency_key: str, receipt: ControlReceipt) -> ControlReceipt:
        """Persist one immutable terminal receipt for a prior reservation."""
        if not idempotency_key:
            raise ReplayLedgerError("idempotency key must be non-empty")
        normalized = parse_control_receipt(control_receipt_to_dict(receipt))
        if normalized.status not in _TERMINAL_STATUSES:
            raise ReplayLedgerError(
                f"receipt status {normalized.status.value!r} is not terminal"
            )
        serialized = json.dumps(
            control_receipt_to_dict(normalized),
            sort_keys=True,
            separators=(",", ":"),
        )
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """SELECT action_id, state, receipt_json, action_json
                   FROM control_action_replay WHERE idempotency_key = ?""",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                raise ReplayLedgerError("idempotency key was not reserved")
            if row[0] != normalized.action_id:
                raise ReplayLedgerError("receipt action ID does not match reservation")
            reserved_action = self._action_from_json(row[3])
            if normalized.runtime_id != reserved_action.target_runtime_id:
                raise ReplayLedgerError(
                    "receipt runtime ID does not match action target"
                )
            if normalized.trace_id != reserved_action.trace_id:
                raise ReplayLedgerError("receipt trace ID does not match action trace")
            if normalized.causation_id != reserved_action.causation_id:
                raise ReplayLedgerError(
                    "receipt causation ID does not match action causation"
                )
            if row[1] == "terminal":
                existing = self._receipt_from_json(row[2])
                existing_json = json.dumps(
                    control_receipt_to_dict(existing),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if existing_json != serialized:
                    raise ReplayLedgerError("terminal receipt is immutable")
                connection.execute("COMMIT")
                return existing
            connection.execute(
                """UPDATE control_action_replay
                   SET state = 'terminal', receipt_json = ?
                   WHERE idempotency_key = ? AND state = 'in_progress'""",
                (serialized, idempotency_key),
            )
            connection.execute("COMMIT")
            return normalized
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _prepare_database_path(self) -> None:
        if not self._path.name or str(self._path) == ":memory:":
            raise ReplayLedgerError("an explicit database file path is required")
        if not self._path.parent.exists():
            raise ReplayLedgerError("database parent directory does not exist")
        if self._path.exists() or self._path.is_symlink():
            self._validate_existing_database_path()
            return
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except FileExistsError:
            self._validate_existing_database_path()
            return
        try:
            os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        finally:
            os.close(descriptor)

    def _validate_existing_database_path(self) -> None:
        metadata = self._path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ReplayLedgerError("database path must not be a symbolic link")
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayLedgerError("database path must be a regular file")
        os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def _reservation_from_row(row: tuple[Any, ...]) -> ReplayReservation:
        action = PersistentReplayLedger._action_from_json(row[4])
        if row[1] == "in_progress":
            return ReplayReservation(
                ReplayDisposition.DUPLICATE_IN_PROGRESS,
                row[0],
                action=action,
            )
        receipt = PersistentReplayLedger._receipt_from_json(row[2])
        return ReplayReservation(
            ReplayDisposition.DUPLICATE_TERMINAL,
            row[0],
            action=action,
            receipt=receipt,
        )

    @staticmethod
    def _action_from_json(serialized: str) -> ControlAction:
        try:
            payload = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as error:
            raise ReplayLedgerError("persisted action is corrupt") from error
        if not isinstance(payload, dict):
            raise ReplayLedgerError("persisted action is not an object")
        try:
            return parse_control_action(payload)
        except ControlActionContractError as error:
            raise ReplayLedgerError("persisted action violates its contract") from error

    @staticmethod
    def _receipt_from_json(serialized: str) -> ControlReceipt:
        try:
            payload = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as error:
            raise ReplayLedgerError("persisted receipt is corrupt") from error
        if not isinstance(payload, dict):
            raise ReplayLedgerError("persisted receipt is not an object")
        try:
            return parse_control_receipt(payload)
        except ControlActionContractError as error:
            raise ReplayLedgerError(
                "persisted receipt violates its contract"
            ) from error


def _serialize_action(action: ControlAction) -> str:
    return json.dumps(
        control_action_to_dict(action),
        sort_keys=True,
        separators=(",", ":"),
    )
