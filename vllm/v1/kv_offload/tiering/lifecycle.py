# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-level lifecycle metadata for tiered KV offloading.

This module intentionally does not move KV data by itself.  The existing
TieringOffloadingManager owns primary/secondary transfers; the lifecycle layer
tracks which request/session owns retained block keys and when idle retained
state should expire.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    IDLE_RETAINED = "idle_retained"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass(slots=True)
class SessionKVState:
    session_id: str
    status: LifecycleStatus
    created_at: float
    last_access_at: float
    ttl_deadline: float | None = None
    active_req_ids: set[str] = field(default_factory=set)
    retained_req_ids: set[str] = field(default_factory=set)
    block_keys: set[OffloadKey] = field(default_factory=set)

    @property
    def is_idle(self) -> bool:
        return self.status is LifecycleStatus.IDLE_RETAINED


@dataclass(slots=True)
class LifecycleConfig:
    idle_ttl_sec: float = 0.0
    delete_expired_secondary: bool = False

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> "LifecycleConfig":
        return cls(
            idle_ttl_sec=float(extra_config.get("lifecycle_idle_ttl_sec", 0.0)),
            delete_expired_secondary=bool(
                extra_config.get("lifecycle_delete_expired_secondary", False)
            ),
        )


def get_session_id(req_context: ReqContext) -> str:
    """Return the lifecycle session id for a request.

    The caller may pass one of the accepted keys through kv_transfer_params.
    Falling back to req_id preserves current behavior when no session identity
    is provided.
    """

    params = req_context.kv_transfer_params or {}
    for key in ("session_id", "conversation_id", "kv_session_id"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return req_context.req_id


class SessionLifecycleManager:
    """Tracks active and idle-retained KV sessions.

    This manager is deliberately conservative:
    - TTL defaults to disabled (0 seconds), so existing FS reuse behavior is
      unchanged unless explicitly configured.
    - Expiration deletes secondary tier block files only when
      lifecycle_delete_expired_secondary is enabled and the tier exposes a
      file_mapper-compatible get_file_name().
    """

    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._sessions: dict[str, SessionKVState] = {}
        self._req_to_session: dict[str, str] = {}

    def on_new_request(self, req_context: ReqContext) -> str:
        now = time.monotonic()
        session_id = get_session_id(req_context)
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionKVState(
                session_id=session_id,
                status=LifecycleStatus.ACTIVE,
                created_at=now,
                last_access_at=now,
            )
            self._sessions[session_id] = state
        else:
            state.status = LifecycleStatus.ACTIVE
            state.last_access_at = now
            state.ttl_deadline = None

        state.active_req_ids.add(req_context.req_id)
        self._req_to_session[req_context.req_id] = session_id
        return session_id

    def record_request_keys(
        self, req_context: ReqContext, keys: Iterable[OffloadKey]
    ) -> None:
        state = self._get_by_req(req_context)
        if state is None:
            return
        state.block_keys.update(keys)
        state.last_access_at = time.monotonic()

    def on_request_finished(self, req_context: ReqContext) -> None:
        state = self._get_by_req(req_context)
        if state is None:
            return

        now = time.monotonic()
        state.active_req_ids.discard(req_context.req_id)
        state.retained_req_ids.add(req_context.req_id)
        state.last_access_at = now
        if not state.active_req_ids:
            state.status = LifecycleStatus.IDLE_RETAINED
            if self.config.idle_ttl_sec > 0:
                state.ttl_deadline = now + self.config.idle_ttl_sec
            else:
                state.ttl_deadline = None

    def expire_idle_sessions(self, secondary_tiers: Iterable[object]) -> int:
        if self.config.idle_ttl_sec <= 0:
            return 0

        now = time.monotonic()
        expired: list[SessionKVState] = []
        for state in self._sessions.values():
            if not state.is_idle or state.ttl_deadline is None:
                continue
            if state.ttl_deadline <= now:
                expired.append(state)

        for state in expired:
            state.status = LifecycleStatus.EXPIRED
            if self.config.delete_expired_secondary:
                self._delete_secondary_blocks(state, secondary_tiers)
            self._delete_state(state)
        return len(expired)

    def has_pending_expiration(self) -> bool:
        """Return whether idle lifecycle state still needs scheduler ticks."""
        if self.config.idle_ttl_sec <= 0:
            return False
        return any(
            state.is_idle and state.ttl_deadline is not None
            for state in self._sessions.values()
        )

    def reset_active_primary_state(self) -> None:
        """Keep idle metadata while dropping active request mappings."""
        for state in self._sessions.values():
            state.active_req_ids.clear()
            if state.status is LifecycleStatus.ACTIVE:
                state.status = LifecycleStatus.IDLE_RETAINED
                if self.config.idle_ttl_sec > 0:
                    state.ttl_deadline = time.monotonic() + self.config.idle_ttl_sec
        self._req_to_session.clear()

    def snapshot(self) -> dict[str, int]:
        active = idle = 0
        retained_blocks = 0
        for state in self._sessions.values():
            retained_blocks += len(state.block_keys)
            if state.status is LifecycleStatus.ACTIVE:
                active += 1
            elif state.status is LifecycleStatus.IDLE_RETAINED:
                idle += 1
        return {
            "sessions": len(self._sessions),
            "active_sessions": active,
            "idle_sessions": idle,
            "retained_blocks": retained_blocks,
        }

    def _get_by_req(self, req_context: ReqContext) -> SessionKVState | None:
        session_id = self._req_to_session.get(req_context.req_id)
        if session_id is None:
            session_id = get_session_id(req_context)
        return self._sessions.get(session_id)

    def _delete_state(self, state: SessionKVState) -> None:
        state.status = LifecycleStatus.DELETED
        self._sessions.pop(state.session_id, None)
        for req_id in state.active_req_ids | state.retained_req_ids:
            self._req_to_session.pop(req_id, None)

    def _delete_secondary_blocks(
        self, state: SessionKVState, secondary_tiers: Iterable[object]
    ) -> None:
        for tier in secondary_tiers:
            file_mapper = getattr(tier, "file_mapper", None)
            if file_mapper is None:
                continue
            for key in state.block_keys:
                if self._is_key_referenced_by_other_session(key, state.session_id):
                    continue
                path = file_mapper.get_file_name(key)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    continue
                except OSError:
                    logger.warning(
                        "Failed to delete expired lifecycle KV block %s",
                        path,
                        exc_info=True,
                    )

    def _is_key_referenced_by_other_session(
        self, key: OffloadKey, session_id: str
    ) -> bool:
        for other in self._sessions.values():
            if other.session_id == session_id:
                continue
            if other.status is LifecycleStatus.DELETED:
                continue
            if key in other.block_keys:
                return True
        return False
