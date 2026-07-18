# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed request controls and executed counters for native KV reuse.

This module does not implement a second prefix cache. It constrains vLLM's
native block-hash lookup and records reuse only after native blocks become
owned by a request through ``KVCacheManager.allocate_slots``.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

PREFIX_SHARING_EXTRA_ARG = "kv_prefix_sharing"
PREFIX_SHARING_FLAT_PREFIX = "kv_prefix_sharing_"
PREFIX_SHARING_FALLBACK_REASONS = (
    "invalid_control_payload",
    "admission_denied",
    "read_not_admitted",
    "native_cache_miss",
    "reuse_wait_threshold_unmet",
    "allocation_capacity_rejected",
    "allocation_exception_rollback",
)


@dataclass(frozen=True)
class PrefixSharingPolicy:
    identity: str | None
    share_domain: str | None
    isolation: tuple[tuple[str, str], ...]
    read_admitted: bool
    write_admitted: bool
    min_reuse_tokens: int
    max_reuse_tokens: int | None
    fallback_reason: str | None = None

    @property
    def admitted(self) -> bool:
        return self.read_admitted or self.write_admitted

    def derive_cache_salt(self, original_salt: str | None, request_id: str) -> str:
        """Namespace native token hashes without trusting identity as content."""
        payload = {
            "version": 1,
            "identity": self.identity,
            "share_domain": self.share_domain,
            "isolation": self.isolation,
            "original_cache_salt": original_salt,
        }
        if not self.admitted:
            # A denied request gets a request-private namespace as a second
            # fail-closed barrier even though reads and writes are also gated.
            payload["denied_request_id"] = request_id
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "kv-prefix-sharing-v1:" + hashlib.sha256(encoded).hexdigest()


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _parse_non_negative_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _parse_flag(value: Any, *, default: bool) -> bool | None:
    if value is None:
        return default
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    return None


def parse_prefix_sharing_policy(request: Any) -> PrefixSharingPolicy | None:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    raw = extra_args.get(PREFIX_SHARING_EXTRA_ARG)
    if raw is None and any(
        str(key).startswith(PREFIX_SHARING_FLAT_PREFIX) for key in extra_args
    ):
        raw = {
            "identity": extra_args.get(f"{PREFIX_SHARING_FLAT_PREFIX}identity"),
            "share_domain": extra_args.get(f"{PREFIX_SHARING_FLAT_PREFIX}share_domain"),
            "isolation": extra_args.get(f"{PREFIX_SHARING_FLAT_PREFIX}isolation"),
            "admit": extra_args.get(f"{PREFIX_SHARING_FLAT_PREFIX}admit"),
            "admit_read": extra_args.get(f"{PREFIX_SHARING_FLAT_PREFIX}admit_read"),
            "admit_write": extra_args.get(f"{PREFIX_SHARING_FLAT_PREFIX}admit_write"),
            "min_reuse_tokens": extra_args.get(
                f"{PREFIX_SHARING_FLAT_PREFIX}min_reuse_tokens"
            ),
            "max_reuse_tokens": extra_args.get(
                f"{PREFIX_SHARING_FLAT_PREFIX}max_reuse_tokens"
            ),
        }
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return PrefixSharingPolicy(
            None, None, (), False, False, 0, None, "invalid_control_payload"
        )

    identity = _non_empty_string(raw.get("identity"))
    share_domain = _non_empty_string(raw.get("share_domain"))
    isolation_raw = raw.get("isolation", {})
    if isinstance(isolation_raw, str):
        try:
            isolation_raw = json.loads(isolation_raw)
        except json.JSONDecodeError:
            isolation_raw = None
    if not isinstance(isolation_raw, Mapping):
        isolation_raw = None
    isolation: tuple[tuple[str, str], ...] = ()
    if isolation_raw is not None:
        isolation_values: list[tuple[str, str]] = []
        for key, value in isolation_raw.items():
            key_string = _non_empty_string(key)
            value_string = _non_empty_string(value)
            if key_string is None or value_string is None:
                isolation_raw = None
                break
            isolation_values.append((key_string, value_string))
        isolation = tuple(sorted(isolation_values))

    min_reuse_tokens = _parse_non_negative_int(raw.get("min_reuse_tokens"), default=0)
    max_reuse_tokens = _parse_non_negative_int(
        raw.get("max_reuse_tokens"), default=None
    )
    valid = (
        identity is not None
        and share_domain is not None
        and isolation_raw is not None
        and min_reuse_tokens is not None
        and (raw.get("max_reuse_tokens") is None or max_reuse_tokens is not None)
    )
    if not valid:
        return PrefixSharingPolicy(
            identity,
            share_domain,
            isolation,
            False,
            False,
            0,
            None,
            "invalid_control_payload",
        )

    admitted = _parse_flag(raw.get("admit"), default=True)
    read_flag = _parse_flag(raw.get("admit_read"), default=True)
    write_flag = _parse_flag(raw.get("admit_write"), default=True)
    if admitted is None or read_flag is None or write_flag is None:
        return PrefixSharingPolicy(
            identity,
            share_domain,
            isolation,
            False,
            False,
            0,
            None,
            "invalid_control_payload",
        )
    read_admitted = admitted and read_flag
    write_admitted = admitted and write_flag
    fallback_reason = None if admitted else "admission_denied"
    return PrefixSharingPolicy(
        identity=identity,
        share_domain=share_domain,
        isolation=isolation,
        read_admitted=read_admitted,
        write_admitted=write_admitted,
        min_reuse_tokens=min_reuse_tokens,
        max_reuse_tokens=max_reuse_tokens,
        fallback_reason=fallback_reason,
    )


@dataclass
class PrefixSharingRuntimeStats:
    configured_requests: int = 0
    admitted_lookups: int = 0
    denied_lookups: int = 0
    native_lookup_hit_requests: int = 0
    native_lookup_hit_tokens: int = 0
    native_published_requests: int = 0
    native_published_blocks: int = 0
    native_attached_requests: int = 0
    native_attached_blocks: int = 0
    consumed_kv_tokens: int = 0
    realized_reuse_requests: int = 0
    avoided_prefill_tokens: int = 0
    allocation_failures: int = 0
    rollback_count: int = 0
    ownership_releases: int = 0
    fallback_reason_histogram: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fallback_reason_histogram"] = dict(
            sorted(self.fallback_reason_histogram.items())
        )
        result["counter_provenance"] = "serving_runtime_kv_cache"
        result["counter_semantics"] = "realized_runtime_reuse"
        return result


class PrefixSharingRuntimeTracker:
    def __init__(self) -> None:
        self.stats = PrefixSharingRuntimeStats()
        self._seen_requests: set[str] = set()
        self._pending_hits: dict[str, tuple[int, int]] = {}
        self._owned_reuse_requests: set[str] = set()
        self._published_requests: set[str] = set()

    def observe_policy(self, request: Any, policy: PrefixSharingPolicy) -> None:
        if request.request_id not in self._seen_requests:
            self._seen_requests.add(request.request_id)
            self.stats.configured_requests += 1
        if not policy.read_admitted:
            self.stats.denied_lookups += 1
            self.stats.fallback_reason_histogram[
                policy.fallback_reason or "read_not_admitted"
            ] += 1

    def observe_lookup(
        self,
        request_id: str,
        *,
        num_tokens: int,
        num_blocks: int,
        fallback_reason: str | None = None,
    ) -> None:
        self.stats.admitted_lookups += 1
        if fallback_reason is not None:
            self.stats.fallback_reason_histogram[fallback_reason] += 1
            self._pending_hits.pop(request_id, None)
            return
        if num_tokens <= 0:
            self.stats.fallback_reason_histogram["native_cache_miss"] += 1
            self._pending_hits.pop(request_id, None)
            return
        self.stats.native_lookup_hit_requests += 1
        self.stats.native_lookup_hit_tokens += num_tokens
        self._pending_hits[request_id] = (num_tokens, num_blocks)

    def observe_attach(self, request_id: str) -> None:
        pending = self._pending_hits.pop(request_id, None)
        if pending is None:
            return
        num_tokens, num_blocks = pending
        self.stats.native_attached_requests += 1
        self.stats.native_attached_blocks += num_blocks
        self.stats.consumed_kv_tokens += num_tokens
        self.stats.realized_reuse_requests += 1
        self.stats.avoided_prefill_tokens += num_tokens
        self._owned_reuse_requests.add(request_id)

    def observe_publish(self, request_id: str, num_blocks: int) -> None:
        if num_blocks <= 0:
            return
        if request_id not in self._published_requests:
            self._published_requests.add(request_id)
            self.stats.native_published_requests += 1
        self.stats.native_published_blocks += num_blocks

    def observe_allocation_failure(self, request_id: str, reason: str) -> None:
        if self._pending_hits.pop(request_id, None) is not None:
            self.stats.allocation_failures += 1
            self.stats.fallback_reason_histogram[reason] += 1

    def observe_rollback(self, request_id: str, reason: str) -> None:
        self._pending_hits.pop(request_id, None)
        self._owned_reuse_requests.discard(request_id)
        self.stats.rollback_count += 1
        self.stats.fallback_reason_histogram[reason] += 1

    def observe_release(self, request_id: str) -> None:
        self._pending_hits.pop(request_id, None)
        self._seen_requests.discard(request_id)
        self._published_requests.discard(request_id)
        if request_id in self._owned_reuse_requests:
            self._owned_reuse_requests.remove(request_id)
            self.stats.ownership_releases += 1

    def snapshot(self, *, reset: bool = False) -> dict[str, Any]:
        result = self.stats.to_dict()
        if reset:
            self.stats = PrefixSharingRuntimeStats()
        return result
