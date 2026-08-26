"""Bounded prompt-free scheduler evidence for ANISE native recapture."""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Mapping
from typing import Any


class NativeRecaptureScopeObserver:
    """Retain only scheduling steps joined to formal recapture request IDs."""

    def __init__(self, *, enabled: bool, capacity: int, request_prefix: str) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("native recapture observer enabled must be boolean")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("native recapture observer capacity must be an integer")
        if not 128 <= capacity <= 65536:
            raise ValueError(
                "native recapture observer capacity must be in [128, 65536]"
            )
        normalized_prefix = str(request_prefix).strip()
        if not normalized_prefix:
            raise ValueError("native recapture request prefix must not be empty")
        self.enabled = enabled
        self.capacity = capacity
        self.request_prefix = normalized_prefix
        self.sequence = 0
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)

    @classmethod
    def from_environment(cls) -> NativeRecaptureScopeObserver:
        raw_enabled = os.environ.get("QBI_NATIVE_RECAPTURE_OBSERVER", "0")
        normalized_enabled = raw_enabled.strip().lower()
        false_values = {"0", "false", "no", "off"}
        true_values = {"1", "true", "yes", "on"}
        if normalized_enabled not in false_values | true_values:
            raise ValueError("QBI_NATIVE_RECAPTURE_OBSERVER must be a boolean flag")
        capacity_text = os.environ.get(
            "QBI_NATIVE_RECAPTURE_OBSERVER_CAPACITY", "8192"
        ).strip()
        if not capacity_text.isdecimal():
            raise ValueError(
                "QBI_NATIVE_RECAPTURE_OBSERVER_CAPACITY must be an integer"
            )
        return cls(
            enabled=normalized_enabled in true_values,
            capacity=int(capacity_text),
            request_prefix=os.environ.get(
                "QBI_NATIVE_RECAPTURE_REQUEST_PREFIX",
                "chatcmpl-anise-native-recapture-",
            ),
        )

    def record(
        self,
        *,
        num_scheduled_tokens: Mapping[str, int],
        total_num_scheduled_tokens: int,
        config_epoch: int,
        max_num_running_reqs: int,
        max_num_scheduled_tokens: int,
    ) -> None:
        if not self.enabled:
            return
        request_ids = sorted(str(request_id) for request_id in num_scheduled_tokens)
        matched_request_ids = [
            request_id
            for request_id in request_ids
            if request_id.startswith(self.request_prefix)
        ]
        if not matched_request_ids:
            return
        self.sequence += 1
        self._receipts.append(
            {
                "schema": "qbi.native-recapture-scheduler-scope.v1",
                "sequence": self.sequence,
                "timestamp_ns": time.time_ns(),
                "config_epoch": int(config_epoch),
                "request_ids": request_ids,
                "matched_request_ids": matched_request_ids,
                "request_count": len(request_ids),
                "matched_request_count": len(matched_request_ids),
                "formed_batch": len(request_ids) > 1,
                "num_scheduled_tokens": {
                    request_id: int(num_scheduled_tokens[request_id])
                    for request_id in request_ids
                },
                "total_num_scheduled_tokens": int(total_num_scheduled_tokens),
                "max_num_running_reqs": int(max_num_running_reqs),
                "max_num_scheduled_tokens": int(max_num_scheduled_tokens),
            }
        )

    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("native recapture sequence cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("native recapture sequence cursor must be non-negative")
        receipts = [
            dict(receipt)
            for receipt in self._receipts
            if int(receipt["sequence"]) > after_sequence
        ]
        oldest = int(self._receipts[0]["sequence"]) if self._receipts else 0
        return {
            "schema": "qbi.native-recapture-scheduler-state.v1",
            "observer_enabled": self.enabled,
            "request_prefix": self.request_prefix,
            "capacity": self.capacity,
            "oldest_available_sequence": oldest,
            "latest_sequence": self.sequence,
            "receipts": receipts,
        }


__all__ = ["NativeRecaptureScopeObserver"]
