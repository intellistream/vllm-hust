# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded prompt-free receipts for request-priority scheduling."""

from __future__ import annotations

import copy
import os
import time
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any


class PrioritySchedulingObserver:
    """Record scheduling decisions only for explicitly scoped request IDs."""

    def __init__(self, *, enabled: bool, capacity: int, request_prefix: str) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("priority scheduling observer enabled must be boolean")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("priority scheduling observer capacity must be an integer")
        if not 128 <= capacity <= 65536:
            raise ValueError(
                "priority scheduling observer capacity must be in [128, 65536]"
            )
        normalized_prefix = str(request_prefix).strip()
        if not normalized_prefix:
            raise ValueError("priority scheduling request prefix must not be empty")
        self.enabled = enabled
        self.capacity = capacity
        self.request_prefix = normalized_prefix
        self.sequence = 0
        self._receipts: deque[dict[str, Any]] = deque(maxlen=capacity)

    @classmethod
    def from_environment(cls) -> PrioritySchedulingObserver:
        raw_enabled = os.environ.get("QBI_PRIORITY_SCHEDULING_OBSERVER", "0")
        normalized_enabled = raw_enabled.strip().lower()
        false_values = {"0", "false", "no", "off"}
        true_values = {"1", "true", "yes", "on"}
        if normalized_enabled not in false_values | true_values:
            raise ValueError("QBI_PRIORITY_SCHEDULING_OBSERVER must be a boolean flag")
        capacity_text = os.environ.get(
            "QBI_PRIORITY_SCHEDULING_OBSERVER_CAPACITY", "8192"
        ).strip()
        if not capacity_text.isdecimal():
            raise ValueError(
                "QBI_PRIORITY_SCHEDULING_OBSERVER_CAPACITY must be an integer"
            )
        return cls(
            enabled=normalized_enabled in true_values,
            capacity=int(capacity_text),
            request_prefix=os.environ.get(
                "QBI_PRIORITY_SCHEDULING_REQUEST_PREFIX",
                "chatcmpl-anise-priority-",
            ),
        )

    def record(
        self,
        *,
        policy: str,
        scheduled_new_request_ids: Iterable[str],
        request_metadata: Mapping[str, Mapping[str, int | float]],
        formed_request_ids: Iterable[str],
        config_epoch: int,
    ) -> None:
        if not self.enabled:
            return
        scheduled_ids = [str(request_id) for request_id in scheduled_new_request_ids]
        matched_ids = [
            request_id
            for request_id in scheduled_ids
            if request_id.startswith(self.request_prefix)
        ]
        if not matched_ids:
            return
        if policy != "priority":
            raise RuntimeError(
                "priority observer saw a scoped request on a non-priority queue"
            )
        missing = [
            request_id
            for request_id in scheduled_ids
            if request_id not in request_metadata
        ]
        if missing:
            raise RuntimeError(f"priority scheduling metadata missing: {missing}")
        self.sequence += 1
        self._receipts.append(
            {
                "schema": "qbi.request-priority-scheduler-scope.v1",
                "sequence": self.sequence,
                "timestamp_ns": time.time_ns(),
                "config_epoch": int(config_epoch),
                "queue_policy": policy,
                "scheduled_new_request_ids": scheduled_ids,
                "matched_scheduled_request_ids": matched_ids,
                "formed_request_ids": sorted(
                    str(value) for value in formed_request_ids
                ),
                "priorities": {
                    request_id: int(request_metadata[request_id]["priority"])
                    for request_id in scheduled_ids
                },
                "arrival_times": {
                    request_id: float(request_metadata[request_id]["arrival_time"])
                    for request_id in scheduled_ids
                },
                "request_count": len(scheduled_ids),
                "matched_request_count": len(matched_ids),
            }
        )

    def state(self, after_sequence: int = 0) -> dict[str, Any]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ValueError("priority scheduling sequence cursor must be an integer")
        if after_sequence < 0:
            raise ValueError("priority scheduling sequence cursor must be non-negative")
        receipts = [
            copy.deepcopy(receipt)
            for receipt in self._receipts
            if int(receipt["sequence"]) > after_sequence
        ]
        oldest = int(self._receipts[0]["sequence"]) if self._receipts else 0
        return {
            "schema": "qbi.request-priority-scheduler-state.v1",
            "observer_enabled": self.enabled,
            "request_prefix": self.request_prefix,
            "capacity": self.capacity,
            "oldest_available_sequence": oldest,
            "latest_sequence": self.sequence,
            "receipts": receipts,
        }


__all__ = ["PrioritySchedulingObserver"]
