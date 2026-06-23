# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Residency state tracking for hierarchical KV-cache shadow metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from vllm.v1.kv_offload.base import (
    OffloadKey,
    get_offload_block_hash,
    get_offload_group_idx,
)


class ResidencyState(str, Enum):
    READY_DEVICE = "READY_DEVICE"
    READY_HOST = "READY_HOST"
    LOADING_H2D = "LOADING_H2D"
    STORING_D2H = "STORING_D2H"
    ABSENT = "ABSENT"


@dataclass(frozen=True, slots=True)
class BlockRef:
    block_hash: bytes
    group_idx: int

    @classmethod
    def from_offload_key(cls, key: OffloadKey) -> BlockRef:
        return cls(
            block_hash=bytes(get_offload_block_hash(key)),
            group_idx=get_offload_group_idx(key),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "block_hash": self.block_hash.hex(),
            "group_idx": self.group_idx,
        }


class ResidencyTracker:
    """Best-effort shadow state for block residency.

    The tracker is intentionally advisory. Scheduler-visible local prefix hits
    override it at record time because device residency may change outside the
    offloading connector.
    """

    def __init__(self) -> None:
        self._states: dict[BlockRef, ResidencyState] = {}

    def get(self, ref: BlockRef) -> ResidencyState:
        return self._states.get(ref, ResidencyState.ABSENT)

    def mark(self, refs: Iterable[BlockRef], state: ResidencyState) -> None:
        for ref in refs:
            if state == ResidencyState.ABSENT:
                self._states.pop(ref, None)
            else:
                self._states[ref] = state

    def mark_keys(self, keys: Iterable[OffloadKey], state: ResidencyState) -> None:
        self.mark((BlockRef.from_offload_key(key) for key in keys), state)

    def snapshot_counts(
        self,
        refs: Iterable[BlockRef],
    ) -> dict[ResidencyState, int]:
        counts = {state: 0 for state in ResidencyState}
        for ref in refs:
            counts[self.get(ref)] += 1
        return counts
