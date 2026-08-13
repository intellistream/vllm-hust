# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure control policy for request-owned decode and prefill windows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from vllm.v1.core.sched.ownership import OwnerLeaseKey


class OwnerWindowPolicyPhase(Enum):
    DECODE = auto()
    PREFILL = auto()


@dataclass(frozen=True)
class OwnerPrefillCandidate:
    """One FCFS-ordered prefill candidate visible at an ack boundary."""

    key: OwnerLeaseKey
    owner_id: int
    wait_steps: int


@dataclass(frozen=True)
class OwnerWindowReadiness:
    """Scheduler-visible readiness facts at one acknowledged boundary."""

    hot_decode_by_owner: tuple[int, ...]
    restoring_by_owner: tuple[int, ...]
    host_restorable_by_owner: tuple[int, ...]
    prefill_candidates: tuple[OwnerPrefillCandidate, ...] = ()


@dataclass(frozen=True)
class OwnerWindowPolicyConfig:
    world_size: int
    decode_observation_steps: int
    hot_low_watermark: int
    hot_high_watermark: int
    prefill_invocation_budget: int
    prefill_max_wait_steps: int

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.decode_observation_steps <= 0:
            raise ValueError("decode_observation_steps must be positive")
        if not 0 <= self.hot_low_watermark < self.hot_high_watermark:
            raise ValueError("HOT watermarks must satisfy 0 <= low < high")
        if self.prefill_invocation_budget <= 0:
            raise ValueError("prefill_invocation_budget must be positive")
        if self.prefill_max_wait_steps <= 0:
            raise ValueError("prefill_max_wait_steps must be positive")


@dataclass(frozen=True)
class OwnerWindowPolicyDecision:
    phase: OwnerWindowPolicyPhase
    prefill_wave: tuple[OwnerLeaseKey, ...]
    reason: str


class OwnerWindowPolicy:
    """Receipt-driven dual-watermark controller.

    The controller owns only phase-switch policy. Admission, execution, and
    lifecycle authority stay with the scheduler. Its frozen prefill snapshot
    cannot be extended by arrivals observed after the wave opens.
    """

    def __init__(self, config: OwnerWindowPolicyConfig) -> None:
        self.config = config
        self.phase = OwnerWindowPolicyPhase.DECODE
        self.decode_steps = 0
        self.prefill_invocations = 0
        self.prefill_wave: tuple[OwnerLeaseKey, ...] = ()

    def decision(self, reason: str = "unchanged") -> OwnerWindowPolicyDecision:
        return OwnerWindowPolicyDecision(
            phase=self.phase,
            prefill_wave=self.prefill_wave,
            reason=reason,
        )

    def start_prefill(
        self, wave: tuple[OwnerLeaseKey, ...], *, reason: str
    ) -> OwnerWindowPolicyDecision:
        """Open one scheduler-selected frozen prefill wave."""
        if self.phase is not OwnerWindowPolicyPhase.DECODE or not wave:
            raise RuntimeError("prefill wave must start from decode with members")
        self.phase = OwnerWindowPolicyPhase.PREFILL
        self.prefill_wave = wave
        self.prefill_invocations = 0
        self.decode_steps = 0
        return self.decision(reason)

    def reset_decode_observation(self) -> None:
        """Forget observations made by a cohort that was dissolved."""
        if self.phase is OwnerWindowPolicyPhase.DECODE:
            self.decode_steps = 0

    def cancel_prefill(self) -> None:
        """Close a wave whose frozen members became unavailable."""
        if self.phase is OwnerWindowPolicyPhase.PREFILL:
            self._close_prefill("prefill-snapshot-unavailable")

    def ack_step(
        self,
        phase: OwnerWindowPolicyPhase,
        readiness: OwnerWindowReadiness,
        *,
        positive_tokens: bool,
    ) -> OwnerWindowPolicyDecision:
        """Observe readiness only after an execution step is acknowledged."""
        self._validate_readiness(readiness)
        if not positive_tokens:
            return self.decision("command-only")
        if phase is not self.phase:
            raise RuntimeError(
                "acknowledged owner-window phase does not match controller state"
            )
        if phase is OwnerWindowPolicyPhase.DECODE:
            return self._ack_decode(readiness)
        return self._ack_prefill(readiness)

    def _ack_decode(self, readiness: OwnerWindowReadiness) -> OwnerWindowPolicyDecision:
        self.decode_steps += 1
        if self.decode_steps < self.config.decode_observation_steps:
            return self.decision("decode-observation-budget")

        self.decode_steps = 0
        wave, reason = self._select_prefill_wave(readiness)
        if not wave:
            return self.decision("decode-reservoir-sufficient")

        return self.start_prefill(wave, reason=reason)

    def _ack_prefill(
        self, readiness: OwnerWindowReadiness
    ) -> OwnerWindowPolicyDecision:
        self.prefill_invocations += 1
        queued = {candidate.key for candidate in readiness.prefill_candidates}
        self.prefill_wave = tuple(key for key in self.prefill_wave if key in queued)

        if not self.prefill_wave:
            return self._close_prefill("prefill-snapshot-complete")
        if self.prefill_invocations >= self.config.prefill_invocation_budget:
            return self._close_prefill("prefill-invocation-budget")
        if all(
            hot + restoring >= self.config.hot_high_watermark
            for hot, restoring in zip(
                readiness.hot_decode_by_owner,
                readiness.restoring_by_owner,
            )
        ):
            return self._close_prefill("hot-high-watermark")
        return self.decision("prefill-snapshot-continues")

    def _close_prefill(self, reason: str) -> OwnerWindowPolicyDecision:
        self.phase = OwnerWindowPolicyPhase.DECODE
        self.prefill_wave = ()
        self.prefill_invocations = 0
        self.decode_steps = 0
        return self.decision(reason)

    def _select_prefill_wave(
        self, readiness: OwnerWindowReadiness
    ) -> tuple[tuple[OwnerLeaseKey, ...], str]:
        selected: list[OwnerLeaseKey] = []
        remaining_by_owner = [0] * self.config.world_size
        for owner_id, (hot, restoring, host_restorable) in enumerate(
            zip(
                readiness.hot_decode_by_owner,
                readiness.restoring_by_owner,
                readiness.host_restorable_by_owner,
            )
        ):
            future_hot = hot + restoring + host_restorable
            if future_hot <= self.config.hot_low_watermark:
                remaining_by_owner[owner_id] = max(
                    0, self.config.hot_high_watermark - future_hot
                )

        guardrail_due = False
        guardrail_owner_selected = [False] * self.config.world_size
        for candidate in readiness.prefill_candidates:
            owner_id = candidate.owner_id
            due = (
                candidate.wait_steps >= self.config.prefill_max_wait_steps
                and not guardrail_owner_selected[owner_id]
            )
            if remaining_by_owner[owner_id] > 0 or due:
                selected.append(candidate.key)
                if remaining_by_owner[owner_id] > 0:
                    remaining_by_owner[owner_id] -= 1
                if due:
                    guardrail_due = True
                    guardrail_owner_selected[owner_id] = True

        reason = "prefill-max-wait" if guardrail_due else "hot-low-watermark"
        return tuple(selected), reason

    def _validate_readiness(self, readiness: OwnerWindowReadiness) -> None:
        counts = (
            readiness.hot_decode_by_owner,
            readiness.restoring_by_owner,
            readiness.host_restorable_by_owner,
        )
        if any(len(values) != self.config.world_size for values in counts):
            raise ValueError("readiness owner counts must match world_size")
        if any(value < 0 for values in counts for value in values):
            raise ValueError("readiness owner counts cannot be negative")
        for candidate in readiness.prefill_candidates:
            if not 0 <= candidate.owner_id < self.config.world_size:
                raise ValueError("prefill candidate owner is out of range")
            if candidate.wait_steps < 0:
                raise ValueError("prefill candidate wait cannot be negative")
