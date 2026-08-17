# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-ID-free evidence and non-acting policy for owner-local prefixes."""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class OwnerPrefixCandidateObservation:
    """One capacity-feasible owner considered for prefix placement."""

    owner_id: int
    projected_free: int
    fresh_demand: int
    live_leases: int
    hinted_tokens: int
    affinity_eligible: bool


@dataclass(frozen=True, slots=True)
class OwnerPrefixAdmissionObservation:
    """Block-ID-free placement evidence emitted before RESERVE."""

    request_id: str
    prefix_digest: str | None
    prefix_tokens: int
    selected_owner: int
    selected_hinted_tokens: int
    reason: Literal["affinity_hit", "bounded_spill", "cold_balance"]
    candidates: tuple[OwnerPrefixCandidateObservation, ...]


@dataclass(frozen=True, slots=True)
class OwnerPrefixReserveObservation:
    """Exact worker outcome for one matching RESERVE."""

    request_id: str
    owner_epoch: int
    owner_id: int
    prefix_digest: str | None
    prefix_tokens: int
    hinted_tokens: int
    accepted: bool
    exact_hit_tokens: int | None
    outcome: Literal[
        "exact_hit",
        "cold_miss",
        "stale_hint_miss",
        "rejected_reserve",
    ]


@dataclass(frozen=True, slots=True)
class OwnerPrefixPublicationObservation:
    """A worker-confirmed request extended an immutable prefix boundary."""

    request_id: str
    owner_id: int
    prefix_digest: str
    previous_tokens: int
    published_tokens: int
    exact_reserve_hit_tokens: int | None
    directory_hint_was_new: bool


OwnerPrefixObservation = (
    OwnerPrefixAdmissionObservation
    | OwnerPrefixReserveObservation
    | OwnerPrefixPublicationObservation
)
OwnerPrefixObservationSink = Callable[[OwnerPrefixObservation], None]


@dataclass(frozen=True, slots=True)
class OwnerPrefixShadowDecision:
    """Non-acting replica recommendation for an observed prefix."""

    action: Literal["keep", "expand", "refuse"]
    reason: Literal[
        "replica_set_sufficient",
        "expand_with_slack",
        "insufficient_execution_slack",
        "insufficient_lead",
        "insufficient_capacity",
    ]
    current_replicas: int
    desired_replicas: int
    target_owner: int | None


class RequestOwnedPrefixReplicaShadow:
    """Pure, non-acting replica estimator for trace and live shadow replay."""

    def __init__(self, world_size: int, rows_per_owner: int) -> None:
        if world_size <= 0 or rows_per_owner <= 0:
            raise ValueError("shadow topology values must be positive")
        self.world_size = world_size
        self.rows_per_owner = rows_per_owner

    def evaluate(
        self,
        *,
        predicted_runnable_seats: int,
        hinted_owners: frozenset[int],
        candidates: Sequence[OwnerPrefixCandidateObservation],
        required_free_blocks: int,
        lead_steps: int,
        materialization_steps: int,
        execution_slack: bool,
    ) -> OwnerPrefixShadowDecision:
        """Return a recommendation without mutating routing or cache state."""

        scalar_values = (
            predicted_runnable_seats,
            required_free_blocks,
            lead_steps,
            materialization_steps,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in scalar_values
        ):
            raise TypeError("shadow demand and cost values must be non-bool ints")
        if predicted_runnable_seats <= 0:
            raise ValueError("predicted_runnable_seats must be positive")
        if any(value < 0 for value in scalar_values[1:]):
            raise ValueError("shadow cost values must be nonnegative")
        if not isinstance(execution_slack, bool):
            raise TypeError("execution_slack must be a bool")
        if any(
            isinstance(owner, bool)
            or not isinstance(owner, int)
            or not 0 <= owner < self.world_size
            for owner in hinted_owners
        ):
            raise ValueError("hinted owner is outside the shadow owner world")
        candidate_owners = [candidate.owner_id for candidate in candidates]
        if len(candidate_owners) != len(set(candidate_owners)):
            raise ValueError("shadow candidates must have unique owners")
        if any(
            isinstance(owner, bool)
            or not isinstance(owner, int)
            or not 0 <= owner < self.world_size
            for owner in candidate_owners
        ):
            raise ValueError("candidate owner is outside the shadow owner world")

        desired = min(
            self.world_size,
            (predicted_runnable_seats + self.rows_per_owner - 1) // self.rows_per_owner,
        )
        current = len(hinted_owners)
        if current >= desired:
            return OwnerPrefixShadowDecision(
                "keep",
                "replica_set_sufficient",
                current,
                desired,
                None,
            )
        if not execution_slack:
            return OwnerPrefixShadowDecision(
                "refuse",
                "insufficient_execution_slack",
                current,
                desired,
                None,
            )
        if lead_steps < materialization_steps:
            return OwnerPrefixShadowDecision(
                "refuse",
                "insufficient_lead",
                current,
                desired,
                None,
            )

        feasible = tuple(
            candidate
            for candidate in candidates
            if candidate.owner_id not in hinted_owners
            and candidate.projected_free >= required_free_blocks
        )
        if not feasible:
            return OwnerPrefixShadowDecision(
                "refuse",
                "insufficient_capacity",
                current,
                desired,
                None,
            )
        target = min(
            feasible,
            key=lambda candidate: (
                -candidate.projected_free,
                candidate.live_leases,
                candidate.owner_id,
            ),
        )
        return OwnerPrefixShadowDecision(
            "expand",
            "expand_with_slack",
            current,
            desired,
            target.owner_id,
        )


def make_prefix_observation_identity(
    block_hashes: Sequence[BlockHash],
    num_prompt_tokens: int,
    scheduler_block_size: int,
    hash_block_size: int,
) -> tuple[str | None, int]:
    """Return an anonymous digest for the longest reusable boundary."""

    max_tokens = min(
        max(num_prompt_tokens - 1, 0),
        len(block_hashes) * hash_block_size,
    )
    prefix_tokens = max_tokens // scheduler_block_size * scheduler_block_size
    if prefix_tokens == 0:
        return None, 0
    terminal_hash_index = prefix_tokens // hash_block_size - 1
    terminal_hash = bytes(block_hashes[terminal_hash_index])
    digest = hashlib.sha256(
        b"stateharbor-owner-prefix-observation-v1\0" + terminal_hash
    ).hexdigest()
    return digest, prefix_tokens
