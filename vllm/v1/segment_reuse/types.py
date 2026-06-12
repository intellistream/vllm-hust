"""Stitch metadata types for segment reuse.

These dataclasses carry the envelope||body decomposition information
through the vLLM serving pipeline:
- EnvelopeBodySpec: attached to Request, describes the E||B boundary
- BodyBlockEntry: stored in BodyBlockRegistry, tracks pinned body blocks
- StitchResult: attached to Request after scheduler decision
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StitchState(str, Enum):
    """Lifecycle states for a stitch-eligible request."""

    # Request received but no boundary metadata found.
    NO_BOUNDARY = "no-boundary"
    # Boundary resolved from request metadata.
    BOUNDARY_RESOLVED = "boundary-resolved"
    # Body hash computed, looking up in registry.
    BODY_LOOKUP = "body-lookup"
    # Body found in registry, stitch committed.
    COMMITTED = "committed"
    # Body found but prefix cache already covers it.
    PREFIX_COVERED = "prefix-covered"
    # Demoted to normal prefix path (e.g., logprob request).
    DEMOTED = "demoted"
    # Seed request: body will be registered after prefill.
    SEED = "seed"


@dataclass(frozen=True)
class EnvelopeBodySpec:
    """Envelope||body decomposition for a single request.

    The prompt is structured as: [envelope tokens] || [body tokens]
    where the envelope is request-specific (dynamic) and the body
    is shared across requests (stable system prompt, persona, tools).

    Attributes:
        envelope_token_count: Number of envelope tokens (|E|).
        body_token_ids: The body token IDs as a tuple for hashing.
        body_hash: SHA-256 digest of the body token sequence.
        boundary_source: How the boundary was determined
            ("client" = from API metadata, "auto-detect" = heuristic).
        total_token_count: Total prompt tokens (|E| + |B|).
    """

    envelope_token_count: int
    body_token_ids: tuple[int, ...]
    body_hash: bytes
    boundary_source: str
    total_token_count: int

    @property
    def body_token_count(self) -> int:
        return len(self.body_token_ids)

    @classmethod
    def from_token_ids(
        cls,
        all_token_ids: list[int] | tuple[int, ...],
        envelope_token_count: int,
        boundary_source: str = "client",
    ) -> "EnvelopeBodySpec":
        """Build an EnvelopeBodySpec from the full token sequence.

        Args:
            all_token_ids: The full prompt token IDs (E || B).
            envelope_token_count: Number of envelope tokens at the front.
            boundary_source: Source of the boundary information.

        Returns:
            An EnvelopeBodySpec with the body hash computed.
        """
        body_ids = tuple(all_token_ids[envelope_token_count:])
        body_hash = hashlib.sha256(
            b"".join(t.to_bytes(4, "big") for t in body_ids)
        ).digest()
        return cls(
            envelope_token_count=envelope_token_count,
            body_token_ids=body_ids,
            body_hash=body_hash,
            boundary_source=boundary_source,
            total_token_count=len(all_token_ids),
        )

    @classmethod
    def from_extra_body(cls, extra_body: dict[str, Any] | None) -> int | None:
        """Extract envelope_token_count from OpenAI extra_body dict.

        Returns None if the metadata is not present.
        """
        if extra_body is None:
            return None
        return extra_body.get("envelope_token_count")


@dataclass
class BodyBlockEntry:
    """A registered body in the BodyBlockRegistry.

    Tracks the pinned KV blocks for a body token sequence.
    Multiple requests can borrow the same body's KV blocks
    simultaneously (ref_count tracks active borrowers).

    Attributes:
        body_hash: SHA-256 digest identifying this body.
        block_ids: Physical KV block IDs pinned for this body.
        token_count: Number of tokens this body occupies.
        block_size: The KV cache block size in tokens.
        ref_count: Number of active borrowers.
        model_key: Model identifier for cross-model isolation.
        created_at: Monotonic timestamp of registration.
    """

    body_hash: bytes
    block_ids: list[int]
    token_count: int
    block_size: int
    ref_count: int = 0
    model_key: str = "default"
    created_at: float = 0.0

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)


@dataclass
class StitchResult:
    """Result of the stitch decision for a request.

    After the scheduler processes a stitch-eligible request, this
    records what happened: how many tokens were reused via stitch,
    which body blocks were borrowed, and the final state.

    Attributes:
        state: The lifecycle state of this stitch attempt.
        envelope_blocks: Number of fresh KV blocks allocated for E.
        reused_body_blocks: Number of body blocks borrowed from registry.
        total_reused_tokens: Total tokens reused (prefix + body).
        body_hash: Hash of the matched body (None if no match).
        body_block_ids: Physical block IDs of the borrowed body.
        demotion_reason: If demoted, why.
    """

    state: StitchState
    envelope_blocks: int = 0
    reused_body_blocks: int = 0
    total_reused_tokens: int = 0
    body_hash: bytes | None = None
    body_block_ids: list[int] = field(default_factory=list)
    demotion_reason: str | None = None

    @property
    def is_committed(self) -> bool:
        return self.state == StitchState.COMMITTED

    @property
    def body_reused_tokens(self) -> int:
        """Tokens reused specifically from body stitch (not prefix)."""
        if self.is_committed and self.reused_body_blocks > 0:
            return self.total_reused_tokens
        return 0
