# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-ID-free prefix-cache protocol for request-owned attention.

Request-owned KV tables are private to one worker rank.  The scheduler may
send content hashes and receive exact hit lengths, but it must never observe
the physical blocks that realize a hit.  These dependency-light dataclasses
are the wire vocabulary for that boundary.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OwnerPrefixDescriptor:
    """Immutable hash chain available to an owner at RESERVE time.

    ``block_hashes`` uses the scheduler's configured hash-block granularity.
    An empty tuple is a valid cache-eligible request shorter than one hash
    block.  ``None`` at the containing allocation descriptor, rather than an
    empty value here, means prefix-cache lookup is disabled for that request.
    """

    block_hashes: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_hashes, tuple):
            raise TypeError(
                "block_hashes must be a tuple of bytes, got "
                f"{self.block_hashes!r}."
            )
        for block_hash in self.block_hashes:
            if not isinstance(block_hash, bytes) or not block_hash:
                raise TypeError(
                    "block_hashes must contain nonempty bytes values, got "
                    f"{block_hash!r}."
                )


def validate_owner_prefix_receipt_hit(
    hit: int | None,
    accepted: bool,
    runnable_num_tokens: int | None,
) -> None:
    """Validate the optional exact hit carried by an owner receipt."""

    if hit is not None and (
        isinstance(hit, bool) or not isinstance(hit, int) or hit < 0
    ):
        raise TypeError(
            "prefix_cache_hit_tokens must be None or a nonnegative "
            f"non-bool int, got {hit!r}."
        )
    if hit is not None and not accepted:
        raise ValueError("a rejected owner receipt cannot publish a prefix-cache hit")
    if (
        hit is not None
        and runnable_num_tokens is not None
        and hit > runnable_num_tokens
    ):
        raise ValueError(
            "prefix_cache_hit_tokens must not exceed runnable_num_tokens, "
            f"got {hit} > {runnable_num_tokens}."
        )
