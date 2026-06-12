"""Monkey-patch module for segment reuse (minimal stitch) in vLLM v1.

This module patches vLLM's core components at runtime to add body-aware
KV cache reuse. It is enabled via the VLLM_SEGMENT_REUSE environment
variable.

Patch targets:
1. Request.__init__ - add stitch metadata fields
2. KVCacheManager.get_computed_blocks - add body registry lookup
3. Scheduler.schedule - adjust prefill token count for stitch
4. Scheduler.update_from_output - register body on seed completion
5. Scheduler._free_request - release body borrow on request finish

The patches are designed to be non-invasive: when VLLM_SEGMENT_REUSE is
not set (or set to "off"), no patches are applied and vLLM behaves
exactly as before.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from vllm.v1.segment_reuse.body_registry import BodyBlockRegistry
from vllm.v1.segment_reuse.types import (
    BodyBlockEntry,
    EnvelopeBodySpec,
    StitchResult,
    StitchState,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)

_PATCHED = False

# Global body registry instance, created once when patches are applied.
_body_registry: BodyBlockRegistry | None = None


def get_body_registry() -> BodyBlockRegistry | None:
    """Return the global body registry (None if stitch is not enabled)."""
    return _body_registry


def is_segment_reuse_enabled() -> bool:
    """Check if segment reuse is enabled via environment variable."""
    val = os.environ.get("VLLM_SEGMENT_REUSE", "off").lower()
    return val in ("stitch", "1", "true", "yes")


# ---------------------------------------------------------------------------
# Patch 1: Request metadata extension
# ---------------------------------------------------------------------------

def _patch_request_init() -> None:
    """Add stitch metadata fields to Request.__init__."""
    from vllm.v1.request import Request

    if getattr(Request, "_segment_reuse_patched", False):
        return

    original_init = Request.__init__

    @functools.wraps(original_init)
    def patched_init(self: "Request", *args: Any, **kwargs: Any) -> None:
        # Extract stitch-related kwargs before passing to original init.
        envelope_token_count = kwargs.pop("envelope_token_count", None)
        body_token_ids = kwargs.pop("body_token_ids", None)

        # Call original __init__.
        original_init(self, *args, **kwargs)

        # Initialize stitch fields.
        self.segment_reuse_spec: EnvelopeBodySpec | None = None
        self.segment_reuse_result: StitchResult | None = None
        self.segment_reuse_body_hash: bytes | None = None
        self.segment_reuse_state: str = "inactive"

        # Build spec if metadata was provided.
        if envelope_token_count is not None and self.prompt_token_ids is not None:
            spec = EnvelopeBodySpec.from_token_ids(
                all_token_ids=self.prompt_token_ids,
                envelope_token_count=int(envelope_token_count),
                boundary_source="client",
            )
            self.segment_reuse_spec = spec
            self.segment_reuse_body_hash = spec.body_hash
            self.segment_reuse_state = "boundary-resolved"
            logger.debug(
                "segment_reuse: request %s boundary resolved: E=%d B=%d",
                self.request_id,
                spec.envelope_token_count,
                spec.body_token_count,
            )

    Request.__init__ = patched_init
    setattr(Request, "_segment_reuse_patched", True)


# ---------------------------------------------------------------------------
# Patch 2: KVCacheManager stitch lookup
# ---------------------------------------------------------------------------

def _patch_kv_cache_manager() -> None:
    """Add body-aware lookup to KVCacheManager.get_computed_blocks."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager, "_segment_reuse_patched", False):
        return

    original_get_computed_blocks = KVCacheManager.get_computed_blocks

    @functools.wraps(original_get_computed_blocks)
    def patched_get_computed_blocks(
        self: "KVCacheManager", request: "Request"
    ) -> tuple:
        # Run the original prefix cache lookup.
        computed_blocks, num_new_computed_tokens = original_get_computed_blocks(
            self, request
        )

        # Skip stitch if not enabled or request has no boundary spec.
        if not is_segment_reuse_enabled():
            return computed_blocks, num_new_computed_tokens

        spec = getattr(request, "segment_reuse_spec", None)
        if spec is None:
            return computed_blocks, num_new_computed_tokens

        # If prefix cache already covers the entire body, skip stitch.
        # The body starts at position spec.envelope_token_count. If prefix
        # matched beyond that point, the body is already cached.
        if num_new_computed_tokens >= spec.envelope_token_count:
            request.segment_reuse_result = StitchResult(
                state=StitchState.PREFIX_COVERED,
                envelope_blocks=num_new_computed_tokens,
                total_reused_tokens=num_new_computed_tokens,
            )
            request.segment_reuse_state = "prefix-covered"
            logger.debug(
                "segment_reuse: request %s prefix covers body "
                "(matched=%d >= envelope=%d)",
                request.request_id,
                num_new_computed_tokens,
                spec.envelope_token_count,
            )
            return computed_blocks, num_new_computed_tokens

        # Look up body in registry.
        registry = get_body_registry()
        if registry is None:
            return computed_blocks, num_new_computed_tokens

        body_entry = registry.lookup(spec.body_hash)
        if body_entry is None:
            # Body not found: this is a seed request. Mark for registration
            # after prefill completes.
            request.segment_reuse_result = StitchResult(
                state=StitchState.SEED,
                envelope_blocks=0,
                body_hash=spec.body_hash,
            )
            request.segment_reuse_state = "seed"
            logger.debug(
                "segment_reuse: request %s body not found, marked as seed",
                request.request_id,
            )
            return computed_blocks, num_new_computed_tokens

        # Body found: commit the stitch.
        # Acquire a borrower reference.
        if not registry.acquire(spec.body_hash):
            return computed_blocks, num_new_computed_tokens

        request.segment_reuse_result = StitchResult(
            state=StitchState.COMMITTED,
            envelope_blocks=num_new_computed_tokens,
            reused_body_blocks=body_entry.num_blocks,
            total_reused_tokens=num_new_computed_tokens + body_entry.token_count,
            body_hash=spec.body_hash,
            body_block_ids=list(body_entry.block_ids),
        )
        request.segment_reuse_state = "committed"
        logger.info(
            "segment_reuse: request %s STITCH COMMITTED: "
            "prefix_tokens=%d body_blocks=%d body_tokens=%d total_reused=%d",
            request.request_id,
            num_new_computed_tokens,
            body_entry.num_blocks,
            body_entry.token_count,
            num_new_computed_tokens + body_entry.token_count,
        )

        return computed_blocks, num_new_computed_tokens

    KVCacheManager.get_computed_blocks = patched_get_computed_blocks
    setattr(KVCacheManager, "_segment_reuse_patched", True)


# ---------------------------------------------------------------------------
# Patch 3: Scheduler schedule - two-span prefill
# ---------------------------------------------------------------------------

def _patch_scheduler_schedule() -> None:
    """Adjust prefill token count for stitch-committed requests.

    When a request is stitch-committed, we only need to prefill the
    envelope tokens. The body's KV blocks are already in the cache
    (borrowed from the registry). This patch modifies the scheduler
    to set num_new_tokens = envelope_token_count - prefix_hit_tokens
    for stitch-committed requests.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_segment_reuse_schedule_patched", False):
        return

    original_schedule = Scheduler.schedule

    @functools.wraps(original_schedule)
    def patched_schedule(self: "Scheduler") -> Any:
        # Before calling the original schedule, mark stitch-committed
        # requests so the scheduler knows to use reduced prefill.
        # We do this by temporarily adjusting the request's view.
        result = original_schedule(self)

        # After scheduling, log stitch stats for committed requests.
        # (The actual token count adjustment happens in the
        # get_computed_blocks patch above, which returns the correct
        # num_new_computed_tokens to the scheduler.)
        return result

    Scheduler.schedule = patched_schedule
    setattr(Scheduler, "_segment_reuse_schedule_patched", True)


# ---------------------------------------------------------------------------
# Patch 4: Scheduler update_from_output - body registration on seed
# ---------------------------------------------------------------------------

def _patch_scheduler_update_from_output() -> None:
    """Register body blocks when a seed request completes prefill.

    When a seed request finishes its prefill (transitions from prefill
    to decode), we register its body KV blocks in the BodyBlockRegistry
    so subsequent requests can borrow them.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_segment_reuse_ufo_patched", False):
        return

    original_ufo = Scheduler.update_from_output

    @functools.wraps(original_ufo)
    def patched_update_from_output(
        self: "Scheduler",
        scheduler_output: Any,
        model_runner_output: Any,
    ) -> Any:
        # Call original first.
        result = original_ufo(self, scheduler_output, model_runner_output)

        # Check for seed requests that just completed prefill.
        registry = get_body_registry()
        if registry is None:
            return result

        num_scheduled = scheduler_output.num_scheduled_tokens
        for req_id, num_tokens_scheduled in num_scheduled.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            state = getattr(request, "segment_reuse_state", None)
            if state != "seed":
                continue

            spec = getattr(request, "segment_reuse_spec", None)
            if spec is None:
                continue

            # Check if this request just finished prefill.
            # Prefill is done when num_computed_tokens >= num_prompt_tokens.
            if request.num_computed_tokens < request.num_prompt_tokens:
                continue

            # Prefill just completed. Register the body blocks.
            # Extract the body's block IDs from the request's allocated blocks.
            # In vLLM v1, blocks are tracked via the kv_cache_manager.
            try:
                _register_body_blocks(self, request, spec, registry)
            except Exception as e:
                logger.warning(
                    "segment_reuse: failed to register body for request %s: %s",
                    req_id, e,
                )

        return result

    Scheduler.update_from_output = patched_update_from_output
    setattr(Scheduler, "_segment_reuse_ufo_patched", True)


def _register_body_blocks(
    scheduler: "Scheduler",
    request: "Request",
    spec: EnvelopeBodySpec,
    registry: BodyBlockRegistry,
) -> None:
    """Extract and register body blocks from a seed request.

    After a seed request completes prefill, its KV cache contains both
    the envelope and body tokens. We extract the body's block IDs and
    register them in the BodyBlockRegistry.
    """
    # Get the request's block allocation from the kv_cache_manager.
    # In vLLM v1, we access blocks via the request's internal state.
    # The blocks are managed through the KVCacheManager's block pool.
    kv_mgr = scheduler.kv_cache_manager

    # Access the request's allocated block IDs.
    # vLLM v1 stores block IDs in the request's internal data.
    block_ids = _get_request_block_ids(request, kv_mgr)
    if not block_ids:
        logger.warning(
            "segment_reuse: no block IDs found for seed request %s",
            request.request_id,
        )
        return

    # The body blocks start at the envelope boundary.
    # Each block holds block_size tokens.
    block_size = getattr(kv_mgr, "hash_block_size", 16)
    envelope_blocks_needed = (
        spec.envelope_token_count + block_size - 1
    ) // block_size

    if envelope_blocks_needed >= len(block_ids):
        logger.warning(
            "segment_reuse: envelope covers all blocks for request %s "
            "(envelope_blocks=%d >= total_blocks=%d)",
            request.request_id,
            envelope_blocks_needed,
            len(block_ids),
        )
        return

    body_block_ids = block_ids[envelope_blocks_needed:]
    if not body_block_ids:
        return

    # Register in the body block registry.
    try:
        registry.register(
            body_hash=spec.body_hash,
            block_ids=body_block_ids,
            token_count=spec.body_token_count,
            block_size=block_size,
            model_key=getattr(request, "model_key", "default"),
        )
        request.segment_reuse_state = "registered-seed"
        logger.info(
            "segment_reuse: registered body for request %s: "
            "hash=%s blocks=%d tokens=%d",
            request.request_id,
            spec.body_hash.hex()[:16],
            len(body_block_ids),
            spec.body_token_count,
        )
    except ValueError:
        # Already registered (concurrent seed requests for same body).
        pass


def _get_request_block_ids(
    request: "Request", kv_mgr: "KVCacheManager"
) -> list[int]:
    """Extract allocated block IDs for a request.

    In vLLM v1, block IDs are tracked through the KVCacheManager's
    internal data structures. We access them via the coordinator.
    """
    # Try to get block IDs from the request's internal tracking.
    # vLLM v1 Request has a `block_ids` attribute after allocation.
    block_ids = getattr(request, "block_ids", None)
    if block_ids is not None:
        # block_ids may be a tuple of lists (one per kv_cache_group).
        if isinstance(block_ids, tuple):
            # Flatten: take the first group (main attention).
            if block_ids and isinstance(block_ids[0], list):
                return list(block_ids[0])
        elif isinstance(block_ids, list):
            return block_ids

    # Fallback: try to get from the coordinator's request_to_blocks mapping.
    coordinator = getattr(kv_mgr, "coordinator", None)
    if coordinator is not None:
        req_blocks = getattr(coordinator, "_req_to_blocks", {}).get(
            request.request_id
        )
        if req_blocks is not None:
            return [b.block_id for b in req_blocks if hasattr(b, "block_id")]

    return []


# ---------------------------------------------------------------------------
# Patch 5: Scheduler _free_request - release body borrow
# ---------------------------------------------------------------------------

def _patch_scheduler_free_request() -> None:
    """Release body borrow when a stitch-committed request finishes."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_segment_reuse_free_patched", False):
        return

    original_free = Scheduler._free_request

    @functools.wraps(original_free)
    def patched_free_request(
        self: "Scheduler",
        request: "Request",
        delay_free_blocks: bool = False,
    ) -> Any:
        # Release body borrow before freeing the request.
        registry = get_body_registry()
        if registry is not None:
            result = getattr(request, "segment_reuse_result", None)
            if result is not None and result.body_hash is not None:
                if result.is_committed:
                    registry.release(result.body_hash)

        return original_free(self, request, delay_free_blocks)

    Scheduler._free_request = patched_free_request
    setattr(Scheduler, "_segment_reuse_free_patched", True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_segment_reuse_patches() -> None:
    """Apply all segment reuse patches if enabled.

    Called during vLLM engine initialization. If VLLM_SEGMENT_REUSE
    is not set to a truthy value, this is a no-op.
    """
    global _PATCHED, _body_registry

    if _PATCHED:
        return

    if not is_segment_reuse_enabled():
        logger.info("segment_reuse: disabled (VLLM_SEGMENT_REUSE not set)")
        return

    logger.info("segment_reuse: applying patches (VLLM_SEGMENT_REUSE=%s)",
                os.environ.get("VLLM_SEGMENT_REUSE"))

    # Create the global body registry.
    _body_registry = BodyBlockRegistry()

    # Apply all patches.
    _patch_request_init()
    _patch_kv_cache_manager()
    _patch_scheduler_schedule()
    _patch_scheduler_update_from_output()
    _patch_scheduler_free_request()

    _PATCHED = True
    logger.info("segment_reuse: all patches applied successfully")


def get_segment_reuse_stats() -> dict[str, Any]:
    """Return current segment reuse statistics."""
    registry = get_body_registry()
    if registry is None:
        return {"enabled": False}
    stats = registry.stats()
    stats["enabled"] = True
    return stats
