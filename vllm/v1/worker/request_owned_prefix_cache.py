# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-local prefix caching over the real hybrid KV cache manager.

The controller deliberately separates allocation from publication.  RESERVE
may attach blocks that were already published by an earlier successful
forward, but every newly allocated block is passed to ``allocate_slots`` with
``delay_cache_blocks=True``.  Only :meth:`commit` publishes newly computed
full blocks after the worker's terminal success fence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.request_owned_prefix import OwnerPrefixDescriptor
from vllm.v1.core.sched.ownership import (
    OwnerCommand,
    OwnerLeaseKey,
    OwnerReceiptBatch,
)
from vllm.v1.request import RequestStatus


class RequestOwnedCacheRequest:
    """Small Request-like facade observed by ``KVCacheManager``.

    Request-owned attention supports text-only, no-LoRA prefix caching.  The
    content hash chain is computed by the scheduler's real ``Request`` and is
    transported without token payloads or physical block identities.
    """

    __slots__ = (
        "block_hashes",
        "cache_salt",
        "num_computed_tokens",
        "num_preemptions",
        "num_prompt_tokens",
        "num_tokens",
        "request_id",
        "skip_reading_prefix_cache",
        "status",
    )

    def __init__(
        self,
        request_id: str,
        num_tokens: int,
        num_computed_tokens: int,
        num_prompt_tokens: int,
        block_hashes: tuple[bytes, ...] = (),
        *,
        skip_reading_prefix_cache: bool = True,
    ) -> None:
        self.request_id = request_id
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.status = RequestStatus.RUNNING
        self.block_hashes = list(block_hashes)
        self.skip_reading_prefix_cache = skip_reading_prefix_cache
        self.cache_salt = None
        self.num_preemptions = 0


@dataclass(frozen=True, slots=True)
class PrefixAllocation:
    accepted: bool
    blocks: KVCacheBlocks | None
    hit_tokens: int


@dataclass(slots=True)
class _PrefixRecord:
    descriptor: OwnerPrefixDescriptor
    num_prompt_tokens: int
    num_tokens: int
    last_commit_tokens: int = 0


class RequestOwnedPrefixCache:
    """Exact owner-local lookup/commit boundary for request-owned KV."""

    def __init__(self, manager: KVCacheManager) -> None:
        self._manager = manager
        self.enabled = manager.enable_caching
        self._records: dict[str, _PrefixRecord] = {}
        self._receipt_hits: dict[tuple[OwnerLeaseKey, int], int] = {}
        self._fail_stop: str | None = None

    def reserve(
        self,
        command: OwnerCommand,
        allocator_id: str,
    ) -> PrefixAllocation:
        """Lookup a jointly complete hybrid prefix, then reserve the suffix."""

        self._guard()
        allocation = command.allocation
        if allocation is None:
            raise ValueError("request-owned prefix RESERVE requires allocation facts")

        descriptor = allocation.prefix if self.enabled else None
        hit_blocks: KVCacheBlocks | None = None
        hit_tokens = 0
        if (
            descriptor is not None
            and allocation.num_computed_tokens == 0
            and allocation.num_prompt_tokens > 0
            and command.required_num_tokens > 0
        ):
            # A RESERVE may cover only the first chunk of a much longer
            # prompt. Never attach a hit beyond the worker-granted horizon.
            lookup_num_tokens = min(
                allocation.num_prompt_tokens, command.required_num_tokens
            )
            lookup = RequestOwnedCacheRequest(
                request_id=allocator_id,
                num_tokens=lookup_num_tokens,
                num_computed_tokens=0,
                num_prompt_tokens=allocation.num_prompt_tokens,
                block_hashes=descriptor.block_hashes,
                skip_reading_prefix_cache=False,
            )
            hit_blocks, hit_tokens = self._manager.get_computed_blocks(lookup)

        computed = allocation.num_computed_tokens
        if hit_tokens:
            if computed != 0:
                raise RuntimeError(
                    "owner-local prefix hit cannot be combined with an existing "
                    "computed prefix"
                )
            computed = hit_tokens

        num_new_tokens = command.required_num_tokens - computed
        blocks: KVCacheBlocks | None = None
        if num_new_tokens > 0:
            request = self._request(
                allocator_id,
                command.required_num_tokens,
                allocation.num_computed_tokens,
                allocation.num_prompt_tokens,
                descriptor,
            )
            blocks = self._manager.allocate_slots(
                request,
                num_new_tokens=num_new_tokens,
                num_new_computed_tokens=hit_tokens,
                new_computed_blocks=hit_blocks,
                full_sequence_must_fit=True,
                has_scheduled_reqs=False,
                # RESERVE precedes execution. Publishing here would make
                # unwritten physical seats reachable by a later request.
                delay_cache_blocks=self.enabled,
            )
            if blocks is None:
                return PrefixAllocation(
                    accepted=False, blocks=None, hit_tokens=0
                )

        if descriptor is not None:
            self._records[allocator_id] = _PrefixRecord(
                descriptor=descriptor,
                num_prompt_tokens=allocation.num_prompt_tokens,
                num_tokens=command.required_num_tokens,
                last_commit_tokens=hit_tokens,
            )
            self._receipt_hits[(command.key, command.command_seq)] = hit_tokens
        return PrefixAllocation(
            accepted=True, blocks=blocks, hit_tokens=hit_tokens
        )

    def allocate(
        self,
        allocator_id: str,
        num_prompt_tokens: int,
        num_computed_tokens: int,
        num_new_tokens: int,
        num_tokens: int,
    ) -> KVCacheBlocks | None:
        """Allocate an EXTEND/RESTORE suffix without a new prefix lookup."""

        self._guard()
        record = self._records.get(allocator_id)
        request = self._request(
            allocator_id,
            num_tokens,
            num_computed_tokens,
            num_prompt_tokens,
            record.descriptor if record is not None else None,
        )
        blocks = self._manager.allocate_slots(
            request,
            num_new_tokens=num_new_tokens,
            full_sequence_must_fit=True,
            has_scheduled_reqs=False,
            delay_cache_blocks=self.enabled,
        )
        if blocks is not None and record is not None:
            record.num_tokens = max(record.num_tokens, num_tokens)
        return blocks

    def commit(self, allocator_id: str, num_computed_tokens: int) -> None:
        """Publish only full hashed blocks below a successful forward fence."""

        self._guard()
        record = self._records.get(allocator_id)
        if record is None:
            return
        max_hash_tokens = (
            len(record.descriptor.block_hashes) * self._manager.hash_block_size
        )
        commit_tokens = min(num_computed_tokens, max_hash_tokens)
        if commit_tokens <= record.last_commit_tokens:
            return
        request = self._request(
            allocator_id,
            record.num_tokens,
            num_computed_tokens,
            record.num_prompt_tokens,
            record.descriptor,
        )
        try:
            self._manager.cache_blocks(request, commit_tokens)
        except BaseException as exc:
            # cache_blocks may have published a subset of hybrid groups before
            # an unexpected exception. Never permit another lookup on this
            # rank after that indeterminate boundary.
            self._fail_stop = (
                f"prefix publication failed for {allocator_id!r} at "
                f"{commit_tokens} tokens ({exc!r})"
            )
            raise RuntimeError(self._fail_stop) from exc
        record.last_commit_tokens = commit_tokens

    def forget(self, allocator_id: str) -> None:
        self._records.pop(allocator_id, None)

    def reset(self) -> bool:
        """Reset only at a quiescent owner-local lifecycle boundary.

        Active records or unacknowledged RESERVE receipts mean that the
        scheduler and worker still share live protocol state.  Refuse the
        reset rather than clearing physical hashes behind that state.  A
        successful real-manager reset makes an earlier publication fail-stop
        recoverable because no published block remains reachable afterwards.
        """

        if self._records or self._receipt_hits:
            return False
        if not self._manager.reset_prefix_cache():
            return False
        self._fail_stop = None
        return True

    def decorate_receipt_batch(self, batch: OwnerReceiptBatch) -> OwnerReceiptBatch:
        """Attach exact physical hits to the matching logical RESERVE receipts."""

        events = tuple(
            replace(
                event,
                prefix_cache_hit_tokens=self._receipt_hits.get(
                    (event.key, event.command_seq), event.prefix_cache_hit_tokens
                ),
            )
            for event in batch.events
        )
        return replace(batch, events=events)

    def acknowledge_receipt_batch(self, batch: OwnerReceiptBatch) -> None:
        """Drop hit annotations only after the logical manager commits."""

        for event in batch.events:
            self._receipt_hits.pop((event.key, event.command_seq), None)

    def _request(
        self,
        allocator_id: str,
        num_tokens: int,
        num_computed_tokens: int,
        num_prompt_tokens: int,
        descriptor: OwnerPrefixDescriptor | None,
    ) -> RequestOwnedCacheRequest:
        return RequestOwnedCacheRequest(
            request_id=allocator_id,
            num_tokens=num_tokens,
            num_computed_tokens=num_computed_tokens,
            num_prompt_tokens=num_prompt_tokens,
            block_hashes=descriptor.block_hashes if descriptor is not None else (),
            skip_reading_prefix_cache=descriptor is None,
        )

    def _guard(self) -> None:
        if self._fail_stop is not None:
            raise RuntimeError(
                "request-owned prefix cache is in fail-stop state: "
                f"{self._fail_stop}"
            )
