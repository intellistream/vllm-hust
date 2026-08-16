# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregation of per-worker :class:`ModelRunnerOutput` for request-owned
attention.

The generic :class:`ModelRunnerOutputAggregator` reuses the existing
all-worker outputs list produced by the executor transports (multiproc
message queues, Ray object refs) and therefore needs no new IPC.  It
composes:

* request-owner receipt envelopes (:class:`OwnerReceiptBatch`) emitted by
  workers for the current step, validated against the request-owner enabled
  contract below;
* when ``expected_sampling_owner_ranks`` is supplied (G3), the per-owner
  sampling envelopes (:class:`OwnerSamplingBatch`) validated against the
  sampling-enabled contract below, followed by a pure merge of all
  owner-partial payloads into one ordinary scheduler-facing
  :class:`ModelRunnerOutput`; and
* when a ``kv_aggregator`` is supplied, the existing KV connector
  composition, run on a copied outputs list so neither the original worker
  outputs nor the shared ``EMPTY_MODEL_RUNNER_OUTPUT`` singleton is ever
  mutated.

When no sampling batches are present (the default), receipt/KV aggregation
is byte-for-byte unchanged.

The request-owner enabled contract is fail-closed:

* Exactly one :class:`OwnerReceiptBatch` per expected process-global owner
  rank, even when ``events`` is empty.  Missing, duplicate, or unexpected
  owner ranks raise :class:`RuntimeError`.
* Every batch in one aggregation must carry the same ``emitted_step_seq``.
  An optional ``expected_step_seq`` supplied to :meth:`aggregate` must match.
* Every event's ``owner_id`` must equal the ``owner_rank`` of its enclosing
  batch, so a worker cannot spoof or misroute another owner's receipt.
* Exact repeated events (same ``owner_rank``, ``request_id``,
  ``owner_epoch``, ``command_seq``) with an identical payload are
  deduplicated, so exact payload replay is idempotent; the same identity with
  a conflicting payload is fatal.
* The selected output is shallow-copied before mutation and carries the
  aggregated batches sorted by numeric global owner rank, preserving event
  order within each batch.  A ``None`` selected output fails explicitly
  rather than silently dropping receipts.

The sampling-enabled contract (active only when
``expected_sampling_owner_ranks`` is supplied) is also fail-closed:

* Exactly one :class:`OwnerSamplingBatch` per expected transport slot, even
  when ``row_ids`` is empty.  Missing, duplicate, extra, or unexpected
  batches, ``None`` slots, and batches emitted into a sampling-disabled
  aggregator all raise :class:`RuntimeError`.  Supplying an empty
  ``expected_sampling_owner_ranks`` list is an enabled-but-empty contract
  that fails closed on any transport slot; it is never silently treated as
  disabled.
* Deferred-execute tolerance: when every transport slot is ``None`` (the
  deferred ``execute_model`` round, where every worker deferred sampling),
  :meth:`aggregate` returns ``None`` without demanding receipt or sampling
  envelopes.  A mixed ``None``/non-``None`` round still fails closed (a
  ``None`` slot cannot carry its envelopes), and the sampling-disabled
  aggregator never tolerates ``None`` slots.
* Every batch in one aggregation carries the same ``emitted_step_seq``; an
  optional ``expected_step_seq`` (as bound by
  :meth:`ModelRunnerOutputAggregator.for_step`) must match.
* An all-``None`` outputs list is the deferred ``execute_model`` round:
  :meth:`aggregate` returns ``None`` without demanding envelopes, and the
  same per-step adapter is then reused for the immediate ``sample_tokens``
  round.  A mixed ``None``/non-``None`` round fails closed.
* Each batch's ``owner_rank`` must equal its transport slot's expected
  owner rank, its ``row_ids`` must align exactly 1:1 with the partial
  output's ``req_ids`` (``row_ids[i].request_uid.request_id ==
  req_ids[i]``), and request ids and :class:`GlobalRowId`s must not
  repeat across slots.
* All owner-partial outputs are merged into one scheduler-facing
  :class:`ModelRunnerOutput` with bijective ``req_ids`` /
  ``req_id_to_index`` and one ``sampled_token_ids`` entry per request
  (``[]`` for discarded/no-token attempts).  ``logprobs`` merge only when
  every request has exactly one token (zero-token/discarded attempts make
  row cardinality ambiguous and fail closed); ``num_nans_in_logits`` keys
  must be requests of the enclosing partial output with nonnegative int
  values.  Nonempty prompt-logprobs, pooler, routed-experts, EC, cudagraph,
  KV-without-aggregator, and spec-shaped multi-token payloads fail closed
  rather than silently selecting one worker as carrier.
* The aggregated sampling batches are preserved on the returned output for
  later scheduler semantic validation.  The merge builds a fresh output
  and never mutates worker outputs or the ``EMPTY_MODEL_RUNNER_OUTPUT``
  singleton.

Stale/future lifecycle semantics are deliberately not interpreted here; the
scheduler-side reference coordinator owns receipt validation.

Call sites bind the shared aggregator to one exact scheduler step through
:meth:`ModelRunnerOutputAggregator.for_step`, which returns an immutable
per-step adapter delegating :meth:`~ModelRunnerOutputAggregatorStepAdapter.aggregate`
with ``expected_step_seq`` set to that step.  The shared aggregator itself
stores no per-step state, so concurrent steps cannot re-bind or corrupt each
other; adapters reuse the existing executor transport aggregator slots
(multiproc ``collective_rpc`` ``kv_output_aggregator``, Ray
``FutureWrapper``) by duck typing.
"""

from copy import copy
from typing import Any

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.core.sched.owner_layout import GlobalRowId
from vllm.v1.core.sched.ownership import (
    OwnerLeaseKey,
    OwnerReceipt,
    OwnerReceiptBatch,
)
from vllm.v1.outputs import (
    LogprobsLists,
    ModelRunnerOutput,
    OwnerSamplingBatch,
)

# Identity of a receipt event for deduplication:
# (process-global owner rank, request id, owner reuse epoch, command seq).
_EventIdentity = tuple[int, str, int, int]


class ModelRunnerOutputAggregator:
    """Aggregate all worker outputs into the output of ``output_rank``.

    Args:
        expected_owner_ranks: Process-global owner ranks that must each emit
            exactly one :class:`OwnerReceiptBatch` per aggregation.
        kv_aggregator: Optional :class:`KVOutputAggregator` for KV connector
            composition.  When provided, composition runs on a shallow-copied
            outputs list so neither the original selected output nor the
            shared ``EMPTY_MODEL_RUNNER_OUTPUT`` singleton is mutated.
        expected_sampling_owner_ranks: Optional process-global owner ranks
            that must each emit exactly one :class:`OwnerSamplingBatch` per
            aggregation (G3).  When supplied, all owner-partial outputs are
            merged into one scheduler-facing :class:`ModelRunnerOutput` under
            the fail-closed sampling contract of this module.  ``None`` (the
            default) keeps receipt/KV aggregation byte-for-byte unchanged;
            worker batches arriving at a sampling-disabled aggregator fail
            closed.  An empty list is an enabled-but-empty contract that
            fails closed on any transport slot.
    """

    def __init__(
        self,
        expected_owner_ranks: list[int],
        kv_aggregator: KVOutputAggregator | None = None,
        expected_sampling_owner_ranks: list[int] | None = None,
    ) -> None:
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in expected_owner_ranks
        ):
            raise TypeError(
                "expected_owner_ranks must contain nonnegative non-bool ints."
            )
        if len(set(expected_owner_ranks)) != len(expected_owner_ranks):
            raise ValueError("expected_owner_ranks must not contain duplicates.")
        if expected_sampling_owner_ranks is not None and any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in expected_sampling_owner_ranks
        ):
            raise TypeError(
                "expected_sampling_owner_ranks must contain nonnegative non-bool ints."
            )
        if expected_sampling_owner_ranks is not None and len(
            set(expected_sampling_owner_ranks)
        ) != len(expected_sampling_owner_ranks):
            raise ValueError(
                "expected_sampling_owner_ranks must not contain duplicates."
            )
        self._expected_owner_ranks = sorted(expected_owner_ranks)
        self._kv_aggregator = kv_aggregator
        self._expected_sampling_owner_ranks = (
            None
            if expected_sampling_owner_ranks is None
            else sorted(expected_sampling_owner_ranks)
        )

    def for_step(self, step_seq: int) -> "ModelRunnerOutputAggregatorStepAdapter":
        """Return an immutable, stateless per-step adapter bound to ``step_seq``.

        ``step_seq`` must be a positive non-bool ``int``; anything else
        fails closed at this call.  The adapter delegates every
        :meth:`aggregate` call with ``expected_step_seq=step_seq`` so stale or
        future worker receipts fail closed at this exact call site.  The
        shared aggregator itself stores no mutable per-step state, so one
        aggregator can serve any number of concurrently bound adapters
        without cross-step interference.
        """
        return ModelRunnerOutputAggregatorStepAdapter(self, step_seq)

    def aggregate(
        self,
        outputs: list[ModelRunnerOutput | None],
        output_rank: int = 0,
        expected_step_seq: int | None = None,
    ) -> ModelRunnerOutput | None:
        """Aggregate the all-worker outputs for one step.

        The outputs list is reused from the existing executor transport and is
        never mutated.  Returns a shallow copy of ``outputs[output_rank]``
        carrying the aggregated owner receipt batches sorted by numeric
        global owner rank.

        With the sampling contract enabled, an all-``None`` outputs list is
        the deferred ``execute_model`` round: it returns ``None`` without
        demanding envelopes, so the bound adapter can be reused for the
        immediate ``sample_tokens`` round.  Mixed ``None``/non-``None``
        rounds fail closed.
        """
        if not outputs:
            raise RuntimeError(
                "ModelRunnerOutputAggregator.aggregate() requires at least one "
                "worker output."
            )
        if expected_step_seq is not None and (
            isinstance(expected_step_seq, bool)
            or not isinstance(expected_step_seq, int)
            or expected_step_seq <= 0
        ):
            raise RuntimeError(
                "ModelRunnerOutputAggregator: expected_step_seq must be a "
                f"positive non-bool int, got {expected_step_seq!r}."
            )

        if self._expected_sampling_owner_ranks is not None and all(
            output is None for output in outputs
        ):
            # Deferred execute round: every worker deferred sampling, so no
            # receipt or sampling envelopes exist yet.  Propagate None
            # without demanding envelopes; the immediate sample_tokens round
            # carries them.  The exact transport cardinality is still
            # validated against both enabled rank contracts before the fast
            # path, so a truncated or padded all-None round can never be
            # silently accepted.  Mixed None/non-None rounds fail closed in
            # the collection paths below (a None transport slot cannot carry
            # its envelopes), and the sampling-disabled aggregator never
            # tolerates None slots.
            expected_sampling = len(self._expected_sampling_owner_ranks)
            if len(outputs) != expected_sampling:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: deferred transport round "
                    f"returned {len(outputs)} worker output slots, expected "
                    f"{expected_sampling} for the enabled sampling contract."
                )
            if self._expected_owner_ranks and len(outputs) != len(
                self._expected_owner_ranks
            ):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: deferred transport round "
                    f"returned {len(outputs)} worker output slots, expected "
                    f"{len(self._expected_owner_ranks)} for the enabled "
                    "owner receipt contract."
                )
            return None

        sampling_batches = None
        if self._expected_sampling_owner_ranks is not None:
            # Enabled, including an explicitly empty rank contract (which
            # fails closed on any transport slot instead of being silently
            # treated as disabled).
            sampling_batches = self._collect_owner_sampling_batches(
                outputs, expected_step_seq
            )
        else:
            self._reject_unexpected_sampling_batches(outputs)

        batches = self._collect_owner_batches(outputs, expected_step_seq)

        if self._kv_aggregator is not None:
            # KVOutputAggregator mutates the selected output in place; operate
            # on a copied outputs list so neither the original worker outputs
            # nor the shared EMPTY_MODEL_RUNNER_OUTPUT singleton is mutated.
            selected = self._kv_aggregator.aggregate(
                [copy(output) if output is not None else None for output in outputs],
                output_rank=output_rank,
            )
        else:
            selected = outputs[output_rank]

        if selected is None:
            raise RuntimeError(
                "ModelRunnerOutputAggregator: selected output (output_rank="
                f"{output_rank}) is None; refusing to silently drop aggregated "
                "owner receipt batches."
            )

        if sampling_batches is None:
            # Sampling disabled: byte-for-byte unchanged receipt/KV path.
            # Shallow copy before mutation: the selected output may be the
            # shared EMPTY_MODEL_RUNNER_OUTPUT singleton or a worker-owned
            # object.
            result = copy(selected)
            result.owner_receipt_batches = batches
            return result

        # Sampling enabled: merge every owner-partial output into one
        # ordinary scheduler-facing output, then compose with the receipt
        # batches and the (copied, KV-aggregated) selected output.
        result = self._merge_owner_outputs(outputs, sampling_batches)
        result.kv_connector_output = selected.kv_connector_output
        result.owner_receipt_batches = batches
        result.owner_sampling_batches = sampling_batches
        return result

    def _collect_owner_batches(
        self,
        outputs: list[ModelRunnerOutput | None],
        expected_step_seq: int | None,
    ) -> list[OwnerReceiptBatch]:
        """Validate the enabled contract and return per-rank batches sorted
        by numeric global owner rank, with exact repeated events deduplicated.
        """
        batches_by_rank: dict[int, OwnerReceiptBatch] = {}
        emitted_step_seq: int | None = None
        if self._expected_owner_ranks and len(outputs) != len(
            self._expected_owner_ranks
        ):
            raise RuntimeError(
                "ModelRunnerOutputAggregator: transport returned "
                f"{len(outputs)} worker output slots, expected "
                f"{len(self._expected_owner_ranks)}."
            )
        for slot, expected_rank in enumerate(self._expected_owner_ranks):
            output = outputs[slot]
            batches = None if output is None else output.owner_receipt_batches
            if batches is None or len(batches) != 1:
                count = 0 if batches is None else len(batches)
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: transport slot "
                    f"{slot} for owner rank {expected_rank} must carry exactly "
                    f"one OwnerReceiptBatch, got {count}."
                )
            batch = batches[0]
            if isinstance(batch.owner_rank, bool) or not isinstance(
                batch.owner_rank, int
            ):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: owner_rank must be a "
                    f"non-bool int, got {batch.owner_rank!r}."
                )
            if batch.owner_rank != expected_rank:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: transport slot "
                    f"{slot} belongs to owner rank {expected_rank}, but its "
                    f"batch claims owner rank {batch.owner_rank}."
                )
            if (
                isinstance(batch.emitted_step_seq, bool)
                or not isinstance(batch.emitted_step_seq, int)
                or batch.emitted_step_seq <= 0
            ):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: emitted_step_seq must be "
                    "a positive non-bool int, got "
                    f"{batch.emitted_step_seq!r}."
                )
            if emitted_step_seq is None:
                emitted_step_seq = batch.emitted_step_seq
            elif batch.emitted_step_seq != emitted_step_seq:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: mixed emitted_step_seq "
                    f"({emitted_step_seq} vs {batch.emitted_step_seq}) in "
                    "one aggregation."
                )
            batches_by_rank[batch.owner_rank] = batch

        missing = [
            rank for rank in self._expected_owner_ranks if rank not in batches_by_rank
        ]
        if missing:
            raise RuntimeError(
                "ModelRunnerOutputAggregator: missing OwnerReceiptBatch for "
                f"expected owner rank(s) {missing}."
            )

        if (
            expected_step_seq is not None
            and emitted_step_seq is not None
            and emitted_step_seq != expected_step_seq
        ):
            raise RuntimeError(
                "ModelRunnerOutputAggregator: emitted_step_seq "
                f"{emitted_step_seq} does not match expected_step_seq "
                f"{expected_step_seq}."
            )

        # Deduplicate exact repeated events by identity, preserving first
        # occurrence order; conflicting payloads for one identity are fatal.
        seen: dict[_EventIdentity, OwnerReceipt] = {}
        aggregated: list[OwnerReceiptBatch] = []
        for rank in self._expected_owner_ranks:
            batch = batches_by_rank[rank]
            deduped_events: list[OwnerReceipt] = []
            for event in batch.events:
                # Every event must belong to the owner that emitted the
                # enclosing batch; otherwise a worker could spoof or misroute
                # another owner's receipt.
                if (
                    isinstance(event.owner_id, bool)
                    or not isinstance(event.owner_id, int)
                    or event.owner_id != batch.owner_rank
                ):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: OwnerReceipt owner_id "
                        f"{event.owner_id} does not match enclosing batch "
                        f"owner_rank {batch.owner_rank} (request_id="
                        f"{event.key.request_id!r}, command_seq="
                        f"{event.command_seq})."
                    )
                if not isinstance(event.key, OwnerLeaseKey):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: OwnerReceipt key must be "
                        f"an OwnerLeaseKey, got {event.key!r}."
                    )
                if not isinstance(event.key.request_id, str):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: request_id must be a string."
                    )
                if (
                    isinstance(event.key.owner_epoch, bool)
                    or not isinstance(event.key.owner_epoch, int)
                    or event.key.owner_epoch < 0
                ):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: owner_epoch must be a "
                        "nonnegative non-bool int."
                    )
                if (
                    isinstance(event.command_seq, bool)
                    or not isinstance(event.command_seq, int)
                    or event.command_seq <= 0
                ):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: command_seq must be a "
                        "positive non-bool int."
                    )
                identity = (
                    batch.owner_rank,
                    event.key.request_id,
                    event.key.owner_epoch,
                    event.command_seq,
                )
                prior = seen.get(identity)
                if prior is None:
                    seen[identity] = event
                    deduped_events.append(event)
                elif prior != event:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: conflicting duplicate "
                        f"OwnerReceipt for identity (owner_rank="
                        f"{batch.owner_rank}, request_id="
                        f"{event.key.request_id!r}, owner_epoch="
                        f"{event.key.owner_epoch}, command_seq="
                        f"{event.command_seq})."
                    )
                # else: exact payload replay, idempotent; skip the duplicate.
            if len(deduped_events) == len(batch.events):
                aggregated.append(batch)
            else:
                aggregated.append(
                    OwnerReceiptBatch(
                        owner_rank=batch.owner_rank,
                        emitted_step_seq=batch.emitted_step_seq,
                        events=tuple(deduped_events),
                        readiness=getattr(batch, "readiness", ()),
                        free_capacity=batch.free_capacity,
                        resident_pages=batch.resident_pages,
                        pending_dma=batch.pending_dma,
                        cache_pool=batch.cache_pool,
                    )
                )
        return aggregated

    def _reject_unexpected_sampling_batches(
        self, outputs: list[ModelRunnerOutput | None]
    ) -> None:
        """Sampling disabled: any worker-claimed :class:`OwnerSamplingBatch`
        is a contract violation and is never silently dropped or carried."""
        for slot, output in enumerate(outputs):
            if output is not None and output.owner_sampling_batches is not None:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: sampling aggregation is "
                    "disabled, but transport slot "
                    f"{slot} carries owner_sampling_batches; refusing to "
                    "silently drop or carry them."
                )

    def _collect_owner_sampling_batches(
        self,
        outputs: list[ModelRunnerOutput | None],
        expected_step_seq: int | None,
    ) -> list[OwnerSamplingBatch]:
        """Validate the sampling-enabled contract and return the per-rank
        batches sorted by numeric global owner rank.

        Exactly one batch per expected transport slot, exact slot owner and
        step fence, no duplicate request ids or :class:`GlobalRowId`s, and
        exact 1:1 ``row_ids`` / partial ``req_ids`` alignment (including a
        bijective partial ``req_id_to_index``).
        """
        assert self._expected_sampling_owner_ranks is not None
        expected_ranks = self._expected_sampling_owner_ranks
        if len(outputs) != len(expected_ranks):
            raise RuntimeError(
                "ModelRunnerOutputAggregator: transport returned "
                f"{len(outputs)} worker output slots, expected "
                f"{len(expected_ranks)}."
            )
        batches_by_rank: dict[int, OwnerSamplingBatch] = {}
        emitted_step_seq: int | None = None
        seen_req_ids: set[str] = set()
        seen_row_ids: set[GlobalRowId] = set()
        for slot, expected_rank in enumerate(expected_ranks):
            output = outputs[slot]
            batches = None if output is None else output.owner_sampling_batches
            if batches is None or len(batches) != 1:
                count = 0 if batches is None else len(batches)
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: transport slot "
                    f"{slot} for owner rank {expected_rank} must carry "
                    f"exactly one OwnerSamplingBatch, got {count}."
                )
            batch = batches[0]
            assert output is not None
            if not isinstance(batch, OwnerSamplingBatch):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: transport slot "
                    f"{slot} for owner rank {expected_rank} must carry an "
                    f"OwnerSamplingBatch, got {type(batch).__name__}."
                )
            if isinstance(batch.owner_rank, bool) or not isinstance(
                batch.owner_rank, int
            ):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: owner_rank must be a "
                    f"non-bool int, got {batch.owner_rank!r}."
                )
            if batch.owner_rank != expected_rank:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: transport slot "
                    f"{slot} belongs to owner rank {expected_rank}, but its "
                    f"batch claims owner rank {batch.owner_rank}."
                )
            if (
                isinstance(batch.emitted_step_seq, bool)
                or not isinstance(batch.emitted_step_seq, int)
                or batch.emitted_step_seq <= 0
            ):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: emitted_step_seq must be "
                    "a positive non-bool int, got "
                    f"{batch.emitted_step_seq!r}."
                )
            if emitted_step_seq is None:
                emitted_step_seq = batch.emitted_step_seq
            elif batch.emitted_step_seq != emitted_step_seq:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: mixed emitted_step_seq "
                    f"({emitted_step_seq} vs {batch.emitted_step_seq}) in "
                    "one aggregation."
                )
            batches_by_rank[batch.owner_rank] = batch

            req_ids = output.req_ids
            if len(batch.row_ids) != len(req_ids):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: owner slot "
                    f"{slot} (owner rank {expected_rank}) batch row_ids "
                    f"length {len(batch.row_ids)} must equal partial output "
                    f"req_ids length {len(req_ids)} (1:1 alignment)."
                )
            if not isinstance(output.req_id_to_index, dict) or len(
                output.req_id_to_index
            ) != len(req_ids):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: partial output "
                    "req_id_to_index must be bijective over req_ids "
                    f"(slot {slot}, owner rank {expected_rank})."
                )
            for i, row_id in enumerate(batch.row_ids):
                if not isinstance(row_id, GlobalRowId):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: row_ids must contain "
                        f"GlobalRowId, got {row_id!r} (slot {slot})."
                    )
                if row_id in seen_row_ids:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: duplicate GlobalRowId "
                        f"{row_id!r} across owner slots (slot {slot})."
                    )
                seen_row_ids.add(row_id)
                request_id = row_id.request_uid.request_id
                if not isinstance(request_id, str):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: request_id must be a "
                        f"string, got {request_id!r} (slot {slot})."
                    )
                if request_id != req_ids[i]:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: row "
                        f"{i} of owner slot {slot} (request_id="
                        f"{request_id!r}) does not match partial output "
                        f"req_ids[{i}]={req_ids[i]!r}."
                    )
            for i, req_id in enumerate(req_ids):
                if not isinstance(req_id, str):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: request_id must be a "
                        f"string, got {req_id!r} (slot {slot})."
                    )
                if req_id in seen_req_ids:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: duplicate request id "
                        f"{req_id!r} across owner slots (slot {slot})."
                    )
                seen_req_ids.add(req_id)
                if output.req_id_to_index.get(req_id) != i:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: partial output "
                        "req_id_to_index must be bijective over req_ids "
                        f"(slot {slot}, owner rank {expected_rank})."
                    )

        if (
            expected_step_seq is not None
            and emitted_step_seq is not None
            and emitted_step_seq != expected_step_seq
        ):
            raise RuntimeError(
                "ModelRunnerOutputAggregator: emitted_step_seq "
                f"{emitted_step_seq} does not match expected_step_seq "
                f"{expected_step_seq}."
            )
        return [batches_by_rank[rank] for rank in expected_ranks]

    def _merge_owner_outputs(
        self,
        outputs: list[ModelRunnerOutput | None],
        sampling_batches: list[OwnerSamplingBatch],
    ) -> ModelRunnerOutput:
        """Merge all owner-partial payloads into one ordinary scheduler-facing
        :class:`ModelRunnerOutput`.

        Builds a fresh output (never mutating worker outputs or the
        ``EMPTY_MODEL_RUNNER_OUTPUT`` singleton).  Merging is proven only
        for: bijective ``req_ids`` / ``req_id_to_index``, one
        ``sampled_token_ids`` list per request (``[]`` for
        discarded/no-token attempts and one-or-more entries for ordinary or
        speculative output), and ``logprobs`` /
        ``num_nans_in_logits`` under the shape invariants below.
        ``logprobs`` merge is only proven when every request has exactly one
        token, because a zero-token/discarded attempt may or may not
        contribute a logprobs row; any mixed cardinality fails closed.  Any
        other nonempty payload field fails closed instead of silently
        selecting one worker as the fixed carrier.
        """
        merged_req_ids: list[str] = []
        merged_sampled_token_ids: list[list[int]] = []
        logprobs_parts: list[LogprobsLists] = []
        max_num_logprobs: int | None = None
        any_logprobs = False
        any_zero_token = False
        any_multi_token = False
        merged_num_nans: dict[str, int] = {}
        any_num_nans = False
        for slot, output in enumerate(outputs):
            if output.prompt_logprobs_dict:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: merging nonempty "
                    "prompt_logprobs_dict is unsupported (slot "
                    f"{slot}, owner rank {sampling_batches[slot].owner_rank}); "
                    "refusing to silently select a fixed carrier."
                )
            if output.pooler_output is not None:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: merging pooler_output is "
                    f"unsupported (slot {slot}, owner rank "
                    f"{sampling_batches[slot].owner_rank}); refusing to "
                    "silently select a fixed carrier."
                )
            if output.routed_experts is not None:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: merging routed_experts is "
                    f"unsupported (slot {slot}, owner rank "
                    f"{sampling_batches[slot].owner_rank}); refusing to "
                    "silently select a fixed carrier."
                )
            if output.ec_connector_output is not None:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: merging ec_connector_output "
                    f"is unsupported (slot {slot}, owner rank "
                    f"{sampling_batches[slot].owner_rank}); refusing to "
                    "silently select a fixed carrier."
                )
            if output.cudagraph_stats is not None:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: merging cudagraph_stats is "
                    f"unsupported (slot {slot}, owner rank "
                    f"{sampling_batches[slot].owner_rank}); refusing to "
                    "silently select a fixed carrier."
                )
            if output.kv_connector_output is not None and self._kv_aggregator is None:
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: kv_connector_output on "
                    f"slot {slot} without a KV aggregator is unsupported; "
                    "refusing to silently select a fixed carrier."
                )
            if len(output.sampled_token_ids) != len(output.req_ids):
                raise RuntimeError(
                    "ModelRunnerOutputAggregator: sampled_token_ids must be "
                    f"aligned 1:1 with req_ids (slot {slot}: "
                    f"{len(output.sampled_token_ids)} token lists vs "
                    f"{len(output.req_ids)} requests)."
                )
            for tokens in output.sampled_token_ids:
                if not isinstance(tokens, list):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: every sampled-token "
                        f"entry must be a list (slot {slot}, got "
                        f"{type(tokens).__name__}); refusing to merge."
                    )
                if not tokens:
                    any_zero_token = True
                elif len(tokens) > 1:
                    any_multi_token = True
            logprobs = output.logprobs
            if logprobs is not None:
                if not isinstance(logprobs, LogprobsLists):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: logprobs must be a "
                        f"LogprobsLists (slot {slot}), got "
                        f"{type(logprobs).__name__}."
                    )
                if logprobs.cu_num_generated_tokens is not None:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: spec-shaped "
                        "multi-token logprobs (cu_num_generated_tokens) are "
                        f"unsupported (slot {slot}); refusing to merge."
                    )
                if (
                    len(logprobs.logprob_token_ids.shape) != 2
                    or logprobs.logprobs.shape != logprobs.logprob_token_ids.shape
                    or len(logprobs.sampled_token_ranks.shape) != 1
                    or len(logprobs.sampled_token_ranks)
                    != len(logprobs.logprob_token_ids)
                ):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: malformed logprobs "
                        f"arrays (slot {slot}); refusing to merge."
                    )
                if len(logprobs.logprob_token_ids) != len(output.req_ids):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: logprobs row count "
                        f"{len(logprobs.logprob_token_ids)} must equal "
                        f"partial output req_ids count {len(output.req_ids)} "
                        f"(slot {slot})."
                    )
                if max_num_logprobs is None:
                    max_num_logprobs = logprobs.logprob_token_ids.shape[1]
                elif logprobs.logprob_token_ids.shape[1] != max_num_logprobs:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: inconsistent "
                        f"max_num_logprobs across slots (slot {slot}); "
                        "refusing to merge."
                    )
                any_logprobs = True
                logprobs_parts.append(logprobs)
            merged_req_ids.extend(output.req_ids)
            merged_sampled_token_ids.extend(output.sampled_token_ids)
            if output.num_nans_in_logits is not None:
                if not isinstance(output.num_nans_in_logits, dict):
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: num_nans_in_logits "
                        f"must be a dict (slot {slot}), got "
                        f"{type(output.num_nans_in_logits).__name__}."
                    )
                any_num_nans = True
                req_ids_set = set(output.req_ids)
                for req_id, num_nans in output.num_nans_in_logits.items():
                    if not isinstance(req_id, str):
                        raise RuntimeError(
                            "ModelRunnerOutputAggregator: "
                            "num_nans_in_logits key must be a string, got "
                            f"{req_id!r} (slot {slot})."
                        )
                    if req_id not in req_ids_set:
                        raise RuntimeError(
                            "ModelRunnerOutputAggregator: "
                            f"num_nans_in_logits key {req_id!r} is not a "
                            f"request of partial output (slot {slot}); "
                            "refusing to merge a foreign key."
                        )
                    if (
                        isinstance(num_nans, bool)
                        or not isinstance(num_nans, int)
                        or num_nans < 0
                    ):
                        raise RuntimeError(
                            "ModelRunnerOutputAggregator: "
                            "num_nans_in_logits values must be nonnegative "
                            f"non-bool ints, got {num_nans!r} (slot {slot})."
                        )
                    if req_id in merged_num_nans:
                        raise RuntimeError(
                            "ModelRunnerOutputAggregator: duplicate "
                            f"num_nans_in_logits entry for {req_id!r} "
                            f"(slot {slot})."
                        )
                    merged_num_nans[req_id] = num_nans

        if any_logprobs and any_zero_token:
            raise RuntimeError(
                "ModelRunnerOutputAggregator: merging logprobs alongside "
                "zero-token/discarded requests is ambiguous (a discarded "
                "attempt may or may not contribute a logprobs row); "
                "refusing to merge."
            )
        if any_logprobs and any_multi_token:
            raise RuntimeError(
                "ModelRunnerOutputAggregator: merging logprobs alongside "
                "speculative multi-token requests is unsupported until the "
                "per-owner cu_num_generated_tokens vectors are composed; "
                "refusing to misalign merged logprobs."
            )

        if any_logprobs:
            for slot, output in enumerate(outputs):
                if output.req_ids and output.logprobs is None:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: logprobs must be "
                        "present on every nonempty partial output when any "
                        f"partial carries logprobs (slot {slot}); refusing "
                        "to misalign merged logprobs."
                    )
            merged_logprobs: LogprobsLists | None = LogprobsLists(
                np.concatenate([part.logprob_token_ids for part in logprobs_parts]),
                np.concatenate([part.logprobs for part in logprobs_parts]),
                np.concatenate([part.sampled_token_ranks for part in logprobs_parts]),
                None,
            )
        else:
            merged_logprobs = None

        merged_req_id_to_index = {
            req_id: idx for idx, req_id in enumerate(merged_req_ids)
        }
        if len(merged_req_id_to_index) != len(merged_req_ids):
            raise RuntimeError(
                "ModelRunnerOutputAggregator: merged req_ids must be "
                f"bijective; got {len(merged_req_ids)} ids and "
                f"{len(merged_req_id_to_index)} distinct entries."
            )
        return ModelRunnerOutput(
            req_ids=merged_req_ids,
            req_id_to_index=merged_req_id_to_index,
            sampled_token_ids=merged_sampled_token_ids,
            logprobs=merged_logprobs,
            num_nans_in_logits=merged_num_nans if any_num_nans else None,
        )


class ModelRunnerOutputAggregatorStepAdapter:
    """Immutable per-step view of a shared :class:`ModelRunnerOutputAggregator`.

    Binds one exact ``step_seq``: every :meth:`aggregate` call delegates to
    the shared aggregator with ``expected_step_seq`` set to that step, so a
    stale or future worker emission fails closed.  The adapter is frozen
    after construction: ``_aggregator`` and ``_step_seq`` cannot be
    reassigned, no new attributes can be added, and ``step_seq`` must be a
    positive non-bool ``int``.  Creating or using it never mutates the
    shared aggregator or any other adapter.

    It exposes the same duck-typed ``aggregate(outputs, output_rank=...)``
    surface as :class:`KVOutputAggregator`, so executors can pass it through
    existing aggregator slots (the multiproc ``collective_rpc``
    ``kv_output_aggregator`` argument and the Ray ``FutureWrapper``
    aggregator) without new keywords.
    """

    __slots__ = ("_aggregator", "_step_seq", "_frozen")

    def __init__(self, aggregator: ModelRunnerOutputAggregator, step_seq: int) -> None:
        if isinstance(step_seq, bool) or not isinstance(step_seq, int):
            raise TypeError(
                "ModelRunnerOutputAggregatorStepAdapter: step_seq must be an "
                f"int, got {type(step_seq).__name__} ({step_seq!r})."
            )
        if step_seq <= 0:
            raise ValueError(
                "ModelRunnerOutputAggregatorStepAdapter: step_seq must be "
                f"positive, got {step_seq}."
            )
        self._aggregator = aggregator
        self._step_seq = step_seq
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable after construction."
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        # Unconditional: deleting the frozen flag would unlock mutation, so
        # attribute deletion is never allowed on the adapter.
        raise AttributeError(f"{type(self).__name__} is immutable after construction.")

    @property
    def step_seq(self) -> int:
        """The exact scheduler step sequence this adapter is bound to."""
        return self._step_seq

    def aggregate(
        self,
        outputs: list[ModelRunnerOutput | None],
        output_rank: int = 0,
    ) -> ModelRunnerOutput | None:
        """Aggregate one step's worker outputs against the bound step.

        Delegates to the shared aggregator with
        ``expected_step_seq=self.step_seq``; ``output_rank`` selects the
        output carrier exactly as on the shared aggregator.
        """
        return self._aggregator.aggregate(
            outputs,
            output_rank=output_rank,
            expected_step_seq=self._step_seq,
        )
