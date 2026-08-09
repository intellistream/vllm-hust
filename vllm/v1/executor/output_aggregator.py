# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregation of per-worker :class:`ModelRunnerOutput` for G0
request-owned attention.

The generic :class:`ModelRunnerOutputAggregator` reuses the existing
all-worker outputs list produced by the executor transports (multiproc
message queues, Ray object refs) and therefore needs no new IPC.  It
composes:

* request-owner receipt envelopes (:class:`OwnerReceiptBatch`) emitted by
  workers for the current step, validated against the request-owner enabled
  contract below; and, when a ``kv_aggregator`` is supplied,
* the existing KV connector composition, run on a copied outputs list so
  neither the original worker outputs nor the shared
  ``EMPTY_MODEL_RUNNER_OUTPUT`` singleton is ever mutated.

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

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.core.sched.ownership import OwnerReceipt, OwnerReceiptBatch
from vllm.v1.outputs import ModelRunnerOutput

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
    """

    def __init__(
        self,
        expected_owner_ranks: list[int],
        kv_aggregator: KVOutputAggregator | None = None,
    ) -> None:
        self._expected_owner_ranks: list[int] = sorted(set(expected_owner_ranks))
        self._kv_aggregator = kv_aggregator

    def for_step(self, step_seq: int) -> "ModelRunnerOutputAggregatorStepAdapter":
        """Return an immutable, stateless per-step adapter bound to ``step_seq``.

        ``step_seq`` must be a nonnegative non-bool ``int``; anything else
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
        """
        if not outputs:
            raise RuntimeError(
                "ModelRunnerOutputAggregator.aggregate() requires at least one "
                "worker output."
            )

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

        # Shallow copy before mutation: the selected output may be the shared
        # EMPTY_MODEL_RUNNER_OUTPUT singleton or a worker-owned object.
        result = copy(selected)
        result.owner_receipt_batches = batches
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
        for output in outputs:
            if output is None:
                continue
            for batch in output.owner_receipt_batches or ():
                if batch.owner_rank not in self._expected_owner_ranks:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: unexpected owner rank "
                        f"{batch.owner_rank} (expected "
                        f"{self._expected_owner_ranks})."
                    )
                if batch.owner_rank in batches_by_rank:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: duplicate OwnerReceiptBatch "
                        f"for owner rank {batch.owner_rank}."
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
                if event.owner_id != batch.owner_rank:
                    raise RuntimeError(
                        "ModelRunnerOutputAggregator: OwnerReceipt owner_id "
                        f"{event.owner_id} does not match enclosing batch "
                        f"owner_rank {batch.owner_rank} (request_id="
                        f"{event.key.request_id!r}, command_seq="
                        f"{event.command_seq})."
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
                        free_capacity=batch.free_capacity,
                        resident_pages=batch.resident_pages,
                        pending_dma=batch.pending_dma,
                    )
                )
        return aggregated


class ModelRunnerOutputAggregatorStepAdapter:
    """Immutable per-step view of a shared :class:`ModelRunnerOutputAggregator`.

    Binds one exact ``step_seq``: every :meth:`aggregate` call delegates to
    the shared aggregator with ``expected_step_seq`` set to that step, so a
    stale or future worker emission fails closed.  The adapter is frozen
    after construction: ``_aggregator`` and ``_step_seq`` cannot be
    reassigned, no new attributes can be added, and ``step_seq`` must be a
    nonnegative non-bool ``int``.  Creating or using it never mutates the
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
        if step_seq < 0:
            raise ValueError(
                "ModelRunnerOutputAggregatorStepAdapter: step_seq must be "
                f"nonnegative, got {step_seq}."
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
