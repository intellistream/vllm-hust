# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-ID-free request-owned restore contracts and demand evidence.

The scheduler may describe *what* must become usable and *when*, but it must
never learn owner-local block IDs or packed byte geometry.  The public types in
this module are therefore safe on the scheduler/worker control wire.  Exact
destinations and canonical spans live in ``vllm.v1.worker`` instead.

These contracts are deliberately policy-free.  In particular, logical token
scale is retained only as a separately named diagnostic proxy; it can never be
substituted for an observed owner-private restore demand.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

RESTORE_INTENT_SCHEMA = "request-owned-restore-intent/v1"
RESTORE_CERTIFICATE_SCHEMA = "request-owned-restore-certificate/v1"
RESTORE_DEMAND_RECEIPT_SCHEMA = "request-owned-restore-demand/v1"
RESTORE_DEMAND_AGGREGATE_SCHEMA = "request-owned-restore-demand-aggregate/v1"

# A regression oracle for every object that may cross the scheduler wire.
# The worker-private plan intentionally uses several of these names.
RESTORE_SCHEDULER_WIRE_FORBIDDEN_FIELDS = frozenset(
    {
        "allocation_generation",
        "aliases",
        "block_ids",
        "block_size_tokens",
        "block_stride_bytes",
        "canonical_span",
        "destination_block_ids",
        "device_block_ids",
        "geometry_fingerprint",
        "local_block_ids",
        "local_destination_block_ids",
        "offset_bytes",
        "packed_geometry_fingerprint",
        "page_bytes",
        "plan_seq",
        "runtime_num_blocks",
    }
)


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise TypeError(f"{name} must be a {qualifier} non-bool int, got {value!r}.")
    return value


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string, got {value!r}.")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode one contract value deterministically, without NaN spellings."""

    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return json.dumps(
        _json_value(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class RestorePhase(str, Enum):
    PREFILL = "prefill"
    DECODE_RESUME = "decode-resume"


class RestoreCertificateStatus(str, Enum):
    COLD = "COLD"
    RESTORING = "RESTORING"
    HOT = "HOT"
    FAILED = "FAILED"
    RELEASED = "RELEASED"


@dataclass(frozen=True, slots=True)
class RestoreIntent:
    """Scheduler-authored logical restore request.

    ``activation_generation`` is the scheduler's monotonic incarnation fence
    for repeated cold-to-hot activations of the same owner lease.  It is not a
    worker allocation generation and carries no physical-layout authority.
    """

    request_uid: str
    owner_rank: int
    owner_epoch: int
    activation_generation: int
    phase: RestorePhase
    required_token_extent: int
    valid_prefix_token_extent: int
    first_consume_step: int
    max_wait_steps: int
    urgency_class: str
    policy_reason: str
    schema: str = RESTORE_INTENT_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_string("request_uid", self.request_uid)
        _require_int("owner_rank", self.owner_rank)
        _require_int("owner_epoch", self.owner_epoch)
        _require_int("activation_generation", self.activation_generation, minimum=1)
        if not isinstance(self.phase, RestorePhase):
            raise TypeError(f"phase must be a RestorePhase, got {self.phase!r}.")
        _require_int("required_token_extent", self.required_token_extent)
        _require_int("valid_prefix_token_extent", self.valid_prefix_token_extent)
        if self.valid_prefix_token_extent > self.required_token_extent:
            raise ValueError(
                "valid_prefix_token_extent must not exceed required_token_extent, "
                f"got {self.valid_prefix_token_extent} > "
                f"{self.required_token_extent}."
            )
        _require_int("first_consume_step", self.first_consume_step, minimum=1)
        _require_int("max_wait_steps", self.max_wait_steps)
        _require_nonempty_string("urgency_class", self.urgency_class)
        _require_nonempty_string("policy_reason", self.policy_reason)
        if self.schema != RESTORE_INTENT_SCHEMA:
            raise ValueError(
                f"unsupported restore intent schema {self.schema!r}; "
                f"expected {RESTORE_INTENT_SCHEMA!r}."
            )

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (
            self.request_uid,
            self.owner_rank,
            self.owner_epoch,
            self.activation_generation,
        )

    def to_wire_dict(self) -> dict[str, object]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class RestoreCertificate:
    """Owner-authored, block-ID-free restore state for one activation."""

    request_uid: str
    owner_rank: int
    owner_epoch: int
    activation_generation: int
    required_blocks: int
    reserved_blocks: int
    restoring_blocks: int
    hot_blocks: int
    landing_hot_watermark: int
    tail_hot_watermark: int
    scheduled_bytes: int
    completed_bytes: int
    deadline_miss_count: int
    status: RestoreCertificateStatus
    failure_reason: str | None = None
    schema: str = RESTORE_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_string("request_uid", self.request_uid)
        _require_int("owner_rank", self.owner_rank)
        _require_int("owner_epoch", self.owner_epoch)
        _require_int("activation_generation", self.activation_generation, minimum=1)
        for name in (
            "required_blocks",
            "reserved_blocks",
            "restoring_blocks",
            "hot_blocks",
            "landing_hot_watermark",
            "tail_hot_watermark",
            "scheduled_bytes",
            "completed_bytes",
            "deadline_miss_count",
        ):
            _require_int(name, getattr(self, name))
        if self.required_blocks > self.reserved_blocks:
            raise ValueError("required_blocks must not exceed reserved_blocks")
        if self.restoring_blocks + self.hot_blocks > self.reserved_blocks:
            raise ValueError("restoring plus HOT blocks must fit the reservation")
        if self.landing_hot_watermark > self.hot_blocks:
            raise ValueError("landing HOT watermark must not exceed hot_blocks")
        if self.tail_hot_watermark > self.hot_blocks:
            raise ValueError("tail HOT watermark must not exceed hot_blocks")
        if self.completed_bytes > self.scheduled_bytes:
            raise ValueError("completed_bytes must not exceed scheduled_bytes")
        if not isinstance(self.status, RestoreCertificateStatus):
            raise TypeError(
                f"status must be a RestoreCertificateStatus, got {self.status!r}."
            )
        if self.status is RestoreCertificateStatus.HOT:
            if self.restoring_blocks != 0:
                raise ValueError("a HOT certificate cannot retain restoring blocks")
            if self.hot_blocks < self.required_blocks:
                raise ValueError("a HOT certificate must cover every required block")
            if self.completed_bytes != self.scheduled_bytes:
                raise ValueError("a HOT certificate requires exact byte completion")
        elif self.status is RestoreCertificateStatus.RESTORING:
            if self.required_blocks > self.hot_blocks and self.restoring_blocks == 0:
                raise ValueError("RESTORING requires an outstanding restore block")
        elif self.status is RestoreCertificateStatus.COLD:
            if self.restoring_blocks or self.hot_blocks:
                raise ValueError("COLD cannot claim restoring or HOT blocks")
        elif self.status is RestoreCertificateStatus.RELEASED:
            if self.reserved_blocks or self.restoring_blocks or self.hot_blocks:
                raise ValueError("RELEASED cannot retain live block counts")
        if self.status is RestoreCertificateStatus.FAILED:
            _require_nonempty_string("failure_reason", self.failure_reason)
        elif self.failure_reason is not None:
            raise ValueError("only a FAILED certificate may carry failure_reason")
        if self.schema != RESTORE_CERTIFICATE_SCHEMA:
            raise ValueError(
                f"unsupported restore certificate schema {self.schema!r}; "
                f"expected {RESTORE_CERTIFICATE_SCHEMA!r}."
            )

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (
            self.request_uid,
            self.owner_rank,
            self.owner_epoch,
            self.activation_generation,
        )

    def certifies(self, intent: RestoreIntent) -> bool:
        """Return only the explicit dispatch predicate; never infer from counts."""

        if not isinstance(intent, RestoreIntent):
            raise TypeError(f"intent must be a RestoreIntent, got {intent!r}.")
        return (
            self.identity == intent.identity
            and self.status is RestoreCertificateStatus.HOT
        )

    def to_wire_dict(self) -> dict[str, object]:
        return _json_value(asdict(self))


class RestoreDeadlineGroup(str, Enum):
    LANDING = "landing"
    TAIL = "tail"


@dataclass(frozen=True, slots=True)
class RestoreDemandJobReceipt:
    """Block-ID-free observation of one owner-private plan job."""

    group_index: int
    deadline_group: RestoreDeadlineGroup
    effective_tokens_per_block: int
    valid_token_extents: tuple[int, ...]
    blocks: int
    scheduled_bytes: int
    completed_bytes: int
    scheduled_step: int
    completed_step: int | None

    def __post_init__(self) -> None:
        _require_int("group_index", self.group_index)
        if not isinstance(self.deadline_group, RestoreDeadlineGroup):
            raise TypeError("deadline_group must be a RestoreDeadlineGroup")
        _require_int(
            "effective_tokens_per_block",
            self.effective_tokens_per_block,
            minimum=1,
        )
        if not isinstance(self.valid_token_extents, tuple):
            raise TypeError("valid_token_extents must be a tuple")
        _require_int("blocks", self.blocks, minimum=1)
        if len(self.valid_token_extents) != self.blocks:
            raise ValueError("every demand block must retain one valid token extent")
        for extent in self.valid_token_extents:
            _require_int("valid token extent", extent, minimum=1)
            if extent > self.effective_tokens_per_block:
                raise ValueError("valid token extent exceeds the effective block size")
        _require_int("scheduled_bytes", self.scheduled_bytes, minimum=1)
        _require_int("completed_bytes", self.completed_bytes)
        _require_int("scheduled_step", self.scheduled_step, minimum=1)
        if self.completed_step is not None:
            _require_int("completed_step", self.completed_step, minimum=1)
            if self.completed_step < self.scheduled_step:
                raise ValueError("completed_step cannot precede scheduled_step")
        if self.completed_bytes > self.scheduled_bytes:
            raise ValueError("job completed_bytes must not exceed scheduled_bytes")
        if self.completed_step is None and self.completed_bytes:
            raise ValueError("an incomplete job cannot claim completed bytes")


@dataclass(frozen=True, slots=True)
class RestoreDemandReceipt:
    """One deterministic, block-ID-free activation demand receipt.

    A zero-demand activation is represented by an ordinary receipt with
    ``newly_restored_blocks == 0`` and empty ``jobs``.  It must not be omitted
    from aggregation.
    """

    request_uid: str
    owner_rank: int
    owner_epoch: int
    activation_generation: int
    phase: RestorePhase
    wave_id: str
    source_provenance: str
    workload_provenance: str
    required_blocks: int
    resident_blocks: int
    host_only_blocks: int
    restoring_blocks: int
    newly_restored_blocks: int
    logical_128_token_units_proxy: int | None
    final_footprint_reserved_blocks: int
    jobs: tuple[RestoreDemandJobReceipt, ...]
    wait_steps: int
    deadline_miss_reason: str | None
    terminal_status: RestoreCertificateStatus
    terminal_reason: str | None = None
    observed_start_ns: int | None = None
    observed_end_ns: int | None = None
    schema: str = RESTORE_DEMAND_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_string("request_uid", self.request_uid)
        _require_int("owner_rank", self.owner_rank)
        _require_int("owner_epoch", self.owner_epoch)
        _require_int("activation_generation", self.activation_generation, minimum=1)
        if not isinstance(self.phase, RestorePhase):
            raise TypeError("phase must be a RestorePhase")
        _require_nonempty_string("wave_id", self.wave_id)
        _require_nonempty_string("source_provenance", self.source_provenance)
        _require_nonempty_string("workload_provenance", self.workload_provenance)
        for name in (
            "required_blocks",
            "resident_blocks",
            "host_only_blocks",
            "restoring_blocks",
            "newly_restored_blocks",
            "final_footprint_reserved_blocks",
            "wait_steps",
        ):
            _require_int(name, getattr(self, name))
        if self.logical_128_token_units_proxy is not None:
            _require_int(
                "logical_128_token_units_proxy",
                self.logical_128_token_units_proxy,
            )
        if self.required_blocks > self.final_footprint_reserved_blocks:
            raise ValueError("required demand must fit the final reservation")
        if self.resident_blocks + self.host_only_blocks < self.required_blocks:
            raise ValueError("resident plus host-only facts must cover required blocks")
        if self.newly_restored_blocks > self.host_only_blocks:
            raise ValueError("newly restored blocks cannot exceed host-only blocks")
        if not isinstance(self.jobs, tuple):
            raise TypeError("jobs must be a tuple")
        for job in self.jobs:
            if not isinstance(job, RestoreDemandJobReceipt):
                raise TypeError("jobs must contain RestoreDemandJobReceipt values")
        indices = [job.group_index for job in self.jobs]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("demand jobs must have unique sorted group indices")
        if sum(job.blocks for job in self.jobs) != self.newly_restored_blocks:
            raise ValueError("job block counts must equal newly_restored_blocks")
        if not isinstance(self.terminal_status, RestoreCertificateStatus):
            raise TypeError("terminal_status must be a RestoreCertificateStatus")
        if self.terminal_status is RestoreCertificateStatus.HOT:
            if self.restoring_blocks:
                raise ValueError("a HOT demand receipt cannot retain restoring blocks")
            if any(
                job.completed_bytes != job.scheduled_bytes or job.completed_step is None
                for job in self.jobs
            ):
                raise ValueError(
                    "a HOT demand receipt requires exact completion of every job"
                )
        if self.terminal_status is RestoreCertificateStatus.FAILED:
            _require_nonempty_string("terminal_reason", self.terminal_reason)
        elif self.terminal_reason is not None:
            raise ValueError("only FAILED demand may carry terminal_reason")
        if self.deadline_miss_reason is not None:
            _require_nonempty_string("deadline_miss_reason", self.deadline_miss_reason)
        for name in ("observed_start_ns", "observed_end_ns"):
            value = getattr(self, name)
            if value is not None:
                _require_int(name, value)
        if (
            self.observed_start_ns is not None
            and self.observed_end_ns is not None
            and self.observed_end_ns < self.observed_start_ns
        ):
            raise ValueError("observed_end_ns cannot precede observed_start_ns")
        if self.schema != RESTORE_DEMAND_RECEIPT_SCHEMA:
            raise ValueError(
                f"unsupported demand receipt schema {self.schema!r}; expected "
                f"{RESTORE_DEMAND_RECEIPT_SCHEMA!r}."
            )

    @property
    def scheduled_bytes(self) -> int:
        return sum(job.scheduled_bytes for job in self.jobs)

    @property
    def completed_bytes(self) -> int:
        return sum(job.completed_bytes for job in self.jobs)

    def canonical_bytes(self, *, include_timing: bool = False) -> bytes:
        payload = asdict(self)
        if not include_timing:
            payload.pop("observed_start_ns")
            payload.pop("observed_end_ns")
        return canonical_json_bytes(payload)


@dataclass(frozen=True, slots=True)
class DemandDistribution:
    count: int
    zero_count: int
    total: int
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int


def _nearest_rank(values: Iterable[int]) -> DemandDistribution:
    ordered = sorted(values)
    if not ordered:
        return DemandDistribution(0, 0, 0, 0, 0, 0, 0, 0)

    def percentile(percent: int) -> int:
        return ordered[max(0, math.ceil((percent / 100) * len(ordered)) - 1)]

    return DemandDistribution(
        count=len(ordered),
        zero_count=sum(value == 0 for value in ordered),
        total=sum(ordered),
        p50=percentile(50),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
        maximum=ordered[-1],
    )


def aggregate_restore_demand(
    receipts: Iterable[RestoreDemandReceipt],
) -> dict[str, object]:
    """Aggregate actual activation and per-wave/per-rank demand.

    Wave/rank values are first summed across activations sharing
    ``(phase, wave_id, owner_rank)`` and only then percentile-aggregated.  The
    input must contain the zero-demand activation receipts that are in scope;
    they remain ordinary zeros in both distributions.
    """

    materialized = tuple(receipts)
    if any(not isinstance(item, RestoreDemandReceipt) for item in materialized):
        raise TypeError("receipts must contain RestoreDemandReceipt values")
    identities = [
        (
            item.request_uid,
            item.owner_rank,
            item.owner_epoch,
            item.activation_generation,
        )
        for item in materialized
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("restore demand contains duplicate activation receipts")

    activation_blocks: dict[RestorePhase, list[int]] = defaultdict(list)
    activation_bytes: dict[RestorePhase, list[int]] = defaultdict(list)
    wave_rank: dict[tuple[RestorePhase, str, int], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    for item in materialized:
        activation_blocks[item.phase].append(item.newly_restored_blocks)
        activation_bytes[item.phase].append(item.scheduled_bytes)
        row = wave_rank[(item.phase, item.wave_id, item.owner_rank)]
        row[0] += item.newly_restored_blocks
        row[1] += item.scheduled_bytes
        row[2] += int(item.deadline_miss_reason is not None)

    rows = [
        {
            "phase": phase.value,
            "wave_id": wave_id,
            "owner_rank": owner_rank,
            "newly_restored_blocks": values[0],
            "scheduled_bytes": values[1],
            "deadline_miss_count": values[2],
        }
        for (phase, wave_id, owner_rank), values in sorted(
            wave_rank.items(),
            key=lambda item: (item[0][0].value, item[0][1], item[0][2]),
        )
    ]
    wave_blocks: dict[RestorePhase, list[int]] = defaultdict(list)
    wave_bytes: dict[RestorePhase, list[int]] = defaultdict(list)
    rank_blocks: dict[tuple[RestorePhase, int], list[int]] = defaultdict(list)
    rank_bytes: dict[tuple[RestorePhase, int], list[int]] = defaultdict(list)
    wave_values: dict[tuple[RestorePhase, str], list[tuple[int, int, int]]] = (
        defaultdict(list)
    )
    for row in rows:
        phase = RestorePhase(row["phase"])
        wave_blocks[phase].append(row["newly_restored_blocks"])
        wave_bytes[phase].append(row["scheduled_bytes"])
        owner_rank = row["owner_rank"]
        rank_blocks[(phase, owner_rank)].append(row["newly_restored_blocks"])
        rank_bytes[(phase, owner_rank)].append(row["scheduled_bytes"])
        wave_values[(phase, row["wave_id"])].append(
            (
                owner_rank,
                row["newly_restored_blocks"],
                row["deadline_miss_count"],
            )
        )

    def distributions(
        blocks: dict[RestorePhase, list[int]],
        byte_values: dict[RestorePhase, list[int]],
    ) -> list[dict[str, object]]:
        return [
            {
                "phase": phase.value,
                "blocks": _json_value(asdict(_nearest_rank(blocks[phase]))),
                "bytes": _json_value(asdict(_nearest_rank(byte_values[phase]))),
            }
            for phase in sorted(blocks, key=lambda item: item.value)
        ]

    return {
        "schema": RESTORE_DEMAND_AGGREGATE_SCHEMA,
        "activation_count": len(materialized),
        "activation_distributions": distributions(activation_blocks, activation_bytes),
        "wave_rank_rows": rows,
        "wave_rank_distributions": distributions(wave_blocks, wave_bytes),
        "rank_distributions": [
            {
                "phase": phase.value,
                "owner_rank": owner_rank,
                "blocks": _json_value(
                    asdict(_nearest_rank(rank_blocks[(phase, owner_rank)]))
                ),
                "bytes": _json_value(
                    asdict(_nearest_rank(rank_bytes[(phase, owner_rank)]))
                ),
            }
            for phase, owner_rank in sorted(
                rank_blocks, key=lambda item: (item[0].value, item[1])
            )
        ],
        "wave_rows": [
            {
                "phase": phase.value,
                "wave_id": wave_id,
                "active_owner_ranks": [item[0] for item in values],
                "median_rank_demand_blocks": _nearest_rank(
                    item[1] for item in values
                ).p50,
                "max_rank_demand_blocks": max(item[1] for item in values),
                "owner_skew_blocks": max(item[1] for item in values)
                - min(item[1] for item in values),
                "deadline_miss_count": sum(item[2] for item in values),
            }
            for (phase, wave_id), values in sorted(
                wave_values.items(), key=lambda item: (item[0][0].value, item[0][1])
            )
        ],
    }
