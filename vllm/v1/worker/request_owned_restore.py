# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-private request-owned restore geometry, plans, and receipts.

This module is the physical half of ``RestoreIntent/v1``.  It may carry
rank-local block IDs and packed offsets, so its values must never be attached
to scheduler output.  Plans are deterministic, geometry-fingerprinted, and
fail closed when actual demand is unknown.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256

from vllm.v1.core.sched.restore_contract import (
    RestoreCertificate,
    RestoreCertificateStatus,
    RestoreDeadlineGroup,
    RestoreDemandJobReceipt,
    RestoreDemandReceipt,
    RestoreIntent,
    canonical_json_bytes,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.request_owned_kv import (
    RequestOwnedKVSnapshot,
    request_owned_effective_tokens_per_block,
)

RESTORE_PLAN_SCHEMA = "request-owned-restore-plan/v1"


class RequestOwnedRestoreError(RuntimeError):
    """Fail-closed owner-private restore contract violation."""


class RestoreDemandUnknownError(RequestOwnedRestoreError):
    """Raised when logical scale is offered in place of actual demand."""


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise TypeError(f"{name} must be a {qualifier} non-bool int, got {value!r}.")
    return value


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string, got {value!r}.")
    return value


@dataclass(frozen=True, slots=True)
class CanonicalRestoreSlice:
    """One exact physical slice, with all logical aliases named once."""

    offset_bytes: int
    page_bytes: int
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_int("offset_bytes", self.offset_bytes)
        _require_int("page_bytes", self.page_bytes, minimum=1)
        if not isinstance(self.aliases, tuple) or not self.aliases:
            raise TypeError("aliases must be a nonempty tuple")
        if any(not isinstance(alias, str) or not alias for alias in self.aliases):
            raise TypeError("aliases must contain nonempty strings")
        if tuple(sorted(set(self.aliases))) != self.aliases:
            raise ValueError("aliases must be unique and sorted")


@dataclass(frozen=True, slots=True)
class CanonicalRestoreSpan:
    """Contiguous group-qualified transfer span and its canonical slices."""

    offset_bytes: int
    page_bytes: int
    aliases: tuple[str, ...]
    slices: tuple[CanonicalRestoreSlice, ...]

    def __post_init__(self) -> None:
        _require_int("offset_bytes", self.offset_bytes)
        _require_int("page_bytes", self.page_bytes, minimum=1)
        if not isinstance(self.slices, tuple) or not self.slices:
            raise TypeError("slices must be a nonempty tuple")
        prior_end = self.offset_bytes
        all_aliases: list[str] = []
        seen_physical: set[tuple[int, int]] = set()
        span_end = self.offset_bytes + self.page_bytes
        for item in self.slices:
            if not isinstance(item, CanonicalRestoreSlice):
                raise TypeError("slices must contain CanonicalRestoreSlice values")
            physical = (item.offset_bytes, item.page_bytes)
            if physical in seen_physical:
                raise ValueError("a canonical physical slice appears more than once")
            seen_physical.add(physical)
            if item.offset_bytes != prior_end:
                raise ValueError(
                    "canonical slices must exactly tile the contiguous group span"
                )
            if item.offset_bytes < self.offset_bytes or (
                item.offset_bytes + item.page_bytes > span_end
            ):
                raise ValueError("canonical slice lies outside its group span")
            prior_end = item.offset_bytes + item.page_bytes
            all_aliases.extend(item.aliases)
        if prior_end != span_end:
            raise ValueError(
                "canonical slices must exactly tile the contiguous group span"
            )
        if tuple(sorted(set(all_aliases))) != self.aliases:
            raise ValueError("span aliases must equal the exact slice alias union")
        if len(all_aliases) != len(set(all_aliases)):
            raise ValueError("a logical alias appears in multiple canonical slices")


@dataclass(frozen=True, slots=True)
class RestoreGroupGeometry:
    group_index: int
    effective_tokens_per_block: int
    canonical_span: CanonicalRestoreSpan

    def __post_init__(self) -> None:
        _require_int("group_index", self.group_index)
        _require_int(
            "effective_tokens_per_block",
            self.effective_tokens_per_block,
            minimum=1,
        )
        if not isinstance(self.canonical_span, CanonicalRestoreSpan):
            raise TypeError("canonical_span must be a CanonicalRestoreSpan")


@dataclass(frozen=True, slots=True)
class PackedRestoreGeometry:
    """Live packed geometry used to fence every owner-private plan."""

    block_size_tokens: int
    block_stride_bytes: int
    runtime_num_blocks: int
    groups: tuple[RestoreGroupGeometry, ...]

    def __post_init__(self) -> None:
        _require_int("block_size_tokens", self.block_size_tokens, minimum=1)
        _require_int("block_stride_bytes", self.block_stride_bytes, minimum=1)
        _require_int("runtime_num_blocks", self.runtime_num_blocks, minimum=1)
        if not isinstance(self.groups, tuple) or not self.groups:
            raise TypeError("groups must be a nonempty tuple")
        for group in self.groups:
            if not isinstance(group, RestoreGroupGeometry):
                raise TypeError("groups must contain RestoreGroupGeometry values")
            span = group.canonical_span
            if span.offset_bytes + span.page_bytes > self.block_stride_bytes:
                raise ValueError("group canonical span exceeds the packed stride")
        indices = tuple(group.group_index for group in self.groups)
        if indices != tuple(range(len(indices))):
            raise ValueError("restore geometry groups must be dense and ordered")

    @property
    def fingerprint(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()

    @classmethod
    def from_kv_cache_config(
        cls,
        config: KVCacheConfig,
        *,
        block_size_tokens: int,
    ) -> PackedRestoreGeometry:
        """Derive canonical group spans from the live packed builder output."""

        if not isinstance(config, KVCacheConfig):
            raise TypeError(f"config must be a KVCacheConfig, got {config!r}.")
        _require_int("block_size_tokens", block_size_tokens, minimum=1)
        if not config.kv_cache_tensors:
            raise RequestOwnedRestoreError("packed geometry has no KV tensors")
        strides = {item.block_stride for item in config.kv_cache_tensors}
        if 0 in strides or len(strides) != 1:
            raise RequestOwnedRestoreError(
                "restore planning requires one uniform nonzero packed stride"
            )
        (stride,) = strides
        if any(
            item.size != config.num_blocks * stride for item in config.kv_cache_tensors
        ):
            raise RequestOwnedRestoreError(
                "packed KV tensors must name the exact runtime backing size"
            )

        layer_specs: dict[str, KVCacheSpec] = {}
        group_layers: list[set[str]] = []
        assigned_layers: set[str] = set()
        for group in config.kv_cache_groups:
            names = set(group.layer_names)
            if len(names) != len(group.layer_names):
                raise RequestOwnedRestoreError("KV group contains duplicate layers")
            duplicate_assignments = names & assigned_layers
            if duplicate_assignments:
                raise RequestOwnedRestoreError(
                    "KV layers must belong to exactly one restore group, got "
                    f"duplicates {sorted(duplicate_assignments)!r}"
                )
            assigned_layers.update(names)
            group_layers.append(names)
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                if set(spec.kv_cache_specs) != names:
                    raise RequestOwnedRestoreError(
                        "uniform KV group specs must exactly cover group layers"
                    )
                for name, layer_spec in spec.kv_cache_specs.items():
                    layer_specs[name] = layer_spec
            else:
                for name in names:
                    layer_specs[name] = spec

        descriptors: list[tuple[int, int, tuple[str, ...]]] = []
        for descriptor in config.kv_cache_tensors:
            aliases = tuple(sorted(set(descriptor.shared_by)))
            if not aliases:
                # Packed builders may reserve an unused slot.  It remains part
                # of the stride but has no meaningful group image.
                continue
            if len(aliases) != len(descriptor.shared_by):
                raise RequestOwnedRestoreError("packed tensor aliases are duplicated")
            try:
                page_sizes = {
                    int(layer_specs[name].page_size_bytes) for name in aliases
                }
            except KeyError as exc:
                raise RequestOwnedRestoreError(
                    f"packed tensor names an unknown layer {exc.args[0]!r}"
                ) from exc
            if len(page_sizes) != 1:
                raise RequestOwnedRestoreError(
                    "all aliases of one packed tensor must share page size"
                )
            (page_bytes,) = page_sizes
            descriptors.append((descriptor.offset, page_bytes, aliases))
        descriptors.sort(key=lambda item: (item[0], item[1], item[2]))
        prior_end = 0
        for offset, page_bytes, _ in descriptors:
            if offset < prior_end:
                raise RequestOwnedRestoreError(
                    "packed physical descriptors must be canonical, ordered, "
                    "and non-overlapping; aliases belong in one shared descriptor"
                )
            prior_end = offset + page_bytes

        geometries: list[RestoreGroupGeometry] = []
        for group_index, (group, names) in enumerate(
            zip(config.kv_cache_groups, group_layers)
        ):
            slices: list[CanonicalRestoreSlice] = []
            covered: list[str] = []
            for offset, page_bytes, aliases in descriptors:
                local_aliases = tuple(alias for alias in aliases if alias in names)
                if not local_aliases:
                    continue
                slices.append(
                    CanonicalRestoreSlice(
                        offset_bytes=offset,
                        page_bytes=page_bytes,
                        aliases=local_aliases,
                    )
                )
                covered.extend(local_aliases)
            if set(covered) != names or len(covered) != len(names):
                raise RequestOwnedRestoreError(
                    f"group {group_index} aliases are not covered exactly once"
                )
            start = min(item.offset_bytes for item in slices)
            end = max(item.offset_bytes + item.page_bytes for item in slices)
            effective_tokens = request_owned_effective_tokens_per_block(
                group.kv_cache_spec
            )
            geometries.append(
                RestoreGroupGeometry(
                    group_index=group_index,
                    effective_tokens_per_block=effective_tokens,
                    canonical_span=CanonicalRestoreSpan(
                        offset_bytes=start,
                        page_bytes=end - start,
                        aliases=tuple(sorted(names)),
                        slices=tuple(slices),
                    ),
                )
            )
        return cls(
            block_size_tokens=block_size_tokens,
            block_stride_bytes=stride,
            runtime_num_blocks=config.num_blocks,
            groups=tuple(geometries),
        )


class RestoreChunkKind(str, Enum):
    GROUP_FULL_PAGE = "group_full_page"


@dataclass(frozen=True, slots=True)
class RestorePlanJob:
    local_destination_block_ids: tuple[int, ...]
    group_index: int
    effective_tokens_per_block: int
    valid_token_extents: tuple[int, ...]
    chunk_kind: RestoreChunkKind
    canonical_span: CanonicalRestoreSpan
    deadline_group: RestoreDeadlineGroup
    expected_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.local_destination_block_ids, tuple):
            raise TypeError("local_destination_block_ids must be a tuple")
        if not self.local_destination_block_ids:
            raise ValueError("a restore plan job must contain a destination")
        for block_id in self.local_destination_block_ids:
            _require_int("local destination block id", block_id, minimum=1)
        if len(set(self.local_destination_block_ids)) != len(
            self.local_destination_block_ids
        ):
            raise ValueError("a restore job contains duplicate destination IDs")
        _require_int("group_index", self.group_index)
        _require_int(
            "effective_tokens_per_block",
            self.effective_tokens_per_block,
            minimum=1,
        )
        if not isinstance(self.valid_token_extents, tuple):
            raise TypeError("valid_token_extents must be a tuple")
        if len(self.valid_token_extents) != len(self.local_destination_block_ids):
            raise ValueError("every destination must have one valid token extent")
        for extent in self.valid_token_extents:
            _require_int("valid token extent", extent, minimum=1)
            if extent > self.effective_tokens_per_block:
                raise ValueError("valid token extent exceeds the group block extent")
        if self.chunk_kind is not RestoreChunkKind.GROUP_FULL_PAGE:
            raise ValueError("A0 accepts only group_full_page restore jobs")
        if not isinstance(self.canonical_span, CanonicalRestoreSpan):
            raise TypeError("canonical_span must be a CanonicalRestoreSpan")
        if not isinstance(self.deadline_group, RestoreDeadlineGroup):
            raise TypeError("deadline_group must be a RestoreDeadlineGroup")
        _require_int("expected_bytes", self.expected_bytes)
        exact = len(self.local_destination_block_ids) * self.canonical_span.page_bytes
        if self.expected_bytes != exact:
            raise ValueError(
                "job expected_bytes must use canonical span bytes, got "
                f"{self.expected_bytes} != {exact}"
            )


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """Exact owner-private plan for one scheduler activation."""

    request_uid: str
    owner_rank: int
    owner_epoch: int
    activation_generation: int
    allocation_generation: int
    plan_seq: int
    block_size_tokens: int
    packed_geometry_fingerprint: str
    block_stride_bytes: int
    runtime_num_blocks: int
    jobs: tuple[RestorePlanJob, ...]
    reserved_final_footprint_blocks: int
    landing_required_blocks: int
    tail_required_blocks: int
    schema: str = RESTORE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        _require_nonempty_string("request_uid", self.request_uid)
        for name in ("owner_rank", "owner_epoch"):
            _require_int(name, getattr(self, name))
        for name in (
            "activation_generation",
            "allocation_generation",
            "plan_seq",
            "block_size_tokens",
            "block_stride_bytes",
            "runtime_num_blocks",
        ):
            _require_int(name, getattr(self, name), minimum=1)
        if (
            not isinstance(self.packed_geometry_fingerprint, str)
            or len(self.packed_geometry_fingerprint) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.packed_geometry_fingerprint
            )
        ):
            raise ValueError("packed_geometry_fingerprint must be lowercase SHA-256")
        if not isinstance(self.jobs, tuple):
            raise TypeError("jobs must be a tuple")
        ids: list[int] = []
        indices: list[int] = []
        for job in self.jobs:
            if not isinstance(job, RestorePlanJob):
                raise TypeError("jobs must contain RestorePlanJob values")
            indices.append(job.group_index)
            ids.extend(job.local_destination_block_ids)
            if job.canonical_span.offset_bytes + job.canonical_span.page_bytes > (
                self.block_stride_bytes
            ):
                raise ValueError("job canonical span exceeds packed stride")
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("plan jobs must have unique sorted group indices")
        if len(set(ids)) != len(ids):
            raise ValueError("local destination IDs must be disjoint across groups")
        if any(block_id >= self.runtime_num_blocks for block_id in ids):
            raise ValueError("local destination block ID is outside runtime geometry")
        for name in (
            "reserved_final_footprint_blocks",
            "landing_required_blocks",
            "tail_required_blocks",
        ):
            _require_int(name, getattr(self, name))
        if self.reserved_final_footprint_blocks < len(ids):
            raise ValueError("final footprint must be reserved before restore")
        if self.landing_required_blocks + self.tail_required_blocks != len(ids):
            raise ValueError("landing plus tail counts must cover every plan ID")
        if self.schema != RESTORE_PLAN_SCHEMA:
            raise ValueError(
                f"unsupported restore plan schema {self.schema!r}; expected "
                f"{RESTORE_PLAN_SCHEMA!r}."
            )

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (
            self.request_uid,
            self.owner_rank,
            self.owner_epoch,
            self.activation_generation,
        )

    @property
    def total_destination_ids(self) -> int:
        return sum(len(job.local_destination_block_ids) for job in self.jobs)

    @property
    def expected_bytes(self) -> int:
        return sum(job.expected_bytes for job in self.jobs)

    def assert_current(
        self,
        *,
        intent: RestoreIntent,
        allocation_generation: int,
        plan_seq: int,
        geometry: PackedRestoreGeometry,
    ) -> None:
        """Fence owner, epoch, activation, allocation, plan, and geometry."""

        if not isinstance(intent, RestoreIntent):
            raise TypeError("intent must be a RestoreIntent")
        _require_int("allocation_generation", allocation_generation, minimum=1)
        _require_int("plan_seq", plan_seq, minimum=1)
        if not isinstance(geometry, PackedRestoreGeometry):
            raise TypeError("geometry must be a PackedRestoreGeometry")
        observed = (
            intent.identity,
            allocation_generation,
            plan_seq,
            geometry.fingerprint,
        )
        expected = (
            self.identity,
            self.allocation_generation,
            self.plan_seq,
            self.packed_geometry_fingerprint,
        )
        plan_geometry_matches = (
            self.block_size_tokens == geometry.block_size_tokens
            and self.block_stride_bytes == geometry.block_stride_bytes
            and self.runtime_num_blocks == geometry.runtime_num_blocks
            and all(
                job.group_index < len(geometry.groups)
                and job.effective_tokens_per_block
                == geometry.groups[job.group_index].effective_tokens_per_block
                and job.canonical_span
                == geometry.groups[job.group_index].canonical_span
                for job in self.jobs
            )
        )
        if observed != expected or not plan_geometry_matches:
            raise RequestOwnedRestoreError(
                "stale restore owner/epoch/activation/allocation/plan/geometry fence"
            )

    def assert_bounded_correctness_scope(self, *, max_ids: int = 4) -> None:
        _require_int("max_ids", max_ids, minimum=1)
        if self.total_destination_ids > max_ids:
            raise RequestOwnedRestoreError(
                f"first real restore plan exceeds bounded {max_ids}-ID scope"
            )


def build_group_full_page_restore_plan(
    *,
    intent: RestoreIntent,
    destination: RequestOwnedKVSnapshot,
    geometry: PackedRestoreGeometry,
    plan_seq: int,
    actual_restore_block_ids: tuple[tuple[int, ...], ...] | None,
    valid_token_extents: tuple[tuple[int, ...], ...] | None,
    deadline_groups: tuple[RestoreDeadlineGroup, ...],
    reserved_final_footprint_blocks: int,
) -> RestorePlan:
    """Build one exact plan from explicit owner-observed restore demand.

    ``None`` demand is rejected.  Logical token units, prefix length, or
    ``units * block_stride`` are intentionally not accepted as fallbacks.
    """

    if not isinstance(intent, RestoreIntent):
        raise TypeError("intent must be a RestoreIntent")
    if not isinstance(destination, RequestOwnedKVSnapshot):
        raise TypeError("destination must be a RequestOwnedKVSnapshot")
    if not isinstance(geometry, PackedRestoreGeometry):
        raise TypeError("geometry must be a PackedRestoreGeometry")
    if actual_restore_block_ids is None or valid_token_extents is None:
        raise RestoreDemandUnknownError(
            "actual group-qualified restore demand is unknown; logical scale "
            "and packed stride are not substitutes"
        )
    group_count = len(geometry.groups)
    if not (
        len(destination.tables)
        == len(actual_restore_block_ids)
        == len(valid_token_extents)
        == len(deadline_groups)
        == group_count
    ):
        raise ValueError(
            "destination, demand, extents, and deadlines must cover groups"
        )
    if (
        destination.key.request_id != intent.request_uid
        or destination.key.owner_epoch != intent.owner_epoch
        or destination.owner_rank != intent.owner_rank
    ):
        raise RequestOwnedRestoreError(
            "restore intent does not match destination owner"
        )
    if (
        destination.num_computed_tokens != intent.valid_prefix_token_extent
        or destination.reserved_num_tokens != intent.required_token_extent
        or destination.pending_free
    ):
        raise RequestOwnedRestoreError(
            "restore destination does not match the intent prefix/final footprint"
        )
    final_ids = [
        block_id for table in destination.tables for block_id in table if block_id != 0
    ]
    if len(final_ids) != len(set(final_ids)):
        raise RequestOwnedRestoreError(
            "final destination IDs must be disjoint across groups"
        )
    if reserved_final_footprint_blocks != len(final_ids):
        raise ValueError(
            "reserved final footprint must equal the live destination "
            f"allocation, got {reserved_final_footprint_blocks} != "
            f"{len(final_ids)}"
        )

    jobs: list[RestorePlanJob] = []
    for group, table, ids, extents, deadline in zip(
        geometry.groups,
        destination.tables,
        actual_restore_block_ids,
        valid_token_extents,
        deadline_groups,
    ):
        if not isinstance(ids, tuple) or not isinstance(extents, tuple):
            raise TypeError("actual restore IDs and extents must be group tuples")
        if not isinstance(deadline, RestoreDeadlineGroup):
            raise TypeError("deadline_groups must contain RestoreDeadlineGroup values")
        if any(block_id not in table for block_id in ids):
            raise RequestOwnedRestoreError(
                f"group {group.group_index} demand names a non-destination block"
            )
        if not ids:
            if extents:
                raise ValueError("a zero-demand group cannot carry token extents")
            continue
        expected_extents = tuple(
            min(
                group.effective_tokens_per_block,
                max(
                    0,
                    intent.valid_prefix_token_extent
                    - table.index(block_id) * group.effective_tokens_per_block,
                ),
            )
            for block_id in ids
        )
        if any(extent == 0 for extent in expected_extents):
            raise RequestOwnedRestoreError(
                f"group {group.group_index} demand lies outside the valid prefix"
            )
        if extents != expected_extents:
            raise RequestOwnedRestoreError(
                f"group {group.group_index} valid token extents do not match "
                "the intent prefix"
            )
        span = group.canonical_span
        jobs.append(
            RestorePlanJob(
                local_destination_block_ids=ids,
                group_index=group.group_index,
                effective_tokens_per_block=group.effective_tokens_per_block,
                valid_token_extents=extents,
                chunk_kind=RestoreChunkKind.GROUP_FULL_PAGE,
                canonical_span=span,
                deadline_group=deadline,
                expected_bytes=len(set(ids)) * span.page_bytes,
            )
        )
    landing = sum(
        len(job.local_destination_block_ids)
        for job in jobs
        if job.deadline_group is RestoreDeadlineGroup.LANDING
    )
    tail = sum(
        len(job.local_destination_block_ids)
        for job in jobs
        if job.deadline_group is RestoreDeadlineGroup.TAIL
    )
    plan = RestorePlan(
        request_uid=intent.request_uid,
        owner_rank=intent.owner_rank,
        owner_epoch=intent.owner_epoch,
        activation_generation=intent.activation_generation,
        allocation_generation=destination.allocation_generation,
        plan_seq=plan_seq,
        block_size_tokens=geometry.block_size_tokens,
        packed_geometry_fingerprint=geometry.fingerprint,
        block_stride_bytes=geometry.block_stride_bytes,
        runtime_num_blocks=geometry.runtime_num_blocks,
        jobs=tuple(jobs),
        reserved_final_footprint_blocks=reserved_final_footprint_blocks,
        landing_required_blocks=landing,
        tail_required_blocks=tail,
    )
    # The v1 factory is deliberately the first-real correctness contract,
    # not an unbounded workload planner.  A later evidence-backed schema may
    # lift this limit explicitly; callers cannot silently skip the gate.
    plan.assert_bounded_correctness_scope(max_ids=4)
    return plan


def hot_restore_certificate(
    plan: RestorePlan,
    *,
    required_blocks: int,
    deadline_miss_count: int = 0,
) -> RestoreCertificate:
    """Build a HOT certificate only after exact plan-byte completion."""

    if not isinstance(plan, RestorePlan):
        raise TypeError("plan must be a RestorePlan")
    _require_int("required_blocks", required_blocks)
    _require_int("deadline_miss_count", deadline_miss_count)
    return RestoreCertificate(
        request_uid=plan.request_uid,
        owner_rank=plan.owner_rank,
        owner_epoch=plan.owner_epoch,
        activation_generation=plan.activation_generation,
        required_blocks=required_blocks,
        reserved_blocks=plan.reserved_final_footprint_blocks,
        restoring_blocks=0,
        hot_blocks=required_blocks,
        landing_hot_watermark=plan.landing_required_blocks,
        tail_hot_watermark=plan.total_destination_ids,
        scheduled_bytes=plan.expected_bytes,
        completed_bytes=plan.expected_bytes,
        deadline_miss_count=deadline_miss_count,
        status=RestoreCertificateStatus.HOT,
    )


def terminal_restore_certificate(
    intent: RestoreIntent,
    *,
    status: RestoreCertificateStatus,
    failure_reason: str | None = None,
) -> RestoreCertificate:
    """Emit explicit FAILED or RELEASED cleanup without stale HOT facts."""

    if status not in (
        RestoreCertificateStatus.FAILED,
        RestoreCertificateStatus.RELEASED,
    ):
        raise ValueError("terminal certificate must be FAILED or RELEASED")
    return RestoreCertificate(
        request_uid=intent.request_uid,
        owner_rank=intent.owner_rank,
        owner_epoch=intent.owner_epoch,
        activation_generation=intent.activation_generation,
        required_blocks=0,
        reserved_blocks=0,
        restoring_blocks=0,
        hot_blocks=0,
        landing_hot_watermark=0,
        tail_hot_watermark=0,
        scheduled_bytes=0,
        completed_bytes=0,
        deadline_miss_count=0,
        status=status,
        failure_reason=failure_reason,
    )


def demand_receipt_for_current_plan(
    *,
    intent: RestoreIntent,
    plan: RestorePlan,
    geometry: PackedRestoreGeometry,
    current_allocation_generation: int,
    current_plan_seq: int,
    wave_id: str,
    source_provenance: str,
    workload_provenance: str,
    required_blocks: int,
    resident_blocks: int,
    host_only_blocks: int,
    restoring_blocks: int,
    logical_128_token_units_proxy: int | None,
    scheduled_step: int,
    completed_step: int | None,
    terminal_status: RestoreCertificateStatus,
    deadline_miss_reason: str | None = None,
    terminal_reason: str | None = None,
    observed_start_ns: int | None = None,
    observed_end_ns: int | None = None,
) -> RestoreDemandReceipt:
    """Validated demand receipt constructor requiring the live geometry."""

    plan.assert_current(
        intent=intent,
        allocation_generation=current_allocation_generation,
        plan_seq=current_plan_seq,
        geometry=geometry,
    )
    complete = terminal_status is RestoreCertificateStatus.HOT
    if complete and completed_step is None:
        raise ValueError("HOT demand receipt requires completed_step")
    if not complete and completed_step is not None:
        raise ValueError("non-HOT demand receipt cannot claim completed_step")
    jobs = tuple(
        RestoreDemandJobReceipt(
            group_index=job.group_index,
            deadline_group=job.deadline_group,
            effective_tokens_per_block=job.effective_tokens_per_block,
            valid_token_extents=job.valid_token_extents,
            blocks=len(job.local_destination_block_ids),
            scheduled_bytes=job.expected_bytes,
            completed_bytes=job.expected_bytes if complete else 0,
            scheduled_step=scheduled_step,
            completed_step=completed_step,
        )
        for job in plan.jobs
    )
    return RestoreDemandReceipt(
        request_uid=plan.request_uid,
        owner_rank=plan.owner_rank,
        owner_epoch=plan.owner_epoch,
        activation_generation=plan.activation_generation,
        phase=intent.phase,
        wave_id=wave_id,
        source_provenance=source_provenance,
        workload_provenance=workload_provenance,
        required_blocks=required_blocks,
        resident_blocks=resident_blocks,
        host_only_blocks=host_only_blocks,
        restoring_blocks=restoring_blocks,
        newly_restored_blocks=plan.total_destination_ids,
        logical_128_token_units_proxy=logical_128_token_units_proxy,
        final_footprint_reserved_blocks=plan.reserved_final_footprint_blocks,
        jobs=jobs,
        wait_steps=(
            0 if completed_step is None else max(0, completed_step - scheduled_step)
        ),
        deadline_miss_reason=deadline_miss_reason,
        terminal_status=terminal_status,
        terminal_reason=terminal_reason,
        observed_start_ns=observed_start_ns,
        observed_end_ns=observed_end_ns,
    )


def total_plan_ids(plans: Iterable[RestorePlan]) -> int:
    """Small deterministic helper for bounded X-line correctness gates."""

    total = 0
    for plan in plans:
        if not isinstance(plan, RestorePlan):
            raise TypeError("plans must contain RestorePlan values")
        total += plan.total_destination_ids
    return total
