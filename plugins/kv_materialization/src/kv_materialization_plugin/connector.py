"""External SimpleCPUOffloadConnector with dynamic materialization choice."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import replace
from importlib import import_module
from typing import TYPE_CHECKING

from kv_materialization_plugin.audit import AuditLog
from kv_materialization_plugin.decision import (
    MaterializationDecision,
    MaterializationDecisionConfig,
    MaterializationMode,
    choose_materialization,
)
from kv_materialization_plugin.metadata import (
    DynamicCPUOffloadMetadata,
    DynamicCPUOffloadWorkerMetadata,
    TimingSampleMetadata,
    as_worker_metadata,
)
from kv_materialization_plugin.telemetry import TelemetryWindow
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

_ASCEND_CONNECTOR_MODULE = (
    "vllm_ascend.distributed.kv_transfer.kv_pool."
    "simple_cpu_offload.simple_cpu_offload_connector"
)
try:
    _SimpleCPUOffloadConnector = import_module(
        _ASCEND_CONNECTOR_MODULE
    ).AscendSimpleCPUOffloadConnector
except ModuleNotFoundError as error:
    missing_module = error.name or ""
    if not (
        missing_module == _ASCEND_CONNECTOR_MODULE
        or _ASCEND_CONNECTOR_MODULE.startswith(f"{missing_module}.")
    ):
        raise
    _SimpleCPUOffloadConnector = import_module(
        "vllm.distributed.kv_transfer.kv_connector.v1."
        "simple_cpu_offload_connector"
    ).SimpleCPUOffloadConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata,
    )
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


def _advance_recompute_progress(
    remaining_tokens: int,
    scheduled_tokens: int,
) -> tuple[int, int, bool]:
    """Consume the scheduled part of a CPU-hit prefix."""
    if remaining_tokens < 0 or scheduled_tokens < 0:
        raise ValueError("Recompute progress must be non-negative")
    recomputed_tokens = min(remaining_tokens, scheduled_tokens)
    remaining_tokens -= recomputed_tokens
    return remaining_tokens, recomputed_tokens, remaining_tokens == 0


class DynamicSimpleCPUOffloadConnector(_SimpleCPUOffloadConnector):
    """Choose CPU KV load or prefix recompute without changing vLLM core code."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        materialization_mode = extra_config.get("materialization_mode", "load")
        forced_mode: MaterializationMode | None = (
            materialization_mode
            if materialization_mode in ("load", "recompute")
            else None
        )
        if materialization_mode not in ("load", "recompute", "dynamic"):
            raise ValueError(
                "materialization_mode must be 'load', 'recompute', or 'dynamic'"
            )
        fallback_mode = extra_config.get("fallback_mode", "load")
        self._decision_config = MaterializationDecisionConfig(
            enabled=materialization_mode == "dynamic",
            forced_mode=forced_mode,
            fallback_mode=fallback_mode,
            min_copy_samples=int(extra_config.get("min_copy_samples", 1)),
            min_recompute_samples=int(
                extra_config.get("min_recompute_samples", 1)
            ),
            max_observation_age_ms=float(
                extra_config.get("max_observation_age_ms", 5000.0)
            ),
        )
        self._kv_bytes_per_block = int(extra_config.get("kv_bytes_per_block", 0))
        if self._kv_bytes_per_block < 0:
            raise ValueError("kv_bytes_per_block must be non-negative")

        self._telemetry = TelemetryWindow(
            max_samples=int(extra_config.get("sample_window_size", 32))
        )
        self._telemetry_input_path = extra_config.get("telemetry_input_path")
        self._telemetry_output_path = extra_config.get("telemetry_output_path")
        if role == KVConnectorRole.SCHEDULER and self._telemetry_input_path:
            self._telemetry.load_json(self._telemetry_input_path)

        audit_enabled = bool(extra_config.get("audit_enabled", True))
        audit_path = extra_config.get("audit_output_path")
        self._audit = AuditLog(
            output_path=(
                audit_path
                if role == KVConnectorRole.SCHEDULER and audit_enabled
                else None
            ),
            run_id=extra_config.get("run_id"),
            mode=materialization_mode,
        )
        self._decisions: dict[str, MaterializationDecision] = {}
        self._decision_hit_tokens: dict[str, int] = {}
        self._decision_times: dict[str, float] = {}
        self._recompute_remaining_tokens: dict[str, int] = {}
        self._new_recompute_attempts: set[str] = set()
        self._worker_copy_samples: list[TimingSampleMetadata] = []
        self._worker_recompute_samples: list[TimingSampleMetadata] = []
        self._load_events_started: dict[
            int, tuple[float, dict[str, int], dict[str, float]]
        ] = {}
        self._recompute_step_started_at: float | None = None
        self._recompute_service_ms: dict[str, float] = {}
        self._recompute_queue_wait_ms: dict[str, float] = {}

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Choose a mode after the native CPU cache lookup."""
        hit_tokens, is_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        self._clear_request_state(request.request_id)
        if not hit_tokens:
            return hit_tokens, is_async

        hit_blocks = max(1, self._estimate_hit_blocks(hit_tokens))
        observation = self._telemetry.snapshot(
            hit_tokens,
            hit_blocks,
            kv_bytes=hit_blocks * self._kv_bytes_per_block,
            max_age_ms=self._decision_config.max_observation_age_ms,
        )
        observation = replace(
            observation,
            active_materialization_count=len(self._decisions),
        )
        decision = choose_materialization(observation, self._decision_config)
        self._decisions[request.request_id] = decision
        self._decision_hit_tokens[request.request_id] = hit_tokens
        self._decision_times[request.request_id] = time.monotonic()
        self._audit.start(
            request.request_id,
            hit_tokens,
            hit_blocks,
            decision,
            gpu_local_hit_tokens=num_computed_tokens,
            observation=observation,
        )
        if decision.mode == "recompute":
            self._recompute_remaining_tokens[request.request_id] = hit_tokens
            self._new_recompute_attempts.add(request.request_id)
            return 0, False
        return hit_tokens, is_async

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Attach recompute requests to otherwise native offload metadata."""
        base = super().build_connector_meta(scheduler_output)
        for request_id in scheduler_output.finished_req_ids:
            self._clear_request_state(request_id)
        if not isinstance(base, SimpleCPUOffloadMetadata):
            return base
        recompute_requests: dict[str, int] = {}
        reset_recompute_requests: set[str] = set()
        completed_recompute_requests: set[str] = set()
        for request_id, scheduled_tokens in (
            scheduler_output.num_scheduled_tokens.items()
        ):
            if (
                request_id not in self._decisions
                or self._decisions[request_id].mode != "recompute"
                or request_id not in self._decision_hit_tokens
            ):
                continue
            remaining = self._recompute_remaining_tokens.get(request_id, 0)
            remaining, recomputed_tokens, completed = _advance_recompute_progress(
                remaining,
                max(0, int(scheduled_tokens)),
            )
            if recomputed_tokens == 0:
                continue
            recompute_requests[request_id] = self._decision_hit_tokens[request_id]
            if request_id in self._new_recompute_attempts:
                reset_recompute_requests.add(request_id)
                self._new_recompute_attempts.discard(request_id)
            if completed:
                completed_recompute_requests.add(request_id)
                self._recompute_remaining_tokens.pop(request_id, None)
            else:
                self._recompute_remaining_tokens[request_id] = remaining
        load_request_ids = set(base.load_event_to_reqs.get(base.load_event, []))
        load_block_counts = {
            request_id: max(
                1,
                self._estimate_hit_blocks(self._decision_hit_tokens[request_id]),
            )
            for request_id in load_request_ids
            if request_id in self._decision_hit_tokens
        }
        if not recompute_requests and not load_block_counts:
            return base
        tracked_request_ids = set(recompute_requests) | set(load_block_counts)
        return DynamicCPUOffloadMetadata.from_base(
            base,
            recompute_requests,
            reset_recompute_requests,
            completed_recompute_requests,
            load_block_counts,
            {
                request_id: self._decision_times[request_id]
                for request_id in tracked_request_ids
                if request_id in self._decision_times
            },
        )

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        """Track the start of one worker-side recompute step."""
        super().bind_connector_metadata(connector_metadata)
        self._recompute_step_started_at = None
        if (
            isinstance(connector_metadata, DynamicCPUOffloadMetadata)
            and connector_metadata.recompute_requests
        ):
            for request_id in connector_metadata.reset_recompute_requests:
                self._recompute_service_ms.pop(request_id, None)
                decision_time = connector_metadata.decision_times.get(request_id)
                if decision_time is not None:
                    self._recompute_queue_wait_ms[request_id] = max(
                        0.0,
                        (time.monotonic() - decision_time) * 1000.0,
                    )
            self._recompute_step_started_at = time.monotonic()

    def clear_connector_metadata(self) -> None:
        """Clear native metadata and per-step worker timing state."""
        super().clear_connector_metadata()
        self._recompute_step_started_at = None

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Observe native completion notifications around worker execution."""
        metadata = self._connector_metadata
        launch_started_at = time.monotonic()
        if (
            isinstance(metadata, DynamicCPUOffloadMetadata)
            and metadata.load_event >= 0
            and metadata.load_cpu_blocks
        ):
            request_blocks = {
                request_id: max(
                    1,
                    metadata.load_block_counts.get(
                        request_id,
                        len(metadata.load_cpu_blocks)
                        // max(
                            1,
                            len(
                                metadata.load_event_to_reqs.get(
                                    metadata.load_event, []
                                )
                            ),
                        ),
                    ),
                )
                for request_id in metadata.load_event_to_reqs.get(
                    metadata.load_event, []
                )
            }
            self._load_events_started.setdefault(
                metadata.load_event,
                (
                    launch_started_at,
                    request_blocks,
                    {
                        request_id: metadata.decision_times.get(
                            request_id, launch_started_at
                        )
                        for request_id in request_blocks
                    },
                ),
            )

        result = super().get_finished(finished_req_ids)
        finished_recving = result[1] or set()
        completed_at = time.monotonic()
        for event_idx, (started_at, request_blocks, decision_times) in list(
            self._load_events_started.items()
        ):
            completed_request_ids = finished_recving.intersection(request_blocks)
            if not completed_request_ids:
                continue
            service_ms = (completed_at - started_at) * 1000.0
            for request_id in completed_request_ids:
                block_count = request_blocks[request_id]
                self._worker_copy_samples.append(
                    TimingSampleMetadata(
                        request_id=request_id,
                        size=block_count,
                        service_ms=service_ms,
                        kv_bytes=block_count * self._kv_bytes_per_block,
                        queue_wait_ms=max(
                            0.0,
                            (
                                launch_started_at
                                - decision_times.get(request_id, launch_started_at)
                            )
                            * 1000.0,
                        ),
                    )
                )
            self._load_events_started.pop(event_idx, None)

        if (
            isinstance(metadata, DynamicCPUOffloadMetadata)
            and metadata.recompute_requests
            and self._recompute_step_started_at is not None
        ):
            step_service_ms = (
                completed_at - self._recompute_step_started_at
            ) * 1000.0
            for request_id, hit_tokens in metadata.recompute_requests.items():
                service_ms = (
                    self._recompute_service_ms.get(request_id, 0.0)
                    + step_service_ms
                )
                if request_id in metadata.completed_recompute_requests:
                    self._worker_recompute_samples.append(
                        TimingSampleMetadata(
                            request_id=request_id,
                            size=max(1, hit_tokens),
                            service_ms=service_ms,
                            queue_wait_ms=self._recompute_queue_wait_ms.pop(
                                request_id, 0.0
                            ),
                        )
                    )
                    self._recompute_service_ms.pop(request_id, None)
                else:
                    self._recompute_service_ms[request_id] = service_ms
        for request_id in finished_req_ids:
            self._recompute_service_ms.pop(request_id, None)
            self._recompute_queue_wait_ms.pop(request_id, None)
        return result

    def build_connector_worker_meta(self):
        """Add timing samples to native worker metadata."""
        base = super().build_connector_worker_meta()
        result = as_worker_metadata(
            base,
            self._worker_copy_samples,
            self._worker_recompute_samples,
        )
        self._worker_copy_samples = []
        self._worker_recompute_samples = []
        return result

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Consume worker samples and derive scheduler-side total latency."""
        super().update_connector_output(connector_output)
        metadata = connector_output.kv_connector_worker_meta
        if not isinstance(metadata, DynamicCPUOffloadWorkerMetadata):
            return

        self._consume_samples(metadata.copy_samples, "cpu_kv_load", is_load=True)
        self._consume_samples(
            metadata.recompute_samples,
            "full_prefix_recompute",
            is_load=False,
        )
        if self._telemetry_output_path:
            self._telemetry.save_json(self._telemetry_output_path)

    def take_audit_records(self):
        """Return accumulated audit records for tests and diagnostics."""
        return self._audit.records()

    def _consume_samples(
        self,
        samples: list[TimingSampleMetadata],
        actual_branch: str,
        is_load: bool,
    ) -> None:
        """Aggregate worker samples and complete scheduler-side observations."""
        grouped: dict[str, list[TimingSampleMetadata]] = defaultdict(list)
        for sample in samples:
            grouped[sample.request_id].append(sample)

        for request_id, request_samples in grouped.items():
            critical_sample = max(request_samples, key=lambda sample: sample.service_ms)
            decision_time = self._decision_times.get(request_id)
            if decision_time is None:
                continue
            total_ms = (time.monotonic() - decision_time) * 1000.0
            service_ms = critical_sample.service_ms
            if service_ms > total_ms:
                self._audit.complete(
                    request_id,
                    actual_branch,
                    total_ms,
                    service_ms=service_ms,
                    status="invalid",
                )
                self._clear_request_state(request_id)
                continue
            extra_wait_ms = total_ms - service_ms
            queue_wait_ms = max(0.0, float(critical_sample.queue_wait_ms))
            if is_load:
                self._telemetry.observe_load(
                    critical_sample.size,
                    total_ms,
                    service_ms,
                    critical_sample.kv_bytes,
                    queue_wait_ms=queue_wait_ms,
                )
            else:
                self._telemetry.observe_recompute(
                    critical_sample.size,
                    total_ms,
                    service_ms,
                    queue_wait_ms=queue_wait_ms,
                )
            self._audit.complete(
                request_id,
                actual_branch,
                total_ms,
                service_ms=service_ms,
                extra_wait_ms=extra_wait_ms,
                queue_wait_ms=queue_wait_ms,
            )
            self._clear_request_state(request_id)

    def _clear_request_state(self, request_id: str) -> None:
        """Release scheduler-side state after a request is decided/completed."""
        self._decisions.pop(request_id, None)
        self._decision_hit_tokens.pop(request_id, None)
        self._decision_times.pop(request_id, None)
        self._recompute_remaining_tokens.pop(request_id, None)
        self._new_recompute_attempts.discard(request_id)
        getattr(self, "_recompute_queue_wait_ms", {}).pop(request_id, None)

    def _estimate_hit_blocks(self, hit_tokens: int) -> int:
        """Convert hit tokens to the connector's scheduler block count."""
        if self.scheduler_manager is None:
            return hit_tokens
        block_size = max(1, int(self.scheduler_manager.block_size))
        return (hit_tokens + block_size - 1) // block_size


__all__ = ["DynamicSimpleCPUOffloadConnector"]
