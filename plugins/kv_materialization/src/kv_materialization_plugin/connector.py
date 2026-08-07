"""External SimpleCPUOffloadConnector with dynamic materialization choice."""

from __future__ import annotations

import time
from collections import defaultdict
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
except ImportError:
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
        self._worker_copy_samples: list[TimingSampleMetadata] = []
        self._worker_recompute_samples: list[TimingSampleMetadata] = []
        self._load_events_started: dict[int, tuple[float, dict[str, int]]] = {}
        self._recompute_started_at: float | None = None
        self._recompute_reported = False

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
        )
        if decision.mode == "recompute":
            return 0, False
        return hit_tokens, is_async

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Attach recompute requests to otherwise native offload metadata."""
        base = super().build_connector_meta(scheduler_output)
        if not isinstance(base, SimpleCPUOffloadMetadata):
            return base
        recompute_requests = {
            request_id: self._decision_hit_tokens[request_id]
            for request_id in scheduler_output.num_scheduled_tokens
            if request_id in self._decisions
            and self._decisions[request_id].mode == "recompute"
            and request_id in self._decision_hit_tokens
        }
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
        return DynamicCPUOffloadMetadata.from_base(
            base,
            recompute_requests,
            load_block_counts,
        )

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        """Track the start of worker-side recompute work."""
        super().bind_connector_metadata(connector_metadata)
        self._recompute_started_at = None
        self._recompute_reported = False
        if (
            isinstance(connector_metadata, DynamicCPUOffloadMetadata)
            and connector_metadata.recompute_requests
        ):
            self._recompute_started_at = time.monotonic()

    def clear_connector_metadata(self) -> None:
        """Clear native metadata and per-step worker timing state."""
        super().clear_connector_metadata()
        self._recompute_started_at = None
        self._recompute_reported = False

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
                (launch_started_at, request_blocks),
            )

        result = super().get_finished(finished_req_ids)
        finished_recving = result[1] or set()
        completed_at = time.monotonic()
        for event_idx, (started_at, request_blocks) in list(
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
                    )
                )
            self._load_events_started.pop(event_idx, None)

        if (
            isinstance(metadata, DynamicCPUOffloadMetadata)
            and metadata.recompute_requests
            and self._recompute_started_at is not None
            and not self._recompute_reported
        ):
            service_ms = (completed_at - self._recompute_started_at) * 1000.0
            for request_id, hit_tokens in metadata.recompute_requests.items():
                self._worker_recompute_samples.append(
                    TimingSampleMetadata(
                        request_id=request_id,
                        size=max(1, hit_tokens),
                        service_ms=service_ms,
                    )
                )
            self._recompute_reported = True
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
            if is_load:
                self._telemetry.observe_load(
                    critical_sample.size,
                    total_ms,
                    service_ms,
                    critical_sample.kv_bytes,
                )
            else:
                self._telemetry.observe_recompute(
                    critical_sample.size,
                    total_ms,
                    service_ms,
                )
            self._audit.complete(
                request_id,
                actual_branch,
                total_ms,
                service_ms=service_ms,
                extra_wait_ms=extra_wait_ms,
            )
            self._clear_request_state(request_id)

    def _clear_request_state(self, request_id: str) -> None:
        """Release scheduler-side state after a request is decided/completed."""
        self._decisions.pop(request_id, None)
        self._decision_hit_tokens.pop(request_id, None)
        self._decision_times.pop(request_id, None)

    def _estimate_hit_blocks(self, hit_tokens: int) -> int:
        """Convert hit tokens to the connector's scheduler block count."""
        if self.scheduler_manager is None:
            return hit_tokens
        block_size = max(1, int(self.scheduler_manager.block_size))
        return (hit_tokens + block_size - 1) // block_size


__all__ = ["DynamicSimpleCPUOffloadConnector"]
