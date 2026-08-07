"""External SimpleCPUOffloadConnector with dynamic materialization choice."""

from __future__ import annotations

import time
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
    TimingSampleMetadata,
    as_worker_metadata,
)
from kv_materialization_plugin.telemetry import TelemetryWindow
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata,
    )
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class DynamicSimpleCPUOffloadConnector(SimpleCPUOffloadConnector):
    """Choose CPU KV load or recompute without changing vLLM core code."""

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
            min_recompute_samples=int(extra_config.get("min_recompute_samples", 1)),
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
        self._audit = AuditLog()
        self._decisions: dict[str, MaterializationDecision] = {}
        self._decision_hit_tokens: dict[str, int] = {}
        self._decision_times: dict[str, float] = {}
        self._recompute_queue_wait_ms: dict[str, float] = {}
        self._worker_copy_samples: list[TimingSampleMetadata] = []
        self._worker_recompute_samples: list[TimingSampleMetadata] = []
        self._load_events_started: dict[
            int, tuple[float, dict[str, tuple[int, float | None]]]
        ] = {}
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
        self._decisions.pop(request.request_id, None)
        self._decision_hit_tokens.pop(request.request_id, None)
        self._decision_times.pop(request.request_id, None)
        self._recompute_queue_wait_ms.pop(request.request_id, None)
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
        self._recompute_queue_wait_ms[request.request_id] = max(
            0.0,
            (time.time() - request.arrival_time) * 1000.0,
        )
        self._audit.start(request.request_id, hit_tokens, hit_blocks, decision)
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
        load_decision_times = {
            request_id: self._decision_times[request_id]
            for request_id in load_request_ids
            if request_id in self._decision_times
        }
        recompute_queue_wait_ms = {
            request_id: self._recompute_queue_wait_ms[request_id]
            for request_id in recompute_requests
            if request_id in self._recompute_queue_wait_ms
        }
        if not recompute_requests and not load_decision_times:
            return base
        return DynamicCPUOffloadMetadata.from_base(
            base,
            recompute_requests,
            load_block_counts,
            load_decision_times,
            recompute_queue_wait_ms,
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
            self._recompute_started_at = time.perf_counter()

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
        launch_started_at = time.perf_counter()
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
                                metadata.load_event_to_reqs.get(metadata.load_event, [])
                            ),
                        ),
                    ),
                )
                for request_id in metadata.load_event_to_reqs.get(
                    metadata.load_event, []
                )
            }
            request_blocks = {
                request_id: (
                    block_count,
                    metadata.load_decision_times.get(request_id),
                )
                for request_id, block_count in request_blocks.items()
            }
            self._load_events_started.setdefault(
                metadata.load_event,
                (launch_started_at, request_blocks),
            )

        result = super().get_finished(finished_req_ids)
        finished_recving = result[1] or set()
        completed_at = time.perf_counter()
        for event_idx, (started_at, request_blocks) in list(
            self._load_events_started.items()
        ):
            if not finished_recving.intersection(request_blocks):
                continue
            for request_id in finished_recving.intersection(request_blocks):
                block_count, decision_time = request_blocks[request_id]
                queue_wait_ms = 0.0
                if decision_time is not None:
                    queue_wait_ms = max(
                        0.0,
                        (started_at - decision_time) * 1000.0,
                    )
                self._worker_copy_samples.append(
                    TimingSampleMetadata(
                        request_id=request_id,
                        size=block_count,
                        service_ms=(completed_at - started_at) * 1000.0,
                        queue_wait_ms=queue_wait_ms,
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
                        queue_wait_ms=metadata.recompute_queue_wait_ms.get(
                            request_id, 0.0
                        ),
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
        """Consume timing samples on the scheduler side."""
        super().update_connector_output(connector_output)
        metadata = connector_output.kv_connector_worker_meta
        if not hasattr(metadata, "copy_samples"):
            return
        for sample in metadata.copy_samples:
            self._telemetry.observe_copy(
                sample.size,
                sample.service_ms,
                sample.queue_wait_ms,
                sample.kv_bytes,
            )
            self._audit.complete(
                sample.request_id,
                "cpu_kv_load",
                sample.service_ms,
            )
        for sample in metadata.recompute_samples:
            self._telemetry.observe_recompute(
                sample.size,
                sample.service_ms,
                sample.queue_wait_ms,
            )
            self._audit.complete(
                sample.request_id,
                "full_prefix_recompute",
                sample.service_ms,
            )

    def take_audit_records(self):
        """Return accumulated audit records for the parent runtime."""
        return self._audit.records()

    def _estimate_hit_blocks(self, hit_tokens: int) -> int:
        """Convert hit tokens to the connector's scheduler block count."""
        if self.scheduler_manager is None:
            return hit_tokens
        block_size = max(1, int(self.scheduler_manager.block_size))
        return (hit_tokens + block_size - 1) // block_size


__all__ = ["DynamicSimpleCPUOffloadConnector"]
