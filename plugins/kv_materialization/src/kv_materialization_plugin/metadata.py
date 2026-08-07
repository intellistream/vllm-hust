"""Plugin-owned metadata for timing samples."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)


@dataclass
class DynamicCPUOffloadMetadata(SimpleCPUOffloadMetadata):
    """Extend native offload metadata with recompute request IDs."""

    recompute_requests: dict[str, int] = field(default_factory=dict)
    load_block_counts: dict[str, int] = field(default_factory=dict)
    load_decision_times: dict[str, float] = field(default_factory=dict)
    recompute_queue_wait_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        base: SimpleCPUOffloadMetadata,
        recompute_requests: dict[str, int],
        load_block_counts: dict[str, int],
        load_decision_times: dict[str, float],
        recompute_queue_wait_ms: dict[str, float],
    ) -> DynamicCPUOffloadMetadata:
        """Copy native metadata without changing its semantics."""
        return cls(
            load_event=base.load_event,
            load_gpu_blocks=base.load_gpu_blocks,
            load_cpu_blocks=base.load_cpu_blocks,
            load_event_to_reqs=base.load_event_to_reqs,
            store_event=base.store_event,
            store_gpu_blocks=base.store_gpu_blocks,
            store_cpu_blocks=base.store_cpu_blocks,
            need_flush=base.need_flush,
            recompute_requests=recompute_requests,
            load_block_counts=load_block_counts,
            load_decision_times=load_decision_times,
            recompute_queue_wait_ms=recompute_queue_wait_ms,
        )


@dataclass(frozen=True)
class TimingSampleMetadata:
    """A completed worker-side timing sample."""

    request_id: str
    size: int
    service_ms: float
    queue_wait_ms: float = 0.0
    kv_bytes: int = 0


@dataclass
class DynamicCPUOffloadWorkerMetadata(SimpleCPUOffloadWorkerMetadata):
    """Carry native store completions and plugin timing samples."""

    copy_samples: list[TimingSampleMetadata] = field(default_factory=list)
    recompute_samples: list[TimingSampleMetadata] = field(default_factory=list)

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> DynamicCPUOffloadWorkerMetadata:
        """Aggregate worker samples while preserving native store counts."""
        if not isinstance(other, DynamicCPUOffloadWorkerMetadata):
            raise TypeError("Cannot aggregate different worker metadata types")
        merged_store_events = dict(self.completed_store_events)
        for event_idx, count in other.completed_store_events.items():
            merged_store_events[event_idx] = (
                merged_store_events.get(event_idx, 0) + count
            )
        return DynamicCPUOffloadWorkerMetadata(
            completed_store_events=merged_store_events,
            copy_samples=[*self.copy_samples, *other.copy_samples],
            recompute_samples=[*self.recompute_samples, *other.recompute_samples],
        )


def as_worker_metadata(
    base: SimpleCPUOffloadWorkerMetadata | None,
    copy_samples: list[TimingSampleMetadata],
    recompute_samples: list[TimingSampleMetadata],
) -> DynamicCPUOffloadWorkerMetadata | None:
    """Build plugin metadata only when there is something to report."""
    if base is None and not copy_samples and not recompute_samples:
        return None
    return DynamicCPUOffloadWorkerMetadata(
        completed_store_events=base.completed_store_events if base else {},
        copy_samples=copy_samples,
        recompute_samples=recompute_samples,
    )


__all__ = [
    "DynamicCPUOffloadMetadata",
    "DynamicCPUOffloadWorkerMetadata",
    "TimingSampleMetadata",
]
