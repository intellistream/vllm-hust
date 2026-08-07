"""Small in-process telemetry window for materialization estimates."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median

from kv_materialization_plugin.decision import MaterializationObservation


@dataclass(frozen=True, slots=True)
class TimingSample:
    """One completed load or recompute measurement."""

    size: int
    service_ms: float
    queue_wait_ms: float
    timestamp: float
    kv_bytes: int = 0


class TelemetryWindow:
    """Keep recent measurements keyed by prefix size."""

    def __init__(self, max_samples: int = 32) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._max_samples = max_samples
        self._copy: dict[int, deque[TimingSample]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._recompute: dict[int, deque[TimingSample]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    def observe_copy(
        self,
        blocks: int,
        service_ms: float,
        queue_wait_ms: float = 0.0,
        kv_bytes: int = 0,
    ) -> None:
        """Record a completed CPU-to-device copy."""
        self._observe(self._copy, blocks, service_ms, queue_wait_ms, kv_bytes)

    def observe_recompute(
        self,
        tokens: int,
        service_ms: float,
        queue_wait_ms: float = 0.0,
    ) -> None:
        """Record a completed prefix recompute."""
        self._observe(self._recompute, tokens, service_ms, queue_wait_ms, 0)

    def snapshot(
        self,
        hit_tokens: int,
        hit_blocks: int,
        kv_bytes: int = 0,
    ) -> MaterializationObservation:
        """Create a decision observation from the closest recent buckets."""
        now = time.monotonic()
        copy = self._closest(self._copy, hit_blocks)
        recompute = self._closest(self._recompute, hit_tokens)
        copy_stats = self._stats(copy, now)
        recompute_stats = self._stats(recompute, now)
        return MaterializationObservation(
            hit_tokens=hit_tokens,
            hit_blocks=hit_blocks,
            kv_bytes=kv_bytes,
            copy_queue_wait_ms=copy_stats[0],
            copy_service_ms=copy_stats[1],
            copy_bandwidth_bytes_per_ms=copy_stats[2],
            copy_observation_age_ms=copy_stats[3],
            copy_sample_count=copy_stats[4],
            recompute_queue_wait_ms=recompute_stats[0],
            recompute_service_ms=recompute_stats[1],
            recompute_observation_age_ms=recompute_stats[3],
            recompute_sample_count=recompute_stats[4],
        )

    @staticmethod
    def _observe(
        buckets: dict[int, deque[TimingSample]],
        size: int,
        service_ms: float,
        queue_wait_ms: float,
        kv_bytes: int,
    ) -> None:
        if size <= 0 or service_ms < 0 or queue_wait_ms < 0 or kv_bytes < 0:
            raise ValueError("Telemetry measurements must be non-negative")
        buckets[size].append(
            TimingSample(
                size=size,
                service_ms=service_ms,
                queue_wait_ms=queue_wait_ms,
                timestamp=time.monotonic(),
                kv_bytes=kv_bytes,
            )
        )

    @staticmethod
    def _closest(
        buckets: dict[int, deque[TimingSample]], size: int
    ) -> deque[TimingSample] | None:
        if not buckets:
            return None
        key = min(buckets, key=lambda candidate: abs(candidate - size))
        return buckets[key]

    @staticmethod
    def _stats(
        samples: deque[TimingSample] | None,
        now: float,
    ) -> tuple[float | None, float | None, float | None, float | None, int]:
        if not samples:
            return None, None, None, None, 0
        queue_waits = [sample.queue_wait_ms for sample in samples]
        services = [sample.service_ms for sample in samples]
        bandwidths = [
            sample.kv_bytes / sample.service_ms
            for sample in samples
            if sample.kv_bytes > 0 and sample.service_ms > 0
        ]
        newest = max(sample.timestamp for sample in samples)
        return (
            float(median(queue_waits)),
            float(median(services)),
            float(median(bandwidths)) if bandwidths else None,
            max(0.0, (now - newest) * 1000.0),
            len(samples),
        )
