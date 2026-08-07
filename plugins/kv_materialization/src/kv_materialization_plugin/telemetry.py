"""Recent end-to-end materialization telemetry and calibration state."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from kv_materialization_plugin.decision import MaterializationObservation


@dataclass(frozen=True, slots=True)
class TimingSample:
    """One completed end-to-end materialization measurement."""

    size: int
    total_ms: float
    service_ms: float
    extra_wait_ms: float
    timestamp: float
    kv_bytes: int = 0


class TelemetryWindow:
    """Keep recent measurements keyed by prefix size."""

    def __init__(self, max_samples: int = 32) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._max_samples = max_samples
        self._load: dict[int, deque[TimingSample]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._recompute: dict[int, deque[TimingSample]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    def observe_load(
        self,
        blocks: int,
        total_ms: float,
        service_ms: float,
        kv_bytes: int = 0,
        timestamp: float | None = None,
    ) -> None:
        """Record a completed CPU-to-device load."""
        self._observe(
            self._load,
            blocks,
            total_ms,
            service_ms,
            kv_bytes,
            timestamp,
        )

    def observe_recompute(
        self,
        tokens: int,
        total_ms: float,
        service_ms: float,
        timestamp: float | None = None,
    ) -> None:
        """Record a completed prefix recompute."""
        self._observe(
            self._recompute,
            tokens,
            total_ms,
            service_ms,
            0,
            timestamp,
        )

    def snapshot(
        self,
        hit_tokens: int,
        hit_blocks: int,
        kv_bytes: int = 0,
    ) -> MaterializationObservation:
        """Create a decision observation from the closest recent buckets."""
        now = time.time()
        load = self._closest(self._load, hit_blocks)
        recompute = self._closest(self._recompute, hit_tokens)
        load_stats = self._stats(load, now)
        recompute_stats = self._stats(recompute, now)
        return MaterializationObservation(
            hit_tokens=hit_tokens,
            hit_blocks=hit_blocks,
            kv_bytes=kv_bytes,
            load_total_ms=load_stats[0],
            load_observation_age_ms=load_stats[1],
            load_sample_count=load_stats[2],
            recompute_total_ms=recompute_stats[0],
            recompute_observation_age_ms=recompute_stats[1],
            recompute_sample_count=recompute_stats[2],
        )

    def state(self) -> dict[str, list[dict[str, Any]]]:
        """Return JSON-compatible calibration state."""
        return {
            "load": [
                asdict(sample)
                for samples in self._load.values()
                for sample in samples
            ],
            "recompute": [
                asdict(sample)
                for samples in self._recompute.values()
                for sample in samples
            ],
        }

    def save_json(self, path: str | Path) -> None:
        """Atomically save calibration state for a later run."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            json.dump(self.state(), temporary, sort_keys=True)
            temporary.write("\n")
            temporary_path = temporary.name
        os.replace(temporary_path, destination)

    def load_json(self, path: str | Path) -> None:
        """Merge calibration state from a JSON file."""
        source = Path(path)
        if not source.is_file():
            return
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Telemetry state must be an object: {source}")
        for branch, buckets in (("load", self._load), ("recompute", self._recompute)):
            samples = value.get(branch, [])
            if not isinstance(samples, list):
                raise ValueError(f"Telemetry state field must be a list: {branch}")
            for raw in samples:
                if not isinstance(raw, dict):
                    raise ValueError(f"Telemetry sample must be an object: {branch}")
                sample = TimingSample(**raw)
                self._validate_sample(sample)
                buckets[sample.size].append(sample)

    def _observe(
        self,
        buckets: dict[int, deque[TimingSample]],
        size: int,
        total_ms: float,
        service_ms: float,
        kv_bytes: int,
        timestamp: float | None,
    ) -> None:
        if timestamp is None:
            timestamp = time.time()
        extra_wait_ms = total_ms - service_ms
        sample = TimingSample(
            size=size,
            total_ms=total_ms,
            service_ms=service_ms,
            extra_wait_ms=extra_wait_ms,
            timestamp=timestamp,
            kv_bytes=kv_bytes,
        )
        self._validate_sample(sample)
        buckets[size].append(sample)

    @staticmethod
    def _validate_sample(sample: TimingSample) -> None:
        if (
            sample.size <= 0
            or sample.total_ms < 0
            or sample.service_ms < 0
            or sample.extra_wait_ms < 0
            or sample.kv_bytes < 0
            or not all(
                math.isfinite(value)
                for value in (
                    sample.total_ms,
                    sample.service_ms,
                    sample.extra_wait_ms,
                    sample.timestamp,
                )
            )
        ):
            raise ValueError("Telemetry measurements must be non-negative")

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
    ) -> tuple[float | None, float | None, int]:
        if not samples:
            return None, None, 0
        newest = max(sample.timestamp for sample in samples)
        return (
            float(median(sample.total_ms for sample in samples)),
            max(0.0, (now - newest) * 1000.0),
            len(samples),
        )


__all__ = ["TelemetryWindow", "TimingSample"]
