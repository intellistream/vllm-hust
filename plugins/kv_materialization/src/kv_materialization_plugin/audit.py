"""Auditable per-request materialization decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from kv_materialization_plugin.decision import (
    MaterializationDecision,
    MaterializationObservation,
)


@dataclass(slots=True)
class AuditRecord:
    """Decision and completion data for one materialization attempt."""

    request_id: str
    hit_tokens: int
    hit_blocks: int
    decision: str
    reason: str
    predicted_load_ms: float | None
    predicted_recompute_ms: float | None
    fallback: bool
    kv_bytes: int = 0
    active_materialization_count: int = 0
    timing_scope: str = (
        "decision_to_worker_sample_return;"
        " phase_queue=admission_to_service_start"
    )
    queue_wait_isolated: bool = False
    load_queue_wait_ms: float | None = None
    load_service_ms: float | None = None
    load_extra_wait_ms: float | None = None
    load_observation_age_ms: float | None = None
    load_sample_count: int = 0
    recompute_service_ms: float | None = None
    recompute_queue_wait_ms: float | None = None
    recompute_extra_wait_ms: float | None = None
    recompute_observation_age_ms: float | None = None
    recompute_sample_count: int = 0
    run_id: str | None = None
    mode: str | None = None
    gpu_local_hit_tokens: int | None = None
    invalid_fields: tuple[str, ...] = ()
    actual_branch: str | None = None
    actual_cost_ms: float | None = None
    service_ms: float | None = None
    extra_wait_ms: float | None = None
    queue_wait_ms: float | None = None
    token_coverage_start: int = 0
    token_coverage_end: int | None = None
    scheduler_ownership: str | None = None
    status: str = "decided"


class AuditLog:
    """Collect records without imposing a logging framework on vLLM."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        run_id: str | None = None,
        mode: str | None = None,
    ) -> None:
        self._records: dict[str, AuditRecord] = {}
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._output = destination.open("a", encoding="utf-8", buffering=1)
        else:
            self._output = None
        self._run_id = run_id
        self._mode = mode

    def start(
        self,
        request_id: str,
        hit_tokens: int,
        hit_blocks: int,
        decision: MaterializationDecision,
        gpu_local_hit_tokens: int | None = None,
        observation: MaterializationObservation | None = None,
    ) -> None:
        """Start or replace a record for a request."""
        self._records[request_id] = AuditRecord(
            request_id=request_id,
            hit_tokens=hit_tokens,
            hit_blocks=hit_blocks,
            decision=decision.mode,
            reason=decision.reason,
            predicted_load_ms=decision.predicted_load_ms,
            predicted_recompute_ms=decision.predicted_recompute_ms,
            fallback=decision.fallback,
            kv_bytes=observation.kv_bytes if observation else 0,
            active_materialization_count=(
                observation.active_materialization_count if observation else 0
            ),
            queue_wait_isolated=bool(
                observation
                and observation.load_queue_wait_ms is not None
                and observation.recompute_queue_wait_ms is not None
            ),
            load_service_ms=observation.load_service_ms if observation else None,
            load_queue_wait_ms=(
                observation.load_queue_wait_ms if observation else None
            ),
            load_extra_wait_ms=(
                observation.load_extra_wait_ms if observation else None
            ),
            load_observation_age_ms=(
                observation.load_observation_age_ms if observation else None
            ),
            load_sample_count=observation.load_sample_count if observation else 0,
            recompute_service_ms=(
                observation.recompute_service_ms if observation else None
            ),
            recompute_queue_wait_ms=(
                observation.recompute_queue_wait_ms if observation else None
            ),
            recompute_extra_wait_ms=(
                observation.recompute_extra_wait_ms if observation else None
            ),
            recompute_observation_age_ms=(
                observation.recompute_observation_age_ms if observation else None
            ),
            recompute_sample_count=(
                observation.recompute_sample_count if observation else 0
            ),
            run_id=self._run_id,
            mode=self._mode,
            gpu_local_hit_tokens=gpu_local_hit_tokens,
            invalid_fields=decision.invalid_fields,
        )

    def complete(
        self,
        request_id: str,
        actual_branch: str,
        actual_cost_ms: float,
        service_ms: float | None = None,
        extra_wait_ms: float | None = None,
        queue_wait_ms: float | None = None,
        status: str = "completed",
    ) -> None:
        """Complete an existing record."""
        record = self._records.get(request_id)
        if record is None:
            return
        record.actual_branch = actual_branch
        record.actual_cost_ms = actual_cost_ms
        record.service_ms = service_ms
        record.extra_wait_ms = extra_wait_ms
        record.queue_wait_ms = queue_wait_ms
        record.token_coverage_end = record.hit_tokens
        record.scheduler_ownership = {
            "cpu_kv_load": "scheduler_reserved_gpu_blocks_until_load_completion",
            "full_prefix_recompute": "scheduler_owned_recompute_progress",
        }.get(actual_branch)
        record.status = status
        self._write(record)

    def close(self) -> None:
        """Flush and close the optional NDJSON output."""
        if self._output is not None:
            self._output.close()
            self._output = None

    def records(self) -> list[AuditRecord]:
        """Return records in insertion order."""
        return list(self._records.values())

    def json_lines(self) -> str:
        """Serialize records as newline-delimited JSON."""
        return "\n".join(
            json.dumps(asdict(record), sort_keys=True) for record in self.records()
        )

    def _write(self, record: AuditRecord) -> None:
        if self._output is None:
            return
        self._output.write(json.dumps(asdict(record), sort_keys=True) + "\n")
