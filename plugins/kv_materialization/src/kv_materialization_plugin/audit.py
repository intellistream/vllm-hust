"""Auditable per-request materialization decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from kv_materialization_plugin.decision import MaterializationDecision


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
    actual_branch: str | None = None
    actual_cost_ms: float | None = None
    status: str = "decided"


class AuditLog:
    """Collect records without imposing a logging framework on vLLM."""

    def __init__(self) -> None:
        self._records: dict[str, AuditRecord] = {}

    def start(
        self,
        request_id: str,
        hit_tokens: int,
        hit_blocks: int,
        decision: MaterializationDecision,
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
        )

    def complete(
        self,
        request_id: str,
        actual_branch: str,
        actual_cost_ms: float,
        status: str = "completed",
    ) -> None:
        """Complete an existing record."""
        record = self._records.get(request_id)
        if record is None:
            return
        record.actual_branch = actual_branch
        record.actual_cost_ms = actual_cost_ms
        record.status = status

    def records(self) -> list[AuditRecord]:
        """Return records in insertion order."""
        return list(self._records.values())

    def json_lines(self) -> str:
        """Serialize records as newline-delimited JSON."""
        return "\n".join(
            json.dumps(asdict(record), sort_keys=True) for record in self.records()
        )
