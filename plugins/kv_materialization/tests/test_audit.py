"""CPU-only tests for materialization audit records."""

from __future__ import annotations

import json

from kv_materialization_plugin.audit import AuditLog
from kv_materialization_plugin.decision import (
    MaterializationDecision,
    MaterializationObservation,
)


def test_audit_record_contains_prediction_and_actual_branch(tmp_path) -> None:
    """An audit record can explain the selected and executed paths."""
    output_path = tmp_path / "audit.jsonl"
    audit = AuditLog(output_path=output_path, run_id="run-1", mode="load")
    audit.start(
        "req-1",
        256,
        2,
        MaterializationDecision(
            mode="load",
            reason="predicted_load_is_lower",
            predicted_load_ms=5.0,
            predicted_recompute_ms=22.0,
        ),
        observation=MaterializationObservation(
            hit_tokens=256,
            hit_blocks=2,
            kv_bytes=1024,
            active_materialization_count=2,
            load_total_ms=5.0,
            load_service_ms=4.0,
            load_queue_wait_ms=1.0,
            load_extra_wait_ms=1.0,
            load_observation_age_ms=10.0,
            load_sample_count=3,
            recompute_total_ms=22.0,
            recompute_service_ms=20.0,
            recompute_queue_wait_ms=2.0,
            recompute_extra_wait_ms=2.0,
            recompute_observation_age_ms=12.0,
            recompute_sample_count=4,
        ),
    )
    audit.complete(
        "req-1",
        "cpu_kv_load",
        5.5,
        service_ms=4.0,
        extra_wait_ms=1.5,
        queue_wait_ms=1.0,
    )

    record = audit.records()[0]
    assert record.request_id == "req-1"
    assert record.actual_branch == "cpu_kv_load"
    assert record.actual_cost_ms == 5.5
    assert record.service_ms == 4.0
    assert record.extra_wait_ms == 1.5
    assert record.kv_bytes == 1024
    assert record.active_materialization_count == 2
    assert "phase_queue=admission_to_service_start" in record.timing_scope
    assert record.queue_wait_isolated is True
    assert record.queue_wait_ms == 1.0
    assert record.load_queue_wait_ms == 1.0
    assert record.recompute_queue_wait_ms == 2.0
    assert record.load_service_ms == 4.0
    assert record.load_sample_count == 3
    assert record.recompute_extra_wait_ms == 2.0
    assert record.recompute_sample_count == 4
    assert json.loads(audit.json_lines())["status"] == "completed"
    audit.close()
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["run_id"] == "run-1"
    assert exported["mode"] == "load"
