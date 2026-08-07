"""CPU-only tests for materialization audit records."""

from __future__ import annotations

import json

from kv_materialization_plugin.audit import AuditLog
from kv_materialization_plugin.decision import MaterializationDecision


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
    )
    audit.complete(
        "req-1",
        "cpu_kv_load",
        5.5,
        service_ms=4.0,
        extra_wait_ms=1.5,
    )

    record = audit.records()[0]
    assert record.request_id == "req-1"
    assert record.actual_branch == "cpu_kv_load"
    assert record.actual_cost_ms == 5.5
    assert record.service_ms == 4.0
    assert record.extra_wait_ms == 1.5
    assert json.loads(audit.json_lines())["status"] == "completed"
    audit.close()
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["run_id"] == "run-1"
    assert exported["mode"] == "load"
