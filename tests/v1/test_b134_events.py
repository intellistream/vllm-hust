import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "vllm" / "v1" / "b134_events.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("b134_events_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emit_is_disabled_without_output_path(monkeypatch) -> None:
    monkeypatch.delenv("B134_EVENTS_FILE", raising=False)
    module = _load_module()
    assert module.EVENTS_ENABLED is False
    module.emit("scheduled", "request-1")


def test_emit_writes_structured_jsonl(monkeypatch, tmp_path) -> None:
    output = tmp_path / "events.jsonl"
    monkeypatch.setenv("B134_EVENTS_FILE", str(output))
    module = _load_module()
    module.emit("transfer_submit", "request-1", bytes=4096, descriptors=4)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["event"] == "transfer_submit"
    assert record["request_id"] == "request-1"
    assert record["fields"] == {"bytes": 4096, "descriptors": 4}
    assert isinstance(record["pid"], int)
    assert isinstance(record["ts_monotonic_ns"], int)


def test_emit_chain_order_is_append_order(monkeypatch, tmp_path) -> None:
    """JSONL must preserve the exact per-request event order across emits."""
    output = tmp_path / "chain.jsonl"
    monkeypatch.setenv("B134_EVENTS_FILE", str(output))
    module = _load_module()

    # Contract for restore requests: wakeup -> admission -> scheduled
    for event in ("wakeup", "admission", "scheduled"):
        module.emit(event, "request-1")

    events = [
        json.loads(line)["event"]
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["wakeup", "admission", "scheduled"]
