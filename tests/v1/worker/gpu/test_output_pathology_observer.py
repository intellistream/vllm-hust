# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[4]
    / "vllm"
    / "v1"
    / "worker"
    / "gpu"
    / "output_pathology_observer.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_output_pathology_observer_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
OBSERVER_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OBSERVER_MODULE
SPEC.loader.exec_module(OBSERVER_MODULE)

OutputPathologyObserver = OBSERVER_MODULE.OutputPathologyObserver
get_observer = OBSERVER_MODULE.get_output_pathology_observer
reset_observer = OBSERVER_MODULE.reset_output_pathology_observer_for_test


@pytest.fixture(autouse=True)
def reset_global_observer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_TP1_OUTPUT_OBSERVER_PATH", raising=False)
    monkeypatch.delenv("VLLM_TP1_OUTPUT_OBSERVER_MAX_EVENTS", raising=False)
    monkeypatch.delenv("VLLM_TP1_OUTPUT_OBSERVER_FLUSH_EVENTS", raising=False)
    reset_observer()
    yield
    reset_observer()


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_default_off_returns_none() -> None:
    assert get_observer() is None
    assert get_observer() is None


def test_event_ledger_links_both_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "p0.jsonl"
    monkeypatch.setenv("VLLM_TP1_OUTPUT_OBSERVER_PATH", str(output_path))
    observer = get_observer()
    assert observer is not None

    output_id = observer.start_output(["req-b", "req-a"], (2, 1), "int64")
    observer.record_copy_issued(
        output_id,
        storage_id=1234,
        event_id=5678,
        nbytes=16,
        dispatch_ns=50,
    )
    observer.record_input_batch_retain(1234)
    observer.record_output_wait(output_id, 100)
    observer.record_output_materialization(output_id, 200)
    observer.record_input_batch_consume(
        1234, wait_ns=300, materialization_ns=400
    )
    observer.flush()

    events = read_events(output_path)
    assert events[0]["schema_version"] == "tp1-output-pathology/v1"
    by_name = {event["event"]: event for event in events}
    assert by_name["output_created"]["request_count"] == 2
    assert by_name["fresh_cpu_tensor_copy_issued"]["storage_id"] == 1234
    assert by_name["input_batch_retain"]["output_id"] == output_id
    assert by_name["input_batch_consume"]["output_id"] == output_id
    assert by_name["output_event_wait"]["wait_ns"] == 100
    assert by_name["output_python_materialization"][
        "materialization_ns"
    ] == 200
    assert by_name["observer_checkpoint"]["dropped_events"] == 0


def test_request_fingerprint_is_order_sensitive(tmp_path: Path) -> None:
    output_path = tmp_path / "fingerprints.jsonl"
    observer = OutputPathologyObserver(str(output_path), flush_events=100)
    observer.start_output(["a", "b"], (2, 1), "int64")
    observer.start_output(["b", "a"], (2, 1), "int64")
    observer.flush()

    created = [
        event
        for event in read_events(output_path)
        if event["event"] == "output_created"
    ]
    assert created[0]["request_fingerprint"] != created[1][
        "request_fingerprint"
    ]


def test_max_event_limit_is_reported(tmp_path: Path) -> None:
    output_path = tmp_path / "limited.jsonl"
    observer = OutputPathologyObserver(
        str(output_path), max_events=1, flush_events=100
    )
    output_id = observer.start_output(["a"], (1, 1), "int64")
    observer.record_output_wait(output_id, 1)
    observer.flush()

    checkpoint = read_events(output_path)[-1]
    assert checkpoint["event_count"] == 1
    assert checkpoint["dropped_events"] == 1


def test_drop_after_periodic_flush_updates_checkpoint(tmp_path: Path) -> None:
    output_path = tmp_path / "limited-after-flush.jsonl"
    observer = OutputPathologyObserver(
        str(output_path), max_events=1, flush_events=1
    )
    output_id = observer.start_output(["a"], (1, 1), "int64")
    observer.record_output_wait(output_id, 1)
    observer.flush()

    checkpoint = read_events(output_path)[-1]
    assert checkpoint["event_count"] == 1
    assert checkpoint["dropped_events"] == 1


def test_write_failure_never_escapes_to_serving(tmp_path: Path) -> None:
    observer = OutputPathologyObserver(str(tmp_path), flush_events=100)
    observer.start_output(["a"], (1, 1), "int64")

    observer.flush()
    observer.record_output_wait(0, 1)
    observer.flush()


def test_invalid_environment_limits_disable_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "VLLM_TP1_OUTPUT_OBSERVER_PATH", str(tmp_path / "invalid.jsonl")
    )
    monkeypatch.setenv("VLLM_TP1_OUTPUT_OBSERVER_MAX_EVENTS", "not-an-int")

    assert get_observer() is None
