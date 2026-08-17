"""Opt-in low-overhead server event sink for M0 measurements."""
from __future__ import annotations
import atexit, json, os, queue, threading, time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "m0-server-event-v1"
_enabled = os.getenv("VLLM_M0_SERVER_EVENTS", "0").lower() in {"1", "true", "yes"}
_sampling = max(0.0, min(1.0, float(os.getenv("VLLM_M0_SERVER_EVENT_SAMPLING_RATE", "1.0"))))
_token_detail = os.getenv("VLLM_M0_SERVER_EVENT_TOKEN_DETAIL", "0").lower() in {"1", "true", "yes"}
_event_dir = Path(os.getenv("VLLM_M0_SERVER_EVENT_DIR", "/tmp/vllm-m0-events"))
_pid = os.getpid()
_q: queue.Queue[dict[str, Any] | None] | None = None
_writer: threading.Thread | None = None
_lock = threading.Lock()
_counts = {"written": 0, "dropped": 0, "writer_errors": 0}
_iteration_counter = 0

def enabled() -> bool:
    return _enabled
def token_detail_enabled() -> bool:
    return _enabled and _token_detail
def next_iteration_id() -> str:
    global _iteration_counter
    _iteration_counter += 1
    return f"iter-{_pid}-{_iteration_counter}"
def _writer_main(path: Path) -> None:
    assert _q is not None
    try:
        with path.open("a", encoding="utf-8") as fp:
            while True:
                item = _q.get()
                if item is None:
                    _q.task_done()
                    break
                try:
                    fp.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                    fp.flush()
                    _counts["written"] += 1
                except Exception:
                    _counts["writer_errors"] += 1
                finally:
                    _q.task_done()
    except Exception:
        _counts["writer_errors"] += 1
def _ensure_writer() -> None:
    global _q, _writer
    if not _enabled or _writer is not None:
        return
    with _lock:
        if _writer is not None:
            return
        try:
            _event_dir.mkdir(parents=True, exist_ok=True)
            _q = queue.Queue(maxsize=int(os.getenv("VLLM_M0_SERVER_EVENT_BUFFER", "8192")))
            _writer = threading.Thread(target=_writer_main, args=(_event_dir / f"server_events.{_pid}.jsonl",), daemon=True)
            _writer.start()
        except Exception:
            _counts["writer_errors"] += 1
def emit(event_type: str, *, request_id: str | None = None, engine_request_id: str | None = None,
         client_request_id: str | None = None, iteration_id: str | None = None,
         payload: dict[str, Any] | None = None, source_component: str = "vllm.v1",
         measurement_method: str = "existing_fork_field", event_monotonic_ns: int | None = None) -> None:
    if not _enabled:
        return
    if _sampling < 1.0 and hash(request_id or "") % 10000 >= int(_sampling * 10000):
        return
    _ensure_writer()
    if _q is None:
        return
    event = {
        "schema_version": SCHEMA_VERSION, "experiment_id": os.getenv("M0_EXPERIMENT_ID"),
        "run_id": os.getenv("M0_RUN_ID"), "trace_id": os.getenv("M0_TRACE_ID"),
        "client_request_id": client_request_id or request_id, "server_request_id": request_id,
        "engine_request_id": engine_request_id or request_id, "iteration_id": iteration_id,
        "batch_id": iteration_id, "event_type": event_type, "process_id": _pid,
        "worker_rank": int(os.getenv("LOCAL_RANK", "0")),
        "device_id": os.getenv("ASCEND_RT_VISIBLE_DEVICES", os.getenv("ASCEND_VISIBLE_DEVICES")),
        "monotonic_ns": event_monotonic_ns or time.monotonic_ns(),
        "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{time.time_ns() % 1_000_000_000:09d}Z",
        "source_component": source_component, "measurement_method": measurement_method,
        "payload": payload or {},
    }
    try:
        _q.put_nowait(event)
    except Exception:
        _counts["dropped"] += 1
def flush() -> None:
    if _q is None:
        return
    try:
        _q.join()
        _q.put_nowait(None)
        if _writer is not None:
            _writer.join(timeout=5)
        _event_dir.mkdir(parents=True, exist_ok=True)
        (_event_dir / f"writer_stats.{_pid}.json").write_text(json.dumps(_counts) + "\n")
    except Exception:
        _counts["writer_errors"] += 1
atexit.register(flush)
