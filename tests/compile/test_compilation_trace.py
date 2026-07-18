# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.compilation.trace import TRACE_PATH_ENV, emit_compilation_trace


def test_emit_compilation_trace_is_opt_in(monkeypatch, tmp_path):
    trace_path = tmp_path / "compile.jsonl"
    monkeypatch.delenv(TRACE_PATH_ENV, raising=False)

    emit_compilation_trace("disabled", value=1)

    assert not trace_path.exists()


def test_emit_compilation_trace_writes_structured_event(monkeypatch, tmp_path):
    trace_path = tmp_path / "compile.jsonl"
    monkeypatch.setenv(TRACE_PATH_ENV, str(trace_path))

    emit_compilation_trace("cache_lookup", key={"shape": 8}, hit=False)

    row = json.loads(trace_path.read_text(encoding="utf-8"))
    assert row["schema_version"] == 1
    assert row["event"] == "cache_lookup"
    assert row["key"] == {"shape": 8}
    assert row["hit"] is False
    assert row["timestamp_ns"] > 0


def test_emit_compilation_trace_never_breaks_serving(monkeypatch, tmp_path):
    monkeypatch.setenv(TRACE_PATH_ENV, str(tmp_path / "missing" / "trace.jsonl"))

    emit_compilation_trace("ignored_io_error")
