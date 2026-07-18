# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from vllm.v1.worker.gpu.adaptive_state_probe import AdaptiveStateProbe


def _record(probe: AdaptiveStateProbe) -> None:
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"decode": 1, "prefill": 4},
        total_num_scheduled_tokens=5,
        scheduled_new_reqs=[object()],
        scheduled_cached_reqs=[object(), object()],
        finished_req_ids=["done"],
        preempted_req_ids=[],
        has_structured_output_requests=True,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={"decode": [1, 2]},
    )
    batch_desc = SimpleNamespace(
        num_tokens=5,
        num_reqs=2,
        cg_mode=SimpleNamespace(name="PIECEWISE"),
    )
    probe.record_step(
        scheduler_output=scheduler_output,
        batch_desc=batch_desc,
        max_query_len=4,
        uniform_tok_count=1,
        dummy_run=False,
        skip_compiled=True,
    )


def test_from_env_is_disabled_without_path(monkeypatch):
    monkeypatch.delenv("VLLM_ADAPTIVE_STATE_PROBE_JSONL", raising=False)
    assert AdaptiveStateProbe.from_env() is None


def test_invalid_integer_env_uses_safe_defaults(monkeypatch, tmp_path):
    output_path = tmp_path / "probe.jsonl"
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_JSONL", str(output_path))
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_EVERY", "invalid")
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_MAX_RECORDS", "invalid")

    probe = AdaptiveStateProbe.from_env()

    assert probe is not None
    assert probe._every_n_steps == 1
    assert probe._max_records == 0


def test_probe_honors_cadence_and_record_limit(tmp_path):
    output_path = tmp_path / "probe.jsonl"
    probe = AdaptiveStateProbe(output_path, every_n_steps=2, max_records=2)

    for _ in range(6):
        _record(probe)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [2, 4]
    assert rows[0]["num_decode_reqs"] == 1
    assert rows[0]["num_prefill_reqs"] == 1
    assert rows[0]["num_tokens"] == 5
    assert rows[0]["batch_cg_mode"] == "PIECEWISE"
    assert rows[0]["has_structured_output_requests"] is True
    assert rows[0]["num_spec_decode_reqs"] == 1


def test_write_failure_disables_probe(tmp_path):
    probe = AdaptiveStateProbe(tmp_path, every_n_steps=1, max_records=0)

    _record(probe)
    first_step = probe._step
    _record(probe)

    assert probe._disabled is True
    assert probe._step == first_step
