# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

from vllm.config import CUDAGraphMode
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker.gpu.adaptive_state_probe import (
    AdaptiveStateProbe,
    ProbeTopology,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def _new_request(req_id: str) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _cached_requests(
    req_ids: list[str], *, num_output_tokens: list[int]
) -> CachedRequestData:
    count = len(req_ids)
    return CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in range(count)],
        all_token_ids={},
        new_block_ids=[None for _ in range(count)],
        num_computed_tokens=[0 for _ in range(count)],
        num_output_tokens=num_output_tokens,
    )


def _scheduler_output(
    *,
    scheduled_tokens: dict[str, int],
    new_reqs: list[NewRequestData] | None = None,
    cached_reqs: CachedRequestData | None = None,
    spec_tokens: dict[str, list[int]] | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs or [],
        scheduled_cached_reqs=cached_reqs or CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled_tokens,
        total_num_scheduled_tokens=sum(scheduled_tokens.values()),
        scheduled_spec_decode_tokens=spec_tokens or {},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _batch_desc(num_tokens: int, num_reqs: int) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
    )


def _probe(
    path: Path,
    *,
    topology: ProbeTopology | None = None,
    every_n_steps: int = 1,
    max_records: int = 100,
    max_bytes: int = 100_000,
    flush_every: int = 8,
    run_id: str = "test-run",
    pid: int = 123,
) -> AdaptiveStateProbe:
    return AdaptiveStateProbe(
        path,
        topology=topology or ProbeTopology(),
        every_n_steps=every_n_steps,
        max_records=max_records,
        max_bytes=max_bytes,
        flush_every=flush_every,
        run_id=run_id,
        pid=pid,
    )


def _record(probe: AdaptiveStateProbe, output: SchedulerOutput) -> None:
    probe.record_step(
        scheduler_output=output,
        batch_desc=_batch_desc(output.total_num_scheduled_tokens, 2),
        max_query_len=max(output.num_scheduled_tokens.values()),
        uniform_tok_count=None,
        dummy_run=False,
        skip_compiled=True,
    )


def _read_rows(probe: AdaptiveStateProbe) -> list[dict]:
    probe.close()
    return [json.loads(line) for line in probe.output_path.read_text().splitlines()]


def _sample_rows(probe: AdaptiveStateProbe) -> list[dict]:
    return [row for row in _read_rows(probe) if row["record_type"] == "sample"]


def test_from_env_is_disabled_without_path(monkeypatch):
    monkeypatch.delenv("VLLM_ADAPTIVE_STATE_PROBE_JSONL", raising=False)
    assert AdaptiveStateProbe.from_env(ProbeTopology()) is None


def test_invalid_integer_env_uses_finite_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_JSONL", str(tmp_path / "probe"))
    monkeypatch.setenv("VLLM_TELEMETRY_RUN_ID", "test-run")
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_MAX_RECORDS", "invalid")
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_MAX_BYTES", "invalid")

    probe = AdaptiveStateProbe.from_env(ProbeTopology())

    assert probe is not None
    assert probe._every_n_steps == 64
    assert probe._max_records == 10_000
    assert probe._max_bytes == 16 * 1024 * 1024
    probe.close()


def test_real_scheduler_types_preserve_spec_and_chunked_prefill_phases(tmp_path):
    probe = _probe(tmp_path / "probe.jsonl")
    sensitive_spec_id = "private-spec-request"
    sensitive_chunk_id = "private-chunk-request"
    output = _scheduler_output(
        scheduled_tokens={sensitive_spec_id: 4, sensitive_chunk_id: 1},
        cached_reqs=_cached_requests(
            [sensitive_spec_id, sensitive_chunk_id], num_output_tokens=[1, 0]
        ),
        spec_tokens={sensitive_spec_id: [10, 11, 12]},
    )

    _record(probe, output)
    rows = _sample_rows(probe)

    assert len(rows) == 1
    row = rows[0]
    assert row["num_reqs"] == 2
    assert row["num_cached_reqs"] == 2
    assert row["num_generation_reqs"] == 1
    assert row["num_generation_tokens"] == 4
    assert row["num_context_reqs"] == 1
    assert row["num_context_tokens"] == 1
    assert row["num_spec_decode_reqs"] == 1
    assert row["num_spec_decode_tokens"] == 3
    serialized = probe.output_path.read_text()
    assert sensitive_spec_id not in serialized
    assert sensitive_chunk_id not in serialized


def test_schema_provenance_and_monotonic_steps(tmp_path):
    topology = ProbeTopology(
        world_rank=3,
        world_size=8,
        dp_rank=1,
        dp_size=2,
        pp_rank=1,
        pp_size=2,
        tp_rank=1,
        tp_size=2,
    )
    probe = _probe(tmp_path / "probe", topology=topology, flush_every=2)
    output = _scheduler_output(
        scheduled_tokens={"new-private-id": 1},
        new_reqs=[_new_request("new-private-id")],
    )

    _record(probe, output)
    _record(probe, output)
    rows = _sample_rows(probe)

    assert [row["monotonic_step"] for row in rows] == [1, 2]
    assert rows[0]["timestamp_ns"] <= rows[1]["timestamp_ns"]
    assert rows[0]["schema_version"] == 1
    assert rows[0]["run_id"] == "test-run"
    assert rows[0]["pid"] == 123
    assert rows[0]["rank"] == 3
    assert rows[0]["world_size"] == 8
    assert rows[0]["layer_id"] == "__batch__"
    assert rows[0]["operator_id"] == "__scheduler_step__"
    assert rows[0]["topology"] == {
        "world_rank": 3,
        "world_size": 8,
        "dp_rank": 1,
        "dp_size": 2,
        "pp_rank": 1,
        "pp_size": 2,
        "tp_rank": 1,
        "tp_size": 2,
    }


def test_cadence_record_and_byte_bounds_count_drops(tmp_path):
    output = _scheduler_output(
        scheduled_tokens={"cached": 1},
        cached_reqs=_cached_requests(["cached"], num_output_tokens=[1]),
    )
    probe = _probe(
        tmp_path / "records",
        every_n_steps=2,
        max_records=2,
        flush_every=8,
    )
    for _ in range(6):
        _record(probe, output)

    all_rows = _read_rows(probe)
    rows = [row for row in all_rows if row["record_type"] == "sample"]
    assert [row["monotonic_step"] for row in rows] == [2, 4]
    assert probe.drop_counts == {
        "cadence": 3,
        "record_limit": 1,
        "byte_limit": 0,
        "summary_only": 0,
        "io": 0,
    }
    assert all_rows[-1]["record_type"] == "summary"
    assert all_rows[-1]["dropped"] == probe.drop_counts

    byte_probe = _probe(tmp_path / "bytes", max_bytes=1)
    _record(byte_probe, output)
    byte_rows = _read_rows(byte_probe)
    assert [row["record_type"] for row in byte_rows] == ["summary"]
    assert byte_probe.drop_counts["byte_limit"] == 1


def test_zero_budget_is_summary_only(tmp_path):
    probe = _probe(tmp_path / "summary", max_records=0, max_bytes=0)
    output = _scheduler_output(scheduled_tokens={"private": 1})

    _record(probe, output)
    rows = _read_rows(probe)

    assert [row["record_type"] for row in rows] == ["summary"]
    assert rows[0]["dropped"]["summary_only"] == 1


def test_default_owner_and_missing_multirank_identity_fail_closed(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_JSONL", str(tmp_path / "probe"))
    monkeypatch.setenv("VLLM_TELEMETRY_RUN_ID", "shared-run")
    monkeypatch.delenv("VLLM_ADAPTIVE_STATE_PROBE_ALL_RANKS", raising=False)

    assert (
        AdaptiveStateProbe.from_env(ProbeTopology(world_rank=None, world_size=2))
        is None
    )
    assert (
        AdaptiveStateProbe.from_env(ProbeTopology(world_rank=1, world_size=2)) is None
    )

    owner = AdaptiveStateProbe.from_env(ProbeTopology(world_rank=0, world_size=2))
    assert owner is not None
    owner.close()


def test_each_rank_and_process_gets_a_distinct_output_path(tmp_path):
    base = tmp_path / "probe.jsonl"
    rank_zero = _probe(
        base,
        topology=ProbeTopology(world_rank=0, world_size=2),
        run_id="same-run",
        pid=100,
    )
    rank_one = _probe(
        base,
        topology=ProbeTopology(world_rank=1, world_size=2),
        run_id="same-run",
        pid=100,
    )
    other_process = _probe(
        base,
        topology=ProbeTopology(world_rank=0, world_size=2),
        run_id="same-run",
        pid=101,
    )

    paths = {probe.output_path for probe in (rank_zero, rank_one, other_process)}
    assert len(paths) == 3
    assert all(
        probe._file is not None for probe in (rank_zero, rank_one, other_process)
    )
    for probe in (rank_zero, rank_one, other_process):
        probe.close()


def test_all_ranks_mode_uses_distinct_rank_files(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_JSONL", str(tmp_path / "probe"))
    monkeypatch.setenv("VLLM_TELEMETRY_RUN_ID", "shared-run")
    monkeypatch.setenv("VLLM_ADAPTIVE_STATE_PROBE_ALL_RANKS", "1")

    rank_zero = AdaptiveStateProbe.from_env(ProbeTopology(world_rank=0, world_size=2))
    rank_one = AdaptiveStateProbe.from_env(ProbeTopology(world_rank=1, world_size=2))

    assert rank_zero is not None
    assert rank_one is not None
    assert rank_zero.output_path != rank_one.output_path
    rank_zero.close()
    rank_one.close()
