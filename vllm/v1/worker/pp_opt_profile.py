# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import csv
import functools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

import vllm.envs as envs_vllm
from vllm.distributed.parallel_state import get_pp_group, get_tp_group

PROFILE_FIELDS = (
    "profile_step",
    "microbatch_id",
    "pp_rank",
    "tp_rank",
    "layer_num",
    "request_num",
    "aggregated_ctx_len",
    "total_scheduled_tokens",
    "t1_ns",
    "t2_ns",
    "t3_ns",
    "t4_ns",
    "t5_ns",
    "t_model_runner_execute_start_ns",
    "cost_forward_ns",
    "cost_total_ns",
    "cost_model_runner_execute",
)

SAMPLE_TOKENS_PROFILE_FIELDS = (
    "start_timestamp_ns",
    "end_timestamp_ns",
)


@dataclass
class ProfileRecord:
    profile_step: int
    microbatch_id: int
    pp_rank: int
    tp_rank: int
    layer_num: int
    request_num: int = 0
    aggregated_ctx_len: int = 0
    total_scheduled_tokens: int = 0
    t1_ns: int | None = None
    t2_ns: int | None = None
    t3_ns: int | None = None
    t4_ns: int | None = None
    t5_ns: int | None = None
    t_model_runner_execute_start_ns: int | None = None


_current_record: ProfileRecord | None = None
_profile_step = 0
_csv_writers: dict[Path, tuple[TextIO, csv.DictWriter]] = {}
_rank_output_paths: dict[tuple[int, int], Path] = {}
_sample_tokens_output_paths: dict[tuple[int, int], Path] = {}


def enabled() -> bool:
    return bool(
        envs_vllm.VLLM_PROFILE_PP_OPT_ENABLED
        and envs_vllm.VLLM_PROFILE_PP_OPT_OUTPUT_PATH
    )


def now_ns() -> int:
    return time.perf_counter_ns()


def _rank_info() -> tuple[int, int]:
    return get_pp_group().rank_in_group, get_tp_group().rank_in_group


def should_profile_current_rank() -> bool:
    return enabled()


def record_active() -> bool:
    return _current_record is not None


def _local_layer_num(vllm_config) -> int:
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if model_config is not None and parallel_config is not None:
        get_num_layers = getattr(model_config, "get_num_layers", None)
        if get_num_layers is not None:
            return int(get_num_layers(parallel_config))
        get_layers_start_end_indices = getattr(
            model_config, "get_layers_start_end_indices", None
        )
        if get_layers_start_end_indices is not None:
            start, end = get_layers_start_end_indices(parallel_config)
            return int(end - start)

    env_layer_num = os.getenv("VLLM_PP_OPT_PROFILE_LAYER_NUM")
    if env_layer_num is not None:
        return int(env_layer_num)

    return -1


def start_record(scheduler_output, vllm_config=None) -> None:
    global _current_record, _profile_step
    if not should_profile_current_rank():
        _current_record = None
        return

    pp_rank, tp_rank = _rank_info()
    _profile_step += 1
    _current_record = ProfileRecord(
        profile_step=_profile_step,
        microbatch_id=getattr(scheduler_output, "microbatch_id", -1),
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        layer_num=_local_layer_num(vllm_config),
        total_scheduled_tokens=getattr(
            scheduler_output, "total_num_scheduled_tokens", 0
        ),
        t1_ns=now_ns(),
    )


def set_microbatch_stats(input_batch, num_scheduled_tokens: np.ndarray) -> None:
    if _current_record is None:
        return

    request_num = int(input_batch.num_reqs)
    scheduled = num_scheduled_tokens[:request_num].astype(np.int64, copy=False)
    computed = input_batch.num_computed_tokens_cpu[:request_num].astype(
        np.int64, copy=False
    )
    _current_record.request_num = request_num
    _current_record.aggregated_ctx_len = int(np.sum(computed + scheduled))
    _current_record.total_scheduled_tokens = int(np.sum(scheduled))


def mark_t2() -> None:
    if _current_record is not None:
        _current_record.t2_ns = now_ns()


def mark_t3() -> None:
    if _current_record is not None:
        _current_record.t3_ns = now_ns()


def mark_t4() -> None:
    if _current_record is not None:
        _current_record.t4_ns = now_ns()


def mark_model_runner_execute_start() -> None:
    if _current_record is not None:
        _current_record.t_model_runner_execute_start_ns = now_ns()


def profile_worker_execute(func):
    @functools.wraps(func)
    def wrapper(self, scheduler_output, *args, **kwargs):
        start_record(scheduler_output, getattr(self, "vllm_config", None))
        try:
            return func(self, scheduler_output, *args, **kwargs)
        finally:
            finish_record()

    return wrapper


def profile_model_runner_execute(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mark_model_runner_execute_start()
        try:
            return func(*args, **kwargs)
        finally:
            mark_t4()

    return wrapper


def profile_worker_sample_tokens(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not should_profile_current_rank():
            return func(*args, **kwargs)

        start_ns = now_ns()
        try:
            return func(*args, **kwargs)
        finally:
            end_ns = now_ns()
            _append_sample_tokens_record(start_ns, end_ns)

    return wrapper


def finish_record() -> None:
    global _current_record
    if _current_record is None:
        return

    _current_record.t5_ns = now_ns()
    try:
        if _is_complete(_current_record):
            _append_record(_current_record)
    finally:
        _current_record = None


def _is_complete(record: ProfileRecord) -> bool:
    return (
        record.request_num > 0
        and record.total_scheduled_tokens > 0
        and record.t1_ns is not None
        and record.t2_ns is not None
        and record.t3_ns is not None
        and record.t4_ns is not None
        and record.t5_ns is not None
        and record.t_model_runner_execute_start_ns is not None
    )


def _rank_output_path(record: ProfileRecord) -> Path:
    rank_key = (record.pp_rank, record.tp_rank)
    cached = _rank_output_paths.get(rank_key)
    if cached is not None:
        return cached

    base = Path(envs_vllm.VLLM_PROFILE_PP_OPT_OUTPUT_PATH)
    suffix = base.suffix or ".csv"
    stem = base.stem if base.suffix else base.name
    output_path = base.with_name(
        f"{stem}_pp{record.pp_rank}_tp{record.tp_rank}{suffix}"
    )
    _rank_output_paths[rank_key] = output_path
    return output_path


def _sample_tokens_output_path(pp_rank: int, tp_rank: int) -> Path:
    rank_key = (pp_rank, tp_rank)
    cached = _sample_tokens_output_paths.get(rank_key)
    if cached is not None:
        return cached

    base = Path(envs_vllm.VLLM_PROFILE_PP_OPT_OUTPUT_PATH)
    suffix = base.suffix or ".csv"
    stem = base.stem if base.suffix else base.name
    output_path = base.with_name(
        f"{stem}_sample_tokens_pp{pp_rank}_tp{tp_rank}{suffix}"
    )
    _sample_tokens_output_paths[rank_key] = output_path
    return output_path


def _get_csv_writer(
    output_path: Path,
    fieldnames: tuple[str, ...],
) -> csv.DictWriter:
    cached = _csv_writers.get(output_path)
    if cached is not None:
        return cached[1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    output_file = output_path.open(
        "a",
        newline="",
        encoding="utf-8",
        buffering=1,
    )
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    _csv_writers[output_path] = (output_file, writer)
    return writer


def _append_sample_tokens_record(start_ns: int, end_ns: int) -> None:
    pp_rank, tp_rank = _rank_info()
    output_path = _sample_tokens_output_path(pp_rank, tp_rank)

    row = {
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
    }

    writer = _get_csv_writer(output_path, SAMPLE_TOKENS_PROFILE_FIELDS)
    writer.writerow(row)


def _append_record(record: ProfileRecord) -> None:
    assert record.t1_ns is not None
    assert record.t2_ns is not None
    assert record.t3_ns is not None
    assert record.t4_ns is not None
    assert record.t5_ns is not None
    assert record.t_model_runner_execute_start_ns is not None

    output_path = _rank_output_path(record)
    row = {
        "profile_step": record.profile_step,
        "microbatch_id": record.microbatch_id,
        "pp_rank": record.pp_rank,
        "tp_rank": record.tp_rank,
        "layer_num": record.layer_num,
        "request_num": record.request_num,
        "aggregated_ctx_len": record.aggregated_ctx_len,
        "total_scheduled_tokens": record.total_scheduled_tokens,
        "t1_ns": record.t1_ns,
        "t2_ns": record.t2_ns,
        "t3_ns": record.t3_ns,
        "t4_ns": record.t4_ns,
        "t5_ns": record.t5_ns,
        "t_model_runner_execute_start_ns": record.t_model_runner_execute_start_ns,
        "cost_forward_ns": record.t3_ns - record.t2_ns,
        "cost_total_ns": record.t5_ns - record.t1_ns,
        "cost_model_runner_execute": (
            record.t4_ns - record.t_model_runner_execute_start_ns
        ),
    }

    writer = _get_csv_writer(output_path, PROFILE_FIELDS)
    writer.writerow(row)
