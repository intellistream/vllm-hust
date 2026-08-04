# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import regex as re

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.utils import compute_iteration_details

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

logger = init_logger(__name__)

_SCHEMA_VERSION = 1
_DEFAULT_EVERY_N_STEPS = 64
_DEFAULT_MAX_RECORDS = 10_000
_DEFAULT_MAX_BYTES = 16 * 1024 * 1024
_DEFAULT_FLUSH_EVERY = 64
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_BATCH_LAYER_ID = "__batch__"
_SCHEDULER_OPERATOR_ID = "__scheduler_step__"


def _read_int_env(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(getattr(envs, name)))
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s; using %d", name, default)
        return default


@dataclass(frozen=True)
class ProbeTopology:
    world_rank: int | None = 0
    world_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1


class AdaptiveStateProbe:
    """Bounded JSONL probe for anonymous model-runner scheduling state."""

    def __init__(
        self,
        jsonl_path: Path,
        *,
        topology: ProbeTopology,
        every_n_steps: int,
        max_records: int,
        max_bytes: int,
        flush_every: int,
        run_id: str,
        pid: int | None = None,
    ):
        self._topology = topology
        self._every_n_steps = max(1, every_n_steps)
        self._max_records = max(0, max_records)
        self._max_bytes = max(0, max_bytes)
        self._flush_every = max(1, flush_every)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id must be a non-secret telemetry identifier")
        if (
            topology.world_rank is None
            or topology.world_size < 1
            or not 0 <= topology.world_rank < topology.world_size
        ):
            raise ValueError("valid world rank topology is required")
        self._run_id = run_id
        self._pid = os.getpid() if pid is None else pid
        self._step = 0
        self._written = 0
        self._bytes_written = 0
        self._pending_flush = 0
        self._dropped_cadence = 0
        self._dropped_record_limit = 0
        self._dropped_byte_limit = 0
        self._dropped_summary_only = 0
        self._dropped_io = 0
        self._warned = False
        self._disabled = False
        self._closed = False
        self._summary_only = self._max_records == 0 or self._max_bytes == 0

        suffix = jsonl_path.suffix or ".jsonl"
        self.output_path = jsonl_path.with_name(
            f"{jsonl_path.stem}.{self._run_id}.rank-{topology.world_rank}."
            f"pid-{self._pid}{suffix}"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.output_path.open(
            "x", encoding="utf-8", buffering=64 * 1024
        )
        atexit.register(self.close)

    @classmethod
    def from_env(cls, topology: ProbeTopology) -> AdaptiveStateProbe | None:
        jsonl = envs.VLLM_ADAPTIVE_STATE_PROBE_JSONL.strip()
        if not jsonl:
            return None
        run_id = envs.VLLM_TELEMETRY_RUN_ID.strip()
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            logger.warning(
                "AdaptiveStateProbe requires a valid VLLM_TELEMETRY_RUN_ID; "
                "disabling probe"
            )
            return None
        if topology.world_rank is None:
            logger.warning("AdaptiveStateProbe requires a global rank; disabling probe")
            return None
        try:
            all_ranks = envs.VLLM_ADAPTIVE_STATE_PROBE_ALL_RANKS
        except ValueError:
            logger.warning(
                "Ignoring invalid VLLM_ADAPTIVE_STATE_PROBE_ALL_RANKS; "
                "using rank-zero ownership"
            )
            all_ranks = False
        if not all_ranks and topology.world_rank != 0:
            return None

        try:
            probe = cls(
                Path(jsonl),
                topology=topology,
                every_n_steps=_read_int_env(
                    "VLLM_ADAPTIVE_STATE_PROBE_EVERY",
                    _DEFAULT_EVERY_N_STEPS,
                    minimum=1,
                ),
                max_records=_read_int_env(
                    "VLLM_ADAPTIVE_STATE_PROBE_MAX_RECORDS",
                    _DEFAULT_MAX_RECORDS,
                    minimum=0,
                ),
                max_bytes=_read_int_env(
                    "VLLM_ADAPTIVE_STATE_PROBE_MAX_BYTES",
                    _DEFAULT_MAX_BYTES,
                    minimum=0,
                ),
                flush_every=_read_int_env(
                    "VLLM_ADAPTIVE_STATE_PROBE_FLUSH_EVERY",
                    _DEFAULT_FLUSH_EVERY,
                    minimum=1,
                ),
                run_id=run_id,
            )
        except (OSError, ValueError):
            logger.warning("AdaptiveStateProbe could not initialize; disabling probe")
            return None
        logger.info(
            "AdaptiveStateProbe enabled: every=%d max_records=%d max_bytes=%d",
            probe._every_n_steps,
            probe._max_records,
            probe._max_bytes,
        )
        return probe

    @property
    def drop_counts(self) -> dict[str, int]:
        return {
            "cadence": self._dropped_cadence,
            "record_limit": self._dropped_record_limit,
            "byte_limit": self._dropped_byte_limit,
            "summary_only": self._dropped_summary_only,
            "io": self._dropped_io,
        }

    def _disable_after_io_error(self) -> None:
        self._dropped_io += 1
        self._disabled = True
        if not self._warned:
            logger.warning("AdaptiveStateProbe I/O failed; disabling probe")
            self._warned = True
        if self._file is not None:
            with suppress(OSError):
                self._file.close()
            self._file = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        if self._file is None:
            return
        try:
            summary = {
                **self._common_fields(),
                "record_type": "summary",
                "observed_steps": self._step,
                "records_written": self._written,
                "record_bytes": self._bytes_written,
                "dropped": self.drop_counts,
            }
            self._file.write(json.dumps(summary, sort_keys=True) + "\n")
            self._file.flush()
            self._file.close()
        except OSError:
            self._disable_after_io_error()
        else:
            self._file = None

    def _common_fields(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self._run_id,
            "rank": self._topology.world_rank,
            "world_size": self._topology.world_size,
            "pid": self._pid,
            "timestamp_ns": time.time_ns(),
            "monotonic_step": self._step,
            "layer_id": _BATCH_LAYER_ID,
            "operator_id": _SCHEDULER_OPERATOR_ID,
        }

    def record_step(
        self,
        *,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
        max_query_len: int,
        uniform_tok_count: int | None,
        dummy_run: bool,
        skip_compiled: bool,
    ) -> None:
        if self._disabled:
            return
        self._step += 1
        if self._step % self._every_n_steps != 0:
            self._dropped_cadence += 1
            return
        if self._summary_only:
            self._dropped_summary_only += 1
            return
        if self._written >= self._max_records:
            self._dropped_record_limit += 1
            return

        try:
            iteration = compute_iteration_details(scheduler_output)
            cg_mode = (
                batch_desc.cg_mode.name
                if isinstance(batch_desc.cg_mode, CUDAGraphMode)
                else "UNKNOWN"
            )
            row = {
                **self._common_fields(),
                "record_type": "sample",
                "topology": {
                    "world_rank": self._topology.world_rank,
                    "world_size": self._topology.world_size,
                    "dp_rank": self._topology.dp_rank,
                    "dp_size": self._topology.dp_size,
                    "pp_rank": self._topology.pp_rank,
                    "pp_size": self._topology.pp_size,
                    "tp_rank": self._topology.tp_rank,
                    "tp_size": self._topology.tp_size,
                },
                "dropped_before": self.drop_counts,
                "dummy_run": bool(dummy_run),
                "skip_compiled": bool(skip_compiled),
                "num_reqs": len(scheduler_output.scheduled_new_reqs)
                + scheduler_output.scheduled_cached_reqs.num_reqs,
                "num_tokens": scheduler_output.total_num_scheduled_tokens,
                "max_query_len": int(max_query_len),
                "uniform_tok_count": uniform_tok_count,
                "num_context_reqs": iteration.num_ctx_requests,
                "num_context_tokens": iteration.num_ctx_tokens,
                "num_generation_reqs": iteration.num_generation_requests,
                "num_generation_tokens": iteration.num_generation_tokens,
                "num_new_reqs": len(scheduler_output.scheduled_new_reqs),
                "num_cached_reqs": scheduler_output.scheduled_cached_reqs.num_reqs,
                "num_finished_reqs": len(scheduler_output.finished_req_ids),
                "num_preempted_reqs": len(scheduler_output.preempted_req_ids or ()),
                "has_structured_output_requests": (
                    scheduler_output.has_structured_output_requests
                ),
                "has_encoder_inputs": bool(scheduler_output.scheduled_encoder_inputs),
                "num_spec_decode_reqs": len(
                    scheduler_output.scheduled_spec_decode_tokens
                ),
                "num_spec_decode_tokens": sum(
                    len(tokens)
                    for tokens in scheduler_output.scheduled_spec_decode_tokens.values()
                ),
                "batch_num_tokens": batch_desc.num_tokens,
                "batch_num_reqs": batch_desc.num_reqs,
                "batch_cg_mode": cg_mode,
            }
            serialized = json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
            record_bytes = len(serialized.encode())
            if self._bytes_written + record_bytes > self._max_bytes:
                self._dropped_byte_limit += 1
                return
            assert self._file is not None
            self._file.write(serialized)
            self._written += 1
            self._bytes_written += record_bytes
            self._pending_flush += 1
            if self._pending_flush >= self._flush_every:
                self._file.flush()
                self._pending_flush = 0
        except (OSError, TypeError, ValueError):
            self._disable_after_io_error()
