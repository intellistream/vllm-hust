# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_EVERY_N_STEPS = 1
_DEFAULT_MAX_RECORDS = 0


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _read_int_env(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(getattr(envs, name)))
    except ValueError:
        logger.warning("Ignoring invalid %s; using %d", name, default)
        return default


class AdaptiveStateProbe:
    """Optional JSONL probe for model-runner scheduling state."""

    def __init__(self, jsonl_path: Path, every_n_steps: int, max_records: int):
        self._jsonl_path = jsonl_path
        self._every_n_steps = max(1, every_n_steps)
        self._max_records = max(0, max_records)
        self._step = 0
        self._written = 0
        self._warned = False
        self._disabled = False

        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> AdaptiveStateProbe | None:
        jsonl = envs.VLLM_ADAPTIVE_STATE_PROBE_JSONL.strip()
        if not jsonl:
            return None

        every_n_steps = _read_int_env(
            "VLLM_ADAPTIVE_STATE_PROBE_EVERY",
            _DEFAULT_EVERY_N_STEPS,
            minimum=1,
        )
        max_records = _read_int_env(
            "VLLM_ADAPTIVE_STATE_PROBE_MAX_RECORDS",
            _DEFAULT_MAX_RECORDS,
            minimum=0,
        )
        try:
            probe = cls(
                Path(jsonl),
                every_n_steps=every_n_steps,
                max_records=max_records,
            )
        except OSError:
            logger.warning(
                "AdaptiveStateProbe could not initialize path %s; disabling probe",
                jsonl,
                exc_info=True,
            )
            return None
        logger.info(
            "AdaptiveStateProbe enabled: path=%s every=%d max_records=%d",
            str(probe._jsonl_path),
            probe._every_n_steps,
            probe._max_records,
        )
        return probe

    def record_step(
        self,
        *,
        scheduler_output: Any,
        batch_desc: Any,
        max_query_len: int,
        uniform_tok_count: int,
        dummy_run: bool,
        skip_compiled: bool,
    ) -> None:
        if self._disabled:
            return
        self._step += 1
        if self._step % self._every_n_steps != 0:
            return
        if self._max_records and self._written >= self._max_records:
            return

        try:
            num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})
            if hasattr(num_scheduled_tokens, "values"):
                scheduled_values = list(num_scheduled_tokens.values())
            else:
                scheduled_values = []

            num_decode_reqs = sum(1 for tok in scheduled_values if tok == 1)
            num_prefill_reqs = sum(1 for tok in scheduled_values if tok > 1)

            row = {
                "time_ns": time.time_ns(),
                "step": self._step,
                "dummy_run": bool(dummy_run),
                "skip_compiled": bool(skip_compiled),
                "num_reqs": _safe_len(num_scheduled_tokens),
                "num_tokens": int(
                    getattr(scheduler_output, "total_num_scheduled_tokens", 0)
                ),
                "max_query_len": int(max_query_len),
                "uniform_tok_count": int(uniform_tok_count),
                "num_decode_reqs": num_decode_reqs,
                "num_prefill_reqs": num_prefill_reqs,
                "num_new_reqs": _safe_len(
                    getattr(scheduler_output, "scheduled_new_reqs", None)
                ),
                "num_cached_reqs": _safe_len(
                    getattr(scheduler_output, "scheduled_cached_reqs", None)
                ),
                "num_finished_reqs": _safe_len(
                    getattr(scheduler_output, "finished_req_ids", None)
                ),
                "num_preempted_reqs": _safe_len(
                    getattr(scheduler_output, "preempted_req_ids", None)
                ),
                "has_structured_output_requests": bool(
                    getattr(scheduler_output, "has_structured_output_requests", False)
                ),
                "has_encoder_inputs": bool(
                    getattr(scheduler_output, "scheduled_encoder_inputs", None)
                ),
                "num_spec_decode_reqs": _safe_len(
                    getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
                ),
                "batch_num_tokens": int(getattr(batch_desc, "num_tokens", 0)),
                "batch_num_reqs": int(getattr(batch_desc, "num_reqs", 0) or 0),
                "batch_cg_mode": str(
                    getattr(getattr(batch_desc, "cg_mode", None), "name", "unknown")
                ),
            }

            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            self._written += 1
        except Exception:
            if not self._warned:
                logger.warning(
                    "AdaptiveStateProbe write failed once; "
                    "suppressing further warnings",
                    exc_info=True,
                )
                self._warned = True
            self._disabled = True
