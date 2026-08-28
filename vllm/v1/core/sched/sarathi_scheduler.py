# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sarathi-Serve stall-free scheduling component port for V1 vLLM."""

from __future__ import annotations

from vllm.v1.core.sched.scheduler import Scheduler

SARATHI_FIXED_CHUNK_SIZE = 512


class SarathiSchedulerPort(Scheduler):
    """Run V1's decode-first scheduler with Sarathi's fixed chunk budget."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.scheduler_config.enable_chunked_prefill:
            raise ValueError("SarathiSchedulerPort requires chunked prefill")
        if self.max_num_scheduled_tokens != SARATHI_FIXED_CHUNK_SIZE:
            raise ValueError(
                "SarathiSchedulerPort requires max-num-batched-tokens=512"
            )
        self.sarathi_scheduler_receipt = {
            "scheduler_type": "sarathi",
            "chunk_size": SARATHI_FIXED_CHUNK_SIZE,
            "dynamic_chunking_schedule": False,
            "decode_first": True,
        }

    def get_baseline_scheduler_receipt(self) -> dict[str, object]:
        return dict(self.sarathi_scheduler_receipt)
