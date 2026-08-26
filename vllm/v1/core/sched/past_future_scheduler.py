# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LightLLM Past-Future Scheduler component port for V1 vLLM."""

from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.past_future_policy import (
    PastFutureAdmissionDecision,
    PastFuturePolicy,
    PastFutureRequestState,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

PAST_FUTURE_SEED = 20260816


class PastFutureSchedulerPort(Scheduler):
    """Admit waiting requests using Past-Future's history-based KV estimate."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._past_future_policy = PastFuturePolicy(seed=PAST_FUTURE_SEED)
        self._past_future_decisions: list[PastFutureAdmissionDecision] = []
        self._past_future_max_running_reqs = self.max_num_running_reqs

    @staticmethod
    def _state(request: Request) -> PastFutureRequestState:
        return PastFutureRequestState(
            request_id=request.request_id,
            computed_tokens=request.num_computed_tokens,
            completed_output_tokens=len(request.output_token_ids),
            max_output_tokens=request.max_tokens,
        )

    def _waiting_admission_limit(self) -> int:
        running = [self._state(request) for request in self.running]
        admitted = 0
        self._past_future_decisions = []
        for request in self.waiting:
            if len(running) >= self._past_future_max_running_reqs:
                break
            decision = self._past_future_policy.decide(
                running=running,
                candidate=self._state(request),
                max_kv_tokens=self.max_num_kv_tokens,
            )
            self._past_future_decisions.append(decision)
            if not decision.admitted:
                break
            running.append(
                PastFutureRequestState(
                    request_id=request.request_id,
                    computed_tokens=request.num_computed_tokens,
                    completed_output_tokens=0,
                    max_output_tokens=request.max_tokens,
                )
            )
            admitted += 1
        return admitted

    def schedule(self):
        # Apply the policy only to new waiting admissions. Existing requests
        # retain native V1 scheduling and preemption semantics.
        original_limit = self.max_num_running_reqs
        allowed = len(self.running) + self._waiting_admission_limit()
        self.max_num_running_reqs = min(original_limit, allowed)
        try:
            return super().schedule()
        finally:
            self.max_num_running_reqs = original_limit

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        status = request.status
        output_tokens = len(request.output_token_ids)
        result = super()._free_request(request, delay_free_blocks)
        # Only successful generation terminals enter the empirical history.
        # Aborts, ignored prompts, transport/runtime errors, and streaming
        # transitions are not observations of a completed output distribution.
        if status in {
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_REPETITION,
        }:
            self._past_future_policy.record_completed_output(output_tokens)
        return result

    def get_baseline_scheduler_receipt(self) -> dict[str, object]:
        return {
            "scheduler_type": "past_future",
            "seed": PAST_FUTURE_SEED,
            "history_output_tokens": list(
                self._past_future_policy.history_output_tokens
            ),
            "last_admission_decisions": [
                decision.to_dict() for decision in self._past_future_decisions
            ],
        }
