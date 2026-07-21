# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QoS-aware request ordering.

Follows the stock FCFS token-budget model: prefill and decode requests
share a single global ``token_budget`` per step.  The QoS controller only
reorders requests — decode work is placed before prefill while any QoS
request has an active deadline — and tracks per-request SLO violations.
No independent prefill budget is applied.
"""

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.metrics.stats import QoSPolicyStats

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class QoSStepDecision:
    """QoS controller decision for one synchronous scheduler step."""

    active: bool
    total_token_budget: int
    min_decode_slack_ms: float | None = None


class QoSSchedulingPolicy:
    """Controller used by the stock V1 scheduler when QoS is enabled.

    Provides deadline-aware queue ordering.  Prefill and decode tokens
    share the scheduler's global ``max_num_scheduled_tokens`` budget
    (the same as FCFS); QoS only changes *which* requests run first.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        config = vllm_config.scheduler_config
        self.enabled = config.enable_qos_scheduling
        self.base_priority_enabled = config.policy == "priority"
        self.max_num_scheduled_tokens = config.max_num_scheduled_tokens or (
            config.max_num_batched_tokens
        )
        self.hybrid_alpha_s_per_token = config.qos_hybrid_alpha

        self.last_decision = self._inactive_decision()
        self._ttft_observations = 0
        self._tbt_observations = 0
        self._ttlt_observations = 0
        self._ttft_violations = 0
        self._tbt_violations = 0
        self._ttlt_violations = 0

        # Counter for O(1) QoS-request existence check.
        self._qos_request_count: int = 0

        if not self.enabled:
            return
        self._validate_supported_runtime(vllm_config)
        logger.info("QoS scheduling enabled (FCFS token-budget model).")

    def _inactive_decision(self) -> QoSStepDecision:
        return QoSStepDecision(
            active=False,
            total_token_budget=self.max_num_scheduled_tokens,
        )

    @staticmethod
    def _validate_supported_runtime(vllm_config: "VllmConfig") -> None:
        scheduler_config = vllm_config.scheduler_config
        parallel_config = vllm_config.parallel_config
        unsupported: list[str] = []
        if scheduler_config.runner_type != "generate":
            unsupported.append("non-generative runners")
        if scheduler_config.is_multimodal_model:
            unsupported.append("multimodal models")
        if parallel_config.pipeline_parallel_size != 1:
            unsupported.append("pipeline parallelism")
        if parallel_config.data_parallel_size != 1:
            unsupported.append("data parallelism")
        if parallel_config.prefill_context_parallel_size != 1:
            unsupported.append("prefill context parallelism")
        if parallel_config.decode_context_parallel_size != 1:
            unsupported.append("decode context parallelism")
        if parallel_config.use_ubatching:
            unsupported.append("ubatching/DBO")
        if parallel_config.enable_expert_parallel:
            unsupported.append("expert parallelism")
        if vllm_config.speculative_config is not None:
            unsupported.append("speculative decoding")
        if vllm_config.kv_transfer_config is not None:
            unsupported.append("KV-transfer/P-D disaggregation")
        if vllm_config.lora_config is not None:
            unsupported.append("LoRA")
        if vllm_config.model_config.logits_processors:
            unsupported.append("model-level logits processors")
        if vllm_config.model_config.enable_return_routed_experts:
            unsupported.append("routed-expert return data")
        if unsupported:
            raise ValueError(
                "QoS scheduling Phase 0-3 does not support: "
                + ", ".join(unsupported)
                + "."
            )

    @staticmethod
    def has_qos_request(requests: Iterable["Request"]) -> bool:
        return any(
            request.qos_state is not None and request.qos_state.has_active_deadline()
            for request in requests
        )

    def increment_qos_count(self) -> None:
        self._qos_request_count += 1

    def decrement_qos_count(self) -> None:
        if self._qos_request_count > 0:
            self._qos_request_count -= 1

    @staticmethod
    def _remaining_work(request: "Request") -> int:
        state = request.qos_state
        expected_output_tokens = (
            state.expected_output_tokens if state is not None else request.max_tokens
        )
        expected_remaining_output = max(
            0, expected_output_tokens - request.num_output_tokens
        )
        remaining_prefill = max(
            0, request.num_prompt_tokens - request.num_computed_tokens
        )
        return remaining_prefill + expected_remaining_output

    def _hybrid_score(self, deadline: float, remaining_work: int) -> float:
        if not math.isfinite(deadline):
            return math.inf
        try:
            penalty = self.hybrid_alpha_s_per_token * remaining_work
        except OverflowError:
            return math.inf
        score = deadline + penalty
        return score if math.isfinite(score) else math.inf

    def waiting_key(self, request: "Request") -> tuple[Any, ...]:
        priority = request.priority if self.base_priority_enabled else 0
        if request.qos_state is None or not request.qos_state.has_active_deadline():
            if self.base_priority_enabled:
                return (priority, 1, 0.0, request.arrival_time, request.request_id)
            return (priority, 1, 0.0, 0.0, "")
        hybrid_score = self._hybrid_score(
            request.qos_state.waiting_deadline(),
            self._remaining_work(request),
        )
        return (
            priority,
            0,
            hybrid_score,
            request.arrival_time,
            request.request_id,
        )

    def running_key(self, request: "Request") -> tuple[Any, ...]:
        is_prefill = request.num_computed_tokens < request.num_prompt_tokens
        priority = request.priority if self.base_priority_enabled else 0
        deadline = (
            math.inf
            if request.qos_state is None
            else request.qos_state.next_token_deadline()
        )
        hybrid_score = self._hybrid_score(deadline, self._remaining_work(request))
        return (
            int(is_prefill),
            priority,
            hybrid_score,
            request.arrival_time,
            request.request_id,
        )

    def order_running(
        self, requests: list["Request"], *, active: bool | None = None
    ) -> None:
        if self.enabled and (
            self.has_qos_request(requests) if active is None else active
        ):
            requests.sort(key=self.running_key)

    @staticmethod
    def _decode_demand(request: "Request") -> int:
        if request.num_computed_tokens < request.num_prompt_tokens:
            return 0
        return max(
            0,
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens,
        )

    def start_step(
        self,
        *,
        running: list["Request"],
        waiting: Iterable["Request"],
        now: float | None = None,
    ) -> QoSStepDecision:
        """Determine whether any active QoS work exists this step.

        Does **not** compute a per-step prefill budget — prefill and decode
        share the scheduler's global ``max_num_scheduled_tokens``, following
        the same FCFS model as the stock V1 scheduler.
        """

        now = time.monotonic() if now is None else now
        if not self.enabled:
            self.last_decision = self._inactive_decision()
            return self.last_decision

        if self._qos_request_count == 0:
            self.last_decision = self._inactive_decision()
            return self.last_decision

        all_requests = [*running, *list(waiting)]
        if not self.has_qos_request(all_requests):
            self.last_decision = self._inactive_decision()
            return self.last_decision

        decode_requests = [
            request for request in running if self._decode_demand(request) > 0
        ]
        qos_decode_deadlines = [
            request.qos_state.next_token_deadline()
            for request in decode_requests
            if request.qos_state is not None
            and math.isfinite(request.qos_state.next_token_deadline())
        ]
        min_decode_slack_ms = (
            None
            if not qos_decode_deadlines
            else (min(qos_decode_deadlines) - now) * 1000.0
        )

        self.last_decision = QoSStepDecision(
            active=True,
            total_token_budget=self.max_num_scheduled_tokens,
            min_decode_slack_ms=min_decode_slack_ms,
        )
        return self.last_decision

    def observe_request_output(
        self,
        request: "Request",
        *,
        num_new_tokens: int,
        finished: bool,
        now: float | None = None,
    ) -> None:
        if not self.enabled or request.qos_state is None:
            return
        now = time.monotonic() if now is None else now
        state = request.qos_state
        had_first_token = state.first_token_time is not None
        if num_new_tokens > 0:
            if not had_first_token and state.ttft_deadline is not None:
                self._ttft_observations += 1
            if state.tbt_slo_s is not None:
                self._tbt_observations += max(
                    0,
                    num_new_tokens - int(not had_first_token),
                )
        ttft, tbt = request.qos_state.observe_tokens(num_new_tokens, now)
        self._ttft_violations += ttft
        self._tbt_violations += tbt
        if finished:
            if state.ttlt_deadline is not None and not state.ttlt_observed:
                self._ttlt_observations += 1
            self._ttlt_violations += request.qos_state.observe_finished(now)
            self.decrement_qos_count()

    def drain_stats(self) -> QoSPolicyStats | None:
        if not self.enabled:
            return None
        decision = self.last_decision
        stats = QoSPolicyStats(
            active=int(decision.active),
            total_token_budget=decision.total_token_budget,
            min_decode_slack_ms=(
                math.nan
                if decision.min_decode_slack_ms is None
                else decision.min_decode_slack_ms
            ),
            ttft_observations=self._ttft_observations,
            tbt_observations=self._tbt_observations,
            ttlt_observations=self._ttlt_observations,
            ttft_violations=self._ttft_violations,
            tbt_violations=self._tbt_violations,
            ttlt_violations=self._ttlt_violations,
        )
        self._ttft_observations = 0
        self._tbt_observations = 0
        self._ttlt_observations = 0
        self._ttft_violations = 0
        self._tbt_violations = 0
        self._ttlt_violations = 0
        return stats
