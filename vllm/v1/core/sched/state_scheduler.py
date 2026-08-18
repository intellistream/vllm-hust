# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped state-aware scheduler used by the State Scheduler topic.

This module intentionally lives on the topic feature carrier.  It consumes
request metadata that has already crossed the OpenAI ``vllm_xargs`` boundary,
acts through vLLM's native priority queue, and returns a receipt through the
existing ``kv_transfer_params`` response extension.  It is not enabled by
default and does not alter the shared runtime mainline.
"""

from __future__ import annotations

import math
import os
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

STATE_SCHEDULER_POLICY_ENV = "STATE_SCHEDULER_POLICY"
STATE_SCHEDULER_RECEIPT_KEY = "state_scheduler_policy_receipt"
STATE_SCHEDULER_RECEIPT_SCHEMA = "state-scheduler-policy-receipt/v1"
SUPPORTED_STATE_SCHEDULER_ARMS = (
    "lru",
    "oracle",
    "workflow_scheduler",
)


def _request_args(request: Request) -> dict[str, Any]:
    params = request.sampling_params
    if params is None or params.extra_args is None:
        return {}
    return dict(params.extra_args)


def _require_string(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_nonnegative_number(args: dict[str, Any], key: str) -> float:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return value


def _external_request_id_matches(internal_id: str, external_id: str) -> bool:
    """Bind the caller ID to vLLM's completion request identity.

    The OpenAI completion serving path materializes a single-prompt request as
    ``cmpl-<request_id>-0``.  Direct engine users may supply the exact ID.
    """

    return internal_id == external_id or internal_id == f"cmpl-{external_id}-0"


def build_state_scheduler_decision(
    request: Request,
    *,
    effective_arm: str,
) -> tuple[int, dict[str, Any]]:
    """Validate request metadata and produce a native priority + receipt."""

    if effective_arm not in SUPPORTED_STATE_SCHEDULER_ARMS:
        raise ValueError(f"unsupported state scheduler arm: {effective_arm!r}")

    args = _request_args(request)
    requested_arm = _require_string(args, "state_scheduler_arm")
    if requested_arm != effective_arm:
        raise ValueError(
            "state_scheduler_arm does not match the endpoint's effective policy"
        )

    external_request_id = _require_string(args, "state_scheduler_request_id")
    if not _external_request_id_matches(request.request_id, external_request_id):
        raise ValueError("state scheduler request identity does not match vLLM request")

    reuse_tokens = _require_nonnegative_number(args, "state_scheduler_reuse_tokens")
    latency_budget_ms = _require_nonnegative_number(
        args, "state_scheduler_latency_budget_ms"
    )
    if latency_budget_ms == 0:
        raise ValueError("state_scheduler_latency_budget_ms must be greater than zero")
    oracle_score = _require_nonnegative_number(args, "state_scheduler_oracle_score")

    if effective_arm == "lru":
        decision_score = 0.0
        priority = 0
        decision_path = ["lru", "arrival-order"]
    elif effective_arm == "oracle":
        decision_score = oracle_score
        priority = -round(decision_score * 1_000_000)
        decision_path = ["oracle", "frozen-oracle-score", "native-priority-queue"]
    else:
        decision_score = reuse_tokens / latency_budget_ms
        priority = -round(decision_score * 1_000_000)
        decision_path = [
            "workflow_scheduler",
            "reuse-tokens-per-latency-budget",
            "native-priority-queue",
        ]

    receipt = {
        "schema": STATE_SCHEDULER_RECEIPT_SCHEMA,
        "request_id": external_request_id,
        "effective_arm": effective_arm,
        "decision_path": decision_path,
        "decision_score": decision_score,
        "oracle_score": oracle_score,
        "recompute_blocks": 0,
        "policy_regret": abs(oracle_score - decision_score),
        "runtime_metrics_bound": False,
    }
    return priority, receipt


class StateScheduler(Scheduler):
    """Native vLLM scheduler with request-scoped state-benefit ordering."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        effective_arm = os.environ.get(STATE_SCHEDULER_POLICY_ENV, "").strip()
        if effective_arm not in SUPPORTED_STATE_SCHEDULER_ARMS:
            raise ValueError(
                f"{STATE_SCHEDULER_POLICY_ENV} must be one of "
                f"{SUPPORTED_STATE_SCHEDULER_ARMS}"
            )
        self.state_scheduler_policy = effective_arm
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.PRIORITY:
            raise ValueError(
                "StateScheduler requires the native vLLM priority scheduling policy"
            )

    def add_request(self, request: Request) -> None:
        priority, receipt = build_state_scheduler_decision(
            request,
            effective_arm=self.state_scheduler_policy,
        )
        request.priority = priority
        setattr(request, STATE_SCHEDULER_RECEIPT_KEY, receipt)
        super().add_request(request)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        output = super().schedule(throttle_prefills=throttle_prefills)
        for request_id in output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is None:
                continue
            receipt = getattr(request, STATE_SCHEDULER_RECEIPT_KEY, None)
            prefill_stats = request.prefill_stats
            if (
                not isinstance(receipt, dict)
                or receipt.get("runtime_metrics_bound") is True
                or prefill_stats is None
                or prefill_stats.num_prompt_tokens <= 0
            ):
                continue
            receipt["recompute_blocks"] = math.ceil(
                prefill_stats.num_computed_tokens / self.block_size
            )
            receipt["decision_path"] = [
                *receipt["decision_path"],
                "native-prefix-cache-stats",
            ]
            receipt["runtime_metrics_bound"] = True
        return output

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        receipt = getattr(request, STATE_SCHEDULER_RECEIPT_KEY, None)
        params = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        if not isinstance(receipt, dict):
            return params
        response_params = dict(params or {})
        response_params[STATE_SCHEDULER_RECEIPT_KEY] = dict(receipt)
        return response_params
