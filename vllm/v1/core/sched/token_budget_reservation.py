# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

TOKEN_BUDGET_PREFILL_RESERVATION_ENV = (
    "VLLM_TOKEN_BUDGET_PRIORITY_PREFILL_RESERVATION_TOKENS"
)


def read_priority_prefill_reservation_tokens() -> int:
    """Read the strict opt-in priority-prefill reservation."""
    raw_value = os.getenv(TOKEN_BUDGET_PREFILL_RESERVATION_ENV)
    if raw_value is None or raw_value == "":
        return 0
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{TOKEN_BUDGET_PREFILL_RESERVATION_ENV} must be an integer"
        ) from exc
    if value < 0:
        raise ValueError(f"{TOKEN_BUDGET_PREFILL_RESERVATION_ENV} must be non-negative")
    return value


def priority_prefill_reservation_tokens(
    *,
    configured_tokens: int,
    scheduling_policy: str,
    request_priority: int,
    remaining_prefill_tokens: int,
    max_step_tokens: int,
    enable_chunked_prefill: bool,
) -> int:
    """Return a safe native token reservation for a selected prefill."""
    if (
        configured_tokens <= 0
        or scheduling_policy != "priority"
        or request_priority >= 0
        or remaining_prefill_tokens <= 0
        or max_step_tokens <= 0
    ):
        return 0

    reserved_tokens = min(
        configured_tokens,
        remaining_prefill_tokens,
        max_step_tokens,
    )
    if not enable_chunked_prefill and reserved_tokens < remaining_prefill_tokens:
        return 0
    return reserved_tokens
