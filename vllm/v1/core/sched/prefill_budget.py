# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def cap_scheduled_tokens(
    num_new_tokens: int,
    long_prefill_token_threshold: int,
    token_budget: int,
) -> int:
    """Apply the per-request chunk cap and remaining step budget."""
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold
    return min(num_new_tokens, token_budget)
