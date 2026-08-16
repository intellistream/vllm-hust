# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure physical-capacity projections for request-owned admission."""

from vllm.v1.core.sched.ownership import OwnerCachePoolSnapshot


def owner_projected_block_demand(
    snapshot: OwnerCachePoolSnapshot,
    required_num_tokens: int,
) -> int:
    """Project unified-pool blocks for a fresh DSV4 horizon.

    Every request owns one table in every KV group. Heterogeneous effective
    token capacities and compressed groups floor logical tokens to a storage
    quantum before block rounding, while every group consumes the same pool.
    """

    if required_num_tokens <= 0:
        return 0
    if not snapshot.groups:
        return 1
    demand = 0
    for group in snapshot.groups:
        quantum = group.allocation_token_quantum
        storage_tokens = required_num_tokens // quantum
        storage_tokens_per_block = group.effective_tokens_per_block // quantum
        group_demand = (
            storage_tokens + storage_tokens_per_block - 1
        ) // storage_tokens_per_block
        if group.fresh_allocation_block_cap is not None:
            group_demand = min(
                group_demand,
                group.fresh_allocation_block_cap,
            )
        demand += group_demand
    return demand
