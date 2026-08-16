# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level reset composition for worker-owned physical prefix caches."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore


def reset_request_owned_prefix_cache(
    engine: "EngineCore",
    reset_running_requests: bool,
    reset_connector: bool,
) -> bool:
    """Reset scheduler state, then every owner-local physical hash index.

    The scheduler cache is only the logical half of request-owned attention.
    Weight updates and explicit resets must not report success while any rank
    still exposes prefix blocks computed by the prior model epoch.
    """

    config = engine.vllm_config
    request_owned_prefix = bool(
        config.scheduler_config.enable_request_owned_attention
        and config.cache_config.enable_prefix_caching
    )
    if request_owned_prefix:
        reset_successful = engine.scheduler._reset_prefix_cache_local(
            reset_running_requests, reset_connector
        )
    else:
        reset_successful = engine.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )
    if reset_successful and request_owned_prefix:
        reset_successful = (
            engine.model_executor.reset_request_owned_prefix_cache()
        )
    if reset_running_requests and not reset_successful:
        raise RuntimeError(
            "Failed to reset request-owned prefix caches after force "
            "preemption. Drain owner control/release work before retrying."
        )
    return reset_successful


def require_prefix_reset(reset_successful: bool) -> None:
    """Turn a best-effort public reset result into a cache-clear fence."""

    if not reset_successful:
        raise RuntimeError(
            "Cannot clear prefix cache while request-owned physical leases "
            "or control receipts remain active."
        )
