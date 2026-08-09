# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-light tests for the per-step owner row layout carrier.

Proves the ForwardContext plumbing (default ``None``, exact object identity
inside the context, restoration after exit) and the uninitialized
GPUModelRunner helper that builds the layout only when request-owned
attention is enabled.  No GPU, NPU, distributed group, or real model runner
is constructed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.config import VllmConfig
from vllm.forward_context import (
    create_forward_context,
    get_forward_context,
    set_forward_context,
)
from vllm.v1.core.sched.owner_layout import (
    OwnerLayoutError,
    OwnerRowLayout,
    build_owner_row_layout,
)
from vllm.v1.core.sched.ownership import OwnerLeaseKey, OwnerLeaseToken
from vllm.v1.worker import gpu_model_runner as gpu_model_runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _lease(
    request_id: str,
    owner_id: int,
    runnable_num_tokens: int,
    step_seq: int = 3,
) -> OwnerLeaseToken:
    return OwnerLeaseToken(
        key=OwnerLeaseKey(request_id=request_id, owner_epoch=0),
        owner_id=owner_id,
        step_seq=step_seq,
        command_seq=1,
        runnable_num_tokens=runnable_num_tokens,
    )


def _layout() -> OwnerRowLayout:
    return build_owner_row_layout(
        step_seq=3,
        request_ids=["a", "a", "b"],
        token_positions=[0, 1, 0],
        leases=[_lease("a", 11, 2), _lease("b", 2, 1)],
        group_ranks=(7, 2, 11),
    )


# -- ForwardContext plumbing -------------------------------------------------


def test_create_forward_context_defaults_to_none() -> None:
    ctx = create_forward_context(
        attn_metadata={}, vllm_config=VllmConfig(), slot_mapping={}
    )
    assert ctx.request_owner_layout is None


def test_set_forward_context_identity_and_restoration() -> None:
    layout = _layout()
    with set_forward_context({}, vllm_config=VllmConfig()):
        assert get_forward_context().request_owner_layout is None
        with set_forward_context(
            {}, vllm_config=VllmConfig(), request_owner_layout=layout
        ):
            # Exact object identity inside the active context.
            assert get_forward_context().request_owner_layout is layout
        # Restoration after the inner context exits: back to the outer value.
        assert get_forward_context().request_owner_layout is None
    # The outer context also restores its predecessor (unset -> not available).
    with pytest.raises(AssertionError):
        get_forward_context()


def test_forward_context_explicit_none_is_not_stale() -> None:
    layout = _layout()
    with set_forward_context({}, vllm_config=VllmConfig(), request_owner_layout=layout):
        assert get_forward_context().request_owner_layout is layout
        # A later step without the layout overwrites it rather than leaking.
        with set_forward_context({}, vllm_config=VllmConfig()):
            assert get_forward_context().request_owner_layout is None


# -- GPUModelRunner helper (uninitialized, no device) ------------------------


def _bare_runner(
    enabled: bool, req_ids: list[str], monkeypatch: pytest.MonkeyPatch
) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.scheduler_config = SimpleNamespace(enable_request_owned_attention=enabled)
    runner.input_batch = SimpleNamespace(req_ids=req_ids)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=(7, 2, 11)),
    )
    return runner


def test_runner_helper_disabled_returns_none(monkeypatch) -> None:
    runner = _bare_runner(False, ["a", "b"], monkeypatch)
    scheduler_output = SimpleNamespace(step_seq=3, scheduled_owner_leases=[])
    req_indices = np.array([0, 1], dtype=np.int64)
    positions_np = np.array([0, 0], dtype=np.int64)
    assert (
        runner._build_request_owner_layout(scheduler_output, req_indices, positions_np)
        is None
    )


def test_runner_helper_enabled_builds_exact_layout(monkeypatch) -> None:
    # req_ids are the input-batch rows; req_indices is the true flattened
    # execution-row order (request b before request a here).
    runner = _bare_runner(True, ["a", "b", "c"], monkeypatch)
    scheduler_output = SimpleNamespace(
        step_seq=3,
        scheduled_owner_leases=[
            _lease("c", 7, 3),
            _lease("a", 11, 3),
            _lease("b", 2, 3),
        ],
    )
    req_indices = np.array([1, 0, 1, 0, 1, 2, 2], dtype=np.int64)
    positions_np = np.array([0, 0, 1, 1, 2, 0, 1], dtype=np.int64)
    layout = runner._build_request_owner_layout(
        scheduler_output, req_indices, positions_np
    )
    assert isinstance(layout, OwnerRowLayout)
    assert layout.step_seq == 3
    assert layout.group_ranks == (7, 2, 11)
    assert [row.row_id.request_uid.request_id for row in layout.global_rows] == [
        "b",
        "a",
        "b",
        "a",
        "b",
        "c",
        "c",
    ]
    assert layout.owner_counts == (2, 3, 2)
    # Matches the directly built layout field for field.
    expected = build_owner_row_layout(
        step_seq=3,
        request_ids=["b", "a", "b", "a", "b", "c", "c"],
        token_positions=[0, 0, 1, 1, 2, 0, 1],
        leases=scheduler_output.scheduled_owner_leases,
        group_ranks=(7, 2, 11),
    )
    assert layout.global_rows == expected.global_rows
    assert layout.owner_rows == expected.owner_rows
    assert layout.owner_counts == expected.owner_counts
    assert layout.forward_permutation == expected.forward_permutation


def test_runner_helper_fails_closed_on_bad_leases(monkeypatch) -> None:
    runner = _bare_runner(True, ["a"], monkeypatch)
    # Position at/beyond the lease's exclusive bound must fail closed.
    scheduler_output = SimpleNamespace(
        step_seq=3,
        scheduled_owner_leases=[_lease("a", 11, 0)],
    )
    with pytest.raises(OwnerLayoutError):
        runner._build_request_owner_layout(
            scheduler_output,
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        )
