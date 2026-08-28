# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-free tests for joined source-backed mechanism telemetry."""

from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.engine.core import _QBI_MECHANISM_RECEIPT_LIMIT, EngineCore
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _core() -> EngineCore:
    core = object.__new__(EngineCore)
    core._runtime_control_boot_id = "boot-test"
    core._qbi_dynamic_mtp_receipts = deque(maxlen=_QBI_MECHANISM_RECEIPT_LIMIT)
    core._qbi_reasoning_budget_receipts = deque(maxlen=_QBI_MECHANISM_RECEIPT_LIMIT)
    core._qbi_dynamic_mtp_sequence = 0
    core._qbi_reasoning_budget_sequence = 0
    core._qbi_dynamic_mtp_dropped_receipts = 0
    core._qbi_reasoning_budget_dropped_receipts = 0
    core._qbi_pending_dynamic_mtp_receipt = None
    core.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[(1, 1, 3), (2, 64, 0)],
        )
    )
    return core


def test_dynamic_mtp_receipt_joins_scheduler_model_runner_and_draft() -> None:
    core = _core()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r1": 1},
        total_num_scheduled_tokens=1,
        num_spec_tokens_to_schedule=3,
    )
    model_output = ModelRunnerOutput(
        req_ids=["r1"],
        req_id_to_index={"r1": 0},
        qbi_dynamic_mtp_receipt={
            "schema": "qbi.dynamic-qwen35-mtp-model-runner-state.v1",
            "speculative_method": "qwen3_5_mtp",
            "member_request_ids": ["r1"],
            "scheduler_selected_num_spec_tokens": 3,
            "model_runner_consumed_num_spec_tokens": 3,
            "proposal_path_executed": True,
            "proposal_work_skipped_proven": False,
            "max_draft_steps": 3,
        },
    )

    core._record_qbi_model_runner_telemetry(scheduler_output, model_output)
    core._finalize_qbi_dynamic_mtp_draft_receipt(
        SimpleNamespace(req_ids=["r1"], draft_token_ids=[[4, 5, 6]])
    )

    state = core.get_dynamic_mtp_state()
    assert state["engine_boot_id"] == "boot-test"
    assert state["latest_sequence"] == 1
    receipt = state["receipts"][0]
    assert receipt["scheduler_effective_k_proven"] is True
    assert receipt["model_runner_effective_k_proven"] is True
    assert receipt["proposed_token_count"] == 3
    assert receipt["draft_member_identity_match"] is True
    assert receipt["proposal_count_within_bound_proven"] is True
    assert receipt["draft_receipt_complete"] is True
    assert core.get_dynamic_mtp_state(after_sequence=1)["receipts"] == []


def test_dynamic_mtp_k_zero_proves_actual_skip_without_draft_rows() -> None:
    core = _core()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r1": 1, "r2": 1},
        total_num_scheduled_tokens=2,
        num_spec_tokens_to_schedule=0,
    )
    model_output = ModelRunnerOutput(
        req_ids=["r1", "r2"],
        req_id_to_index={"r1": 0, "r2": 1},
        qbi_dynamic_mtp_receipt={
            "schema": "qbi.dynamic-qwen35-mtp-model-runner-state.v1",
            "speculative_method": "qwen3_5_mtp",
            "member_request_ids": ["r1", "r2"],
            "scheduler_selected_num_spec_tokens": 0,
            "model_runner_consumed_num_spec_tokens": 0,
            "proposal_path_executed": False,
            "proposal_work_skipped_proven": True,
            "max_draft_steps": 0,
        },
    )

    core._record_qbi_model_runner_telemetry(scheduler_output, model_output)
    core._finalize_qbi_dynamic_mtp_draft_receipt(None)

    receipt = core.get_dynamic_mtp_state()["receipts"][0]
    assert receipt["proposal_path_executed"] is False
    assert receipt["proposal_work_skipped_proven"] is True
    assert receipt["proposed_token_count"] == 0
    assert receipt["draft_member_identity_match"] is True
    assert receipt["proposal_count_within_bound_proven"] is True


def test_dynamic_mtp_fails_effective_proof_when_proposer_did_not_run() -> None:
    core = _core()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r1": 1},
        total_num_scheduled_tokens=1,
        num_spec_tokens_to_schedule=3,
    )
    model_output = ModelRunnerOutput(
        req_ids=["r1"],
        req_id_to_index={"r1": 0},
        qbi_dynamic_mtp_receipt={
            "schema": "qbi.dynamic-qwen35-mtp-model-runner-state.v1",
            "speculative_method": "qwen3_5_mtp",
            "member_request_ids": ["r1"],
            "scheduler_selected_num_spec_tokens": 3,
            "model_runner_consumed_num_spec_tokens": 0,
            "proposal_path_executed": False,
            "proposal_work_skipped_proven": False,
            "max_draft_steps": 0,
        },
    )

    core._record_qbi_model_runner_telemetry(scheduler_output, model_output)
    core._finalize_qbi_dynamic_mtp_draft_receipt(None)

    receipt = core.get_dynamic_mtp_state()["receipts"][0]
    assert receipt["scheduler_effective_k_proven"] is True
    assert receipt["model_runner_effective_k_proven"] is False
    assert receipt["proposed_token_count"] == 0


def test_reasoning_receipt_joins_sampler_state_to_engine_boot() -> None:
    core = _core()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r1": 1},
        total_num_scheduled_tokens=1,
        num_spec_tokens_to_schedule=0,
    )
    model_output = ModelRunnerOutput(
        req_ids=["r1"],
        req_id_to_index={"r1": 0},
        qbi_reasoning_budget_receipts=[
            {
                "schema": "qbi.reasoning-token-budget-sampler-state.v1",
                "request_id": "r1",
                "requested_thinking_token_budget": 16,
                "effective_thinking_token_budget": 16,
                "initialized_start_token_ids": [10],
                "initialized_end_token_ids": [11],
                "prompt_reasoning_tokens": 0,
                "generated_reasoning_tokens": 8,
                "reasoning_end_kind": "forced_budget",
                "forced_end_token_applied": True,
                "sampler_state_tracked": True,
            }
        ],
    )

    core._record_qbi_model_runner_telemetry(scheduler_output, model_output)

    state = core.get_reasoning_budget_state()
    receipt = state["receipts"][0]
    assert receipt["engine_boot_id"] == "boot-test"
    assert receipt["transport_parameter_applied_proven"] is True
    assert receipt["runtime_budget_binding_proven"] is True
    assert receipt["runtime_budget_effective_proven"] is True
    assert receipt["runtime_budget_enforced_proven"] is True
    assert receipt["natural_end_before_budget"] is False
    assert core.get_reasoning_budget_state(after_sequence=1)["receipts"] == []


def test_reasoning_natural_end_is_bound_but_not_claimed_enforced() -> None:
    core = _core()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r1": 1},
        total_num_scheduled_tokens=1,
        num_spec_tokens_to_schedule=0,
    )
    model_output = ModelRunnerOutput(
        req_ids=["r1"],
        req_id_to_index={"r1": 0},
        qbi_reasoning_budget_receipts=[
            {
                "schema": "qbi.reasoning-token-budget-sampler-state.v1",
                "request_id": "r1",
                "requested_thinking_token_budget": 16,
                "effective_thinking_token_budget": 16,
                "initialized_start_token_ids": [10],
                "initialized_end_token_ids": [11],
                "prompt_reasoning_tokens": 0,
                "generated_reasoning_tokens": 4,
                "reasoning_end_kind": "natural_end",
                "forced_end_token_applied": False,
                "sampler_state_tracked": True,
            }
        ],
    )

    core._record_qbi_model_runner_telemetry(scheduler_output, model_output)

    receipt = core.get_reasoning_budget_state()["receipts"][0]
    assert receipt["runtime_budget_binding_proven"] is True
    assert receipt["runtime_budget_effective_proven"] is False
    assert receipt["runtime_budget_enforced_proven"] is False
    assert receipt["natural_end_before_budget"] is True


@pytest.mark.parametrize("after_sequence", [-1, True, 1.5, "1"])
def test_telemetry_cursor_fails_closed(after_sequence) -> None:
    core = _core()
    with pytest.raises(ValueError, match="non-negative integer"):
        core.get_dynamic_mtp_state(after_sequence)
