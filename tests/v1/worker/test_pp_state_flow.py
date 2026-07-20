# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from types import SimpleNamespace

import torch

from vllm.v1.worker import pp_state_flow


def test_emit_is_default_off(monkeypatch, tmp_path):
    output = tmp_path / "trace.jsonl"
    monkeypatch.delenv("VLLM_DIAGNOSE_PP_STATE_FLOW", raising=False)
    monkeypatch.setenv("VLLM_DIAGNOSE_PP_STATE_FLOW_OUTPUT_PATH", str(output))

    pp_state_flow.emit("should_not_exist", value=1)

    assert list(tmp_path.iterdir()) == []


def test_emit_writes_process_local_json(monkeypatch, tmp_path):
    output = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VLLM_DIAGNOSE_PP_STATE_FLOW", "1")
    monkeypatch.setenv("VLLM_DIAGNOSE_PP_STATE_FLOW_OUTPUT_PATH", str(output))

    pp_state_flow.emit("test_event", value=7)

    files = list(tmp_path.glob("trace.pid-*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["schema"] == "vllm.pp_state_flow.v1"
    assert record["event"] == "test_event"
    assert record["value"] == 7


def test_tensor_summary_is_stable_and_bounded():
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    first = pp_state_flow.tensor_summary(tensor)
    second = pp_state_flow.tensor_summary(tensor.clone())

    assert first == second
    assert first["shape"] == [2, 2]
    assert first["values"] == [1, 2, 3, 4]
    assert (
        first["sha256"]
        == hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    )


def test_tensor_dict_summary_sorts_names():
    summary = pp_state_flow.tensor_dict_summary(
        {"z": torch.tensor([2]), "a": torch.tensor([1])}
    )
    assert list(summary) == ["a", "z"]


def test_install_wraps_effective_runner_overrides(monkeypatch):
    events = []
    monkeypatch.setattr(pp_state_flow, "enabled", lambda: True)
    monkeypatch.setattr(
        pp_state_flow, "rank_info", lambda: {"pp_rank": 1, "tp_rank": 0}
    )
    monkeypatch.setattr(
        pp_state_flow,
        "emit",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    class FakeRunner:
        def __init__(self):
            self.input_batch = SimpleNamespace(
                req_ids=["r"],
                num_reqs=1,
                block_table=SimpleNamespace(block_tables=[]),
            )
            self.execute_model_state = None

        def _model_forward(
            self,
            num_tokens_padded,
            input_ids=None,
            positions=None,
            intermediate_tensors=None,
        ):
            return SimpleNamespace(tensors={"hidden": torch.tensor([3.0])})

        def execute_model(self, scheduler_output):
            self._model_forward(1, torch.tensor([11]), torch.tensor([7]), None)
            self.execute_model_state = SimpleNamespace(
                logits=torch.tensor([[1.0, 4.0, 3.0, 2.0, 0.0]])
            )

        def sample_tokens(self):
            self.execute_model_state = None
            return SimpleNamespace(sampled_token_ids=[[1]])

    runner = FakeRunner()
    pp_state_flow.install_model_runner_wrappers(runner)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        microbatch_id=0,
        num_scheduled_tokens={"r": 1},
    )

    runner.execute_model(scheduler_output)
    runner.sample_tokens()

    assert [event["event"] for event in events] == [
        "attention_input_before_forward",
        "pp_send_start",
        "logits_before_sampling",
        "sampled_tokens_synchronized",
    ]
    assert events[0]["input_ids"]["values"] == [11]
    assert events[2]["top5_token_ids"] == [[1, 2, 3, 0, 4]]
    assert events[3]["sampled_token_ids"] == [[1]]
