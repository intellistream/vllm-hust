# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from types import SimpleNamespace

import pytest
import torch
from fastapi.responses import JSONResponse

from vllm.entrypoints.llm import LLM
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.scoring.io_processor import ScoringIOProcessor
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.outputs import (
    ClassificationRequestOutput,
    EmbeddingRequestOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
    ScoringRequestOutput,
)
from vllm.tasks import SCORE_TYPE_MAP

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def _make_pooling_request_output(
    request_id: str,
    data: torch.Tensor,
) -> PoolingRequestOutput:
    return PoolingRequestOutput(
        request_id=request_id,
        outputs=PoolingOutput(data=data),
        prompt_token_ids=[1, 2, 3],
        num_cached_tokens=1,
        finished=True,
    )


def _assert_close_list(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        assert math.isclose(actual_item, expected_item, rel_tol=1e-6, abs_tol=1e-6)


class _FakeEncodeLLM:
    def encode(self, prompts, **kwargs):
        pooling_task = kwargs["pooling_task"]
        if pooling_task == "embed":
            return [
                _make_pooling_request_output(
                    "embed-0",
                    torch.tensor([1.0, 2.0], dtype=torch.float32),
                )
            ]

        if pooling_task == "classify":
            return [
                _make_pooling_request_output(
                    "classify-0",
                    torch.tensor([0.2, 0.8], dtype=torch.float32),
                )
            ]

        raise AssertionError(pooling_task)


class _FakeScoringIOProcessor(ScoringIOProcessor):
    name = "fake-cross"
    pooling_task = "classify"

    def __init__(self):
        pass

    def valid_inputs(self, data_1, data_2):
        return SimpleNamespace(data_1=[data_1], data_2=[data_2])

    def pre_process_offline(self, ctx):
        return [{"prompt_token_ids": [1, 2, 3]}]

    def post_process_offline(self, ctx):
        return ctx.outputs


class _FakeScoreLLM:
    runner_type = "pooling"
    pooling_task = "classify"
    model_config = SimpleNamespace(hf_config=SimpleNamespace(num_labels=1))
    pooling_io_processors = {
        SCORE_TYPE_MAP["classify"]: _FakeScoringIOProcessor(),
    }

    def _lora_request_to_seq(self, lora_request, n_inputs):
        return [None] * n_inputs

    def _params_to_seq(self, pooling_params, n_inputs):
        return [pooling_params] * n_inputs

    def _priority_to_seq(self, priority, n_inputs):
        return [None] * n_inputs

    def _render_and_add_requests(self, **kwargs):
        self.render_call = kwargs

    def _run_engine(self, **kwargs):
        return [
            _make_pooling_request_output(
                "score-0",
                torch.tensor([0.75], dtype=torch.float32),
            )
        ]


def test_llm_embed_materializes_output():
    outputs = LLM.embed(_FakeEncodeLLM(), ["x"], use_tqdm=False)

    assert len(outputs) == 1
    assert isinstance(outputs[0], EmbeddingRequestOutput)
    _assert_close_list(outputs[0].outputs.embedding, [1.0, 2.0])


def test_llm_classify_materializes_output():
    outputs = LLM.classify(_FakeEncodeLLM(), ["x"], use_tqdm=False)

    assert len(outputs) == 1
    assert isinstance(outputs[0], ClassificationRequestOutput)
    _assert_close_list(outputs[0].outputs.probs, [0.2, 0.8])


def test_llm_score_materializes_output():
    outputs = LLM.score(_FakeScoreLLM(), "a", "b", use_tqdm=False)

    assert len(outputs) == 1
    assert isinstance(outputs[0], ScoringRequestOutput)
    assert math.isclose(outputs[0].outputs.score, 0.75, rel_tol=1e-6, abs_tol=1e-6)


def test_serving_embedding_openai_json_response_uses_batch_materialization():
    response = ServingEmbedding._openai_json_response(
        SimpleNamespace(json_response_cls=JSONResponse),
        [
            _make_pooling_request_output(
                "embed-0",
                torch.tensor([1.0, 2.0], dtype=torch.float32),
            )
        ],
        "req",
        1,
        "model",
        "float",
        None,
        None,
    )

    payload = json.loads(response.body)

    _assert_close_list(payload["data"][0]["embedding"], [1.0, 2.0])


def test_serving_classification_build_response_uses_batch_materialization():
    context = SimpleNamespace(
        final_res_batch=[
            _make_pooling_request_output(
                "classify-0",
                torch.tensor([0.2, 0.8], dtype=torch.float32),
            )
        ],
        request_id="req",
        created_time=1,
        model_name="model",
    )

    response = ServingClassification._build_response(
        SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(id2label={0: "neg", 1: "pos"})
            )
        ),
        context,
    )

    payload = json.loads(response.body)

    assert payload["data"][0]["label"] == "pos"
    _assert_close_list(payload["data"][0]["probs"], [0.2, 0.8])


def test_serving_scores_response_builders_use_batch_materialization():
    score_response = ServingScores._request_output_to_score_response(
        SimpleNamespace(),
        [
            _make_pooling_request_output(
                "score-0",
                torch.tensor([0.75], dtype=torch.float32),
            )
        ],
        "req",
        1,
        "model",
    )
    score_payload = json.loads(score_response.body)
    assert math.isclose(
        score_payload["data"][0]["score"],
        0.75,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

    rerank_response = ServingScores._request_output_to_rerank_response(
        SimpleNamespace(),
        [
            _make_pooling_request_output(
                "score-low",
                torch.tensor([0.25], dtype=torch.float32),
            ),
            _make_pooling_request_output(
                "score-high",
                torch.tensor([0.9], dtype=torch.float32),
            ),
        ],
        "req",
        "model",
        ["low", "high"],
        1,
    )
    rerank_payload = json.loads(rerank_response.body)

    assert len(rerank_payload["results"]) == 1
    assert rerank_payload["results"][0]["document"]["text"] == "high"
    assert math.isclose(
        rerank_payload["results"][0]["relevance_score"],
        0.9,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
