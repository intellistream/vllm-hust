# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id


def _parallel_drafting_proposer(hf_config: object) -> SpecDecodeBaseProposer:
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.draft_model_config = SimpleNamespace(hf_config=hf_config)
    proposer.pass_hidden_states_to_model = False
    return proposer


def test_dspark_drafter_uses_target_vocabulary_without_remapping():
    assert DSparkDeepseekV4ForCausalLM.draft_id_to_target_id is None


@pytest.mark.parametrize(
    ("hf_config", "expected_token_id"),
    [
        (
            SimpleNamespace(
                dflash_config={"mask_token_id": 11},
                mask_token_id=12,
                dspark_noise_token_id=13,
                pard_token=14,
                ptd_token_id=15,
            ),
            11,
        ),
        (
            SimpleNamespace(
                mask_token_id=12,
                dspark_noise_token_id=13,
                pard_token=14,
                ptd_token_id=15,
            ),
            12,
        ),
        (
            SimpleNamespace(
                dspark_noise_token_id=13,
                pard_token=14,
                ptd_token_id=15,
            ),
            13,
        ),
        (
            SimpleNamespace(
                dflash_config={"mask_token_id": None},
                mask_token_id=None,
                dspark_noise_token_id=None,
                pard_token=14,
                ptd_token_id=15,
            ),
            14,
        ),
        (SimpleNamespace(pard_token=14, ptd_token_id=15), 14),
        (SimpleNamespace(ptd_token_id=15), 15),
    ],
)
def test_parallel_drafting_token_id_precedence(
    hf_config: SimpleNamespace, expected_token_id: int
):
    proposer = _parallel_drafting_proposer(hf_config)

    proposer._init_parallel_drafting_params()

    assert proposer.parallel_drafting_token_id == expected_token_id
    assert get_parallel_drafting_token_id(hf_config) == expected_token_id


def test_parallel_drafting_token_id_requires_supported_configured_id():
    hf_config = SimpleNamespace(
        dflash_config={"mask_token_id": None},
        mask_token_id=None,
        dspark_noise_token_id=None,
        pard_token=None,
        ptd_token_id=None,
    )
    proposer = _parallel_drafting_proposer(hf_config)

    with pytest.raises(ValueError, match="dflash_config.mask_token_id.*ptd_token_id"):
        proposer._init_parallel_drafting_params()
    with pytest.raises(ValueError, match="dflash_config.mask_token_id.*ptd_token_id"):
        get_parallel_drafting_token_id(hf_config)


def test_dspark_config_inherits_target_quantization(monkeypatch: pytest.MonkeyPatch):
    target_model_config = SimpleNamespace(
        model="deepseek-ai/dspark_qwen3_4b_block7",
        tokenizer="target-tokenizer",
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="auto",
        seed=0,
        tokenizer_revision=None,
        max_model_len=128,
        quantization="fp8",
        enforce_eager=True,
        max_logprobs=20,
        config_format="hf",
    )
    draft_hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
    )
    draft_model_config = SimpleNamespace(
        model=target_model_config.model,
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
        quantization=None,
        verify_with_parallel_config=lambda _parallel_config: None,
    )

    def fake_model_config(**kwargs):
        draft_model_config.quantization = kwargs["quantization"]
        return draft_model_config

    monkeypatch.setattr("vllm.config.speculative.ModelConfig", fake_model_config)
    monkeypatch.setattr(SpeculativeConfig, "update_arch_", lambda _self: None)

    speculative_config = SpeculativeConfig(
        model=target_model_config.model,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        num_speculative_tokens=1,
    )

    assert speculative_config.method == "dspark"
    assert speculative_config.draft_model_config.quantization == "fp8"
