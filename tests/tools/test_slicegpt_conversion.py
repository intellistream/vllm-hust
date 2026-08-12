# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused contract tests for SliceGPT checkpoint conversion."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import load_file

from vllm import ModelRegistry

_TOOLS_DIR = Path(__file__).parents[2] / "tools" / "slicegpt"
sys.path.insert(0, str(_TOOLS_DIR))

from conversion_utils import (  # noqa: E402
    expected_state_names,
    infer_attention_head_dim,
    normalize_slicing_conf,
    validate_shape,
    validate_state_keys,
)

from vllm.model_executor.models.slicegpt_llama import (  # noqa: E402
    SliceGPTLlamaForCausalLM,
)
from vllm.model_executor.models.slicegpt_qwen2 import (  # noqa: E402
    SliceGPTQwen2ForCausalLM,
)


def _slicing_config() -> dict[str, object]:
    return {
        "embedding_dimensions": {"0": 2},
        "head_dimension": 2,
        "attention_input_dimensions": {"0": 2},
        "attention_output_dimensions": {"0": 2},
        "mlp_input_dimensions": {"0": 2},
        "mlp_output_dimensions": {"0": 2},
    }


def _state(*, attention_bias: bool) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.ones(3, 2),
        "lm_head.weight": torch.ones(3, 2),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 2),
        "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 2),
        "model.layers.0.self_attn.v_proj.weight": torch.ones(2, 2),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(2, 4),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 2),
        "model.layers.0.mlp.up_proj.weight": torch.ones(4, 2),
        "model.layers.0.mlp.down_proj.weight": torch.ones(2, 4),
        "model.layers.0.attn_shortcut_Q": torch.eye(2),
        "model.layers.0.mlp_shortcut_Q": torch.eye(2),
    }
    if attention_bias:
        state.update(
            {
                "model.layers.0.self_attn.q_proj.bias": torch.ones(4),
                "model.layers.0.self_attn.k_proj.bias": torch.ones(2),
                "model.layers.0.self_attn.v_proj.bias": torch.ones(2),
            }
        )
    return state


@pytest.mark.parametrize(
    "converter,attention_bias,architecture",
    [
        ("convert_slicegpt_to_vllm.py", False, "SliceGPTLlamaForCausalLM"),
        ("convert_slicegpt_to_vllm_qwen2.py", True, "SliceGPTQwen2ForCausalLM"),
    ],
)
def test_converter_writes_only_validated_tensors(
    tmp_path: Path,
    converter: str,
    attention_bias: bool,
    architecture: str,
) -> None:
    sliced_dir = tmp_path / "sliced"
    base_dir = tmp_path / "base"
    out_dir = tmp_path / "out"
    sliced_dir.mkdir()
    base_dir.mkdir()
    checkpoint = sliced_dir / "model.pt"
    torch.save(_state(attention_bias=attention_bias), checkpoint)
    checkpoint.with_suffix(".json").write_text(json.dumps(_slicing_config()))
    base_config = {
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 4,
        "vocab_size": 3,
    }
    (base_dir / "config.json").write_text(json.dumps(base_config))

    result = subprocess.run(
        [
            sys.executable,
            str(_TOOLS_DIR / converter),
            "--sliced-dir",
            str(sliced_dir),
            "--pt-name",
            checkpoint.name,
            "--base-model-path",
            str(base_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output_config = json.loads((out_dir / "config.json").read_text())
    assert output_config["architectures"] == [architecture]
    assert output_config["slicing_config"]["attn_head_dim"] == 2
    assert set(load_file(out_dir / "model.safetensors")) == expected_state_names(
        1, attention_bias=attention_bias
    )


def test_converter_rejects_missing_and_unexpected_state_tensors() -> None:
    state = _state(attention_bias=False)
    state.pop("lm_head.weight")
    state["not.a.slicegpt.tensor"] = torch.ones(1)

    with pytest.raises(ValueError, match="missing=.*lm_head.weight"):
        validate_state_keys(state, expected_state_names(1, attention_bias=False))


def test_slicing_config_requires_every_layer_and_valid_head_shape() -> None:
    raw = _slicing_config()
    raw["mlp_output_dimensions"] = {}

    with pytest.raises(ValueError, match="missing layer indices"):
        normalize_slicing_conf(raw, num_layers=1)
    with pytest.raises(ValueError, match="divisible"):
        infer_attention_head_dim(torch.ones(3, 2), num_attention_heads=2)
    with pytest.raises(ValueError, match="shape"):
        validate_shape("tensor", torch.ones(2, 3), (3, 2))


@pytest.mark.parametrize(
    "model_cls",
    [SliceGPTLlamaForCausalLM, SliceGPTQwen2ForCausalLM],
)
def test_slicegpt_models_fail_fast_for_pipeline_parallelism(model_cls) -> None:
    vllm_config = MagicMock()
    vllm_config.parallel_config.pipeline_parallel_size = 2

    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        model_cls(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("architecture", "expected_cls"),
    [
        ("SliceGPTLlamaForCausalLM", SliceGPTLlamaForCausalLM),
        ("SliceGPTQwen2ForCausalLM", SliceGPTQwen2ForCausalLM),
    ],
)
def test_slicegpt_architectures_resolve_from_model_registry(
    architecture: str,
    expected_cls: type,
) -> None:
    model_config = MagicMock(
        model_impl="vllm",
        convert_type="none",
        runner_type="generate",
    )

    model_cls, resolved_architecture = ModelRegistry.resolve_model_cls(
        architecture,
        model_config,
    )

    assert resolved_architecture == architecture
    assert model_cls is expected_cls
