# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.warmup.qwen_triton_warmup_config import (
    needs_qwen_triton_warmup,
)


def _model_config(*, text_model_type=None, model_type=None):
    return SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type=text_model_type),
        hf_config=SimpleNamespace(model_type=model_type),
    )


def test_qwen_triton_warmup_family_detection() -> None:
    assert needs_qwen_triton_warmup(_model_config(model_type="qwen3_5"))
    assert needs_qwen_triton_warmup(
        _model_config(text_model_type="qwen3_5_moe", model_type="other")
    )


def test_non_qwen_models_skip_triton_warmup_import_path() -> None:
    assert not needs_qwen_triton_warmup(_model_config(model_type="deepseek_v2"))
    assert not needs_qwen_triton_warmup(_model_config(model_type="glm4"))
    assert not needs_qwen_triton_warmup(SimpleNamespace())
