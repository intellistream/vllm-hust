# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight model-family checks for Qwen Triton kernel warmup."""

_QWEN_MODEL_TYPES = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def needs_qwen_triton_warmup(model_config: object) -> bool:
    """Return whether ``model_config`` uses Qwen's Triton warmup kernels.

    This helper intentionally has no Torch or Triton imports so worker modules
    can call it before a device context exists on non-CUDA backends.
    """
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    for config in (hf_text_config, hf_config):
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            return str(model_type) in _QWEN_MODEL_TYPES
    return False
