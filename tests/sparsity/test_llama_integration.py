# SPDX-License-Identifier: Apache-2.0

"""Lightweight structural tests for Llama sparsity injection.

These tests do NOT require a model download or GPU; they only verify
that the sparsifier modules are created and attached correctly.
"""

import os
import tempfile

import torch

from vllm.sparsity.config import ActivationSparsityConfig


def test_llama_mlp_sparsity_injection():
    """LlamaMLP should create sparsify_gate_up and sparsify_down
    when sparsity_config is provided with calibration data."""
    # We can't import LlamaMLP directly here because it pulls in
    # heavy vLLM internals (quantization, parallelism, etc.).
    # Instead we verify the structural contract by inspecting the
    # build_sparsifier helper.
    from vllm.sparsity.layers import build_sparsifier

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(torch.tensor(0.3), os.path.join(tmpdir, "layers.2.mlp.gate_up.threshold.pt"))
        torch.save(torch.tensor(0.4), os.path.join(tmpdir, "layers.2.mlp.down.threshold.pt"))

        cfg = ActivationSparsityConfig(enable=True, calibration_path=tmpdir)
        gate_up = build_sparsifier(cfg, layer_idx=2, proj_name="mlp.gate_up")
        down = build_sparsifier(cfg, layer_idx=2, proj_name="mlp.down")

        assert gate_up is not None
        assert down is not None

        x = torch.randn(2, 8)
        y_gate = gate_up(x)
        y_down = down(x)
        assert y_gate.shape == x.shape
        assert y_down.shape == x.shape


def test_llama_attention_sparsity_injection():
    """LlamaAttention should create sparsify_qkv and sparsify_o."""
    from vllm.sparsity.layers import build_sparsifier

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(torch.tensor(0.3), os.path.join(tmpdir, "layers.0.self_attn.qkv.threshold.pt"))
        torch.save(torch.tensor(0.4), os.path.join(tmpdir, "layers.0.self_attn.o.threshold.pt"))

        cfg = ActivationSparsityConfig(enable=True, calibration_path=tmpdir)
        qkv = build_sparsifier(cfg, layer_idx=0, proj_name="self_attn.qkv")
        o_proj = build_sparsifier(cfg, layer_idx=0, proj_name="self_attn.o")

        assert qkv is not None
        assert o_proj is not None
