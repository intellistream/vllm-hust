# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import pytest
import torch

from vllm.sparsity.config import ActivationSparsityConfig
from vllm.sparsity.layers import build_sparsifier


def test_build_sparsifier_disabled():
    cfg = ActivationSparsityConfig(enable=False)
    sparsifier = build_sparsifier(cfg, layer_idx=0, proj_name="mlp.gate_up")
    assert sparsifier is None


def test_build_sparsifier_no_calibration_path():
    cfg = ActivationSparsityConfig(enable=True, calibration_path=None)
    sparsifier = build_sparsifier(cfg, layer_idx=0, proj_name="mlp.gate_up")
    assert sparsifier is None


def test_build_sparsifier_missing_threshold():
    cfg = ActivationSparsityConfig(enable=True, calibration_path="/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        build_sparsifier(cfg, layer_idx=0, proj_name="mlp.gate_up")


def test_build_sparsifier_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a dummy threshold file
        threshold_path = os.path.join(tmpdir, "layers.0.mlp.gate_up.threshold.pt")
        torch.save(torch.tensor(0.5), threshold_path)

        cfg = ActivationSparsityConfig(
            enable=True, calibration_path=tmpdir, apply_all_tokens=True
        )
        sparsifier = build_sparsifier(cfg, layer_idx=0, proj_name="mlp.gate_up")

        assert sparsifier is not None
        assert sparsifier.apply_all_tokens is True
        assert sparsifier.prefill_sparsify == "half"
        x = torch.randn(4, 8)
        out = sparsifier(x)
        # With threshold 0.5, some values should be zeroed
        assert (out == 0).any()


def test_build_sparsifier_prefill_half_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        threshold_path = os.path.join(tmpdir, "layers.0.mlp.gate_up.threshold.pt")
        torch.save(torch.tensor(0.5), threshold_path)

        cfg = ActivationSparsityConfig(enable=True, calibration_path=tmpdir)
        sparsifier = build_sparsifier(cfg, layer_idx=0, proj_name="mlp.gate_up")

        assert sparsifier is not None
        assert sparsifier.apply_all_tokens is False
        assert sparsifier.prefill_sparsify == "half"


def test_build_sparsifier_larosa():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Threshold
        torch.save(
            torch.tensor(0.5),
            os.path.join(tmpdir, "layers.2.self_attn.qkv.threshold.pt"),
        )
        # Rotation matrices are load-time weight artifacts, not SparsifyFn modules.
        os.makedirs(os.path.join(tmpdir, "layers.2.self_attn.qkv"))
        hidden = 16
        d = torch.eye(hidden)
        torch.save(d, os.path.join(tmpdir, "layers.2.self_attn.qkv", "D.pt"))

        cfg = ActivationSparsityConfig(
            enable=True,
            method="larosa",
            calibration_path=tmpdir,
            apply_all_tokens=True,
        )
        sparsifier = build_sparsifier(cfg, layer_idx=2, proj_name="self_attn.qkv")

        assert sparsifier is not None
        x = torch.randn(4, hidden)
        out = sparsifier(x)
        assert out.shape == x.shape
        # SparsifyFn itself remains a plain mask; rotations are merged into weights.
        assert (out == 0).any()
