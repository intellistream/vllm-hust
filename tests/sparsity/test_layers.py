# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import pytest
import torch

from vllm.sparsity.config import ActivationSparsityConfig
from vllm.sparsity.distribution import LaRosaSparsifyFn
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
            enable=True,
            calibration_path=tmpdir,
            uniform_sparsity=0.7,
            apply_all_tokens=True,
        )
        sparsifier = build_sparsifier(cfg, layer_idx=0, proj_name="mlp.gate_up")

        assert sparsifier is not None
        assert sparsifier.apply_all_tokens is True
        assert sparsifier.prefill_sparsify == "half"
        assert sparsifier.expected_sparsity == 0.7
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
        os.makedirs(os.path.join(tmpdir, "histograms", "layer-2", "self_attn"))
        hidden = 16
        d = torch.eye(hidden)
        torch.save(
            d,
            os.path.join(tmpdir, "histograms", "layer-2", "self_attn", "D.pt"),
        )

        cfg = ActivationSparsityConfig(
            enable=True,
            method="larosa",
            calibration_path=tmpdir,
            uniform_sparsity=0.5,
        )
        sparsifier = build_sparsifier(cfg, layer_idx=2, proj_name="self_attn.qkv")

        assert sparsifier is not None
        assert isinstance(sparsifier, LaRosaSparsifyFn)
        assert sparsifier.rotate_input is True
        assert sparsifier.sparsity_level == 0.4
        assert sparsifier.expected_sparsity == 0.4
        assert sparsifier.apply_all_tokens is False
        assert sparsifier.prefill_sparsify == "half"
        x = torch.randn(4, hidden)
        out = sparsifier._apply_mask(x)
        assert out.shape == x.shape
        assert (out == 0).any()


def test_build_sparsifier_larosa_honors_all_token_prefill_policy():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "histograms", "layer-2", "self_attn"))
        hidden = 16
        torch.save(
            torch.eye(hidden),
            os.path.join(tmpdir, "histograms", "layer-2", "self_attn", "D.pt"),
        )

        cfg = ActivationSparsityConfig(
            enable=True,
            method="larosa",
            calibration_path=tmpdir,
            uniform_sparsity=0.5,
            apply_all_tokens=True,
            prefill_sparsify="all",
        )
        sparsifier = build_sparsifier(cfg, layer_idx=2, proj_name="self_attn.qkv")

        assert isinstance(sparsifier, LaRosaSparsifyFn)
        assert sparsifier.apply_all_tokens is True
        assert sparsifier.prefill_sparsify == "all"


def test_build_sparsifier_larosa_second_site_no_rotation_required():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ActivationSparsityConfig(
            enable=True,
            method="larosa",
            calibration_path=tmpdir,
            uniform_sparsity=0.5,
        )

        sparsifier = build_sparsifier(cfg, layer_idx=2, proj_name="mlp.down")

        assert isinstance(sparsifier, LaRosaSparsifyFn)
        assert sparsifier.rotate_input is False
        assert sparsifier.sparsity_level == 0.6
        assert sparsifier.expected_sparsity == 0.6
        assert sparsifier.apply_all_tokens is False
        assert sparsifier.prefill_sparsify == "half"
