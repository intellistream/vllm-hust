# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile

import pytest
import torch

from vllm.sparsity.utils import (
    compute_thresholds_from_histograms,
    load_threshold,
)


def test_load_threshold_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "layers.5.self_attn.qkv.threshold.pt")
        torch.save(torch.tensor(0.42), path)

        t = load_threshold(tmpdir, layer_idx=5, proj_name="self_attn.qkv")
        assert isinstance(t, torch.Tensor)
        assert t.item() == pytest.approx(0.42)


def test_load_threshold_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_threshold(tmpdir, layer_idx=0, proj_name="mlp.down")


def test_compute_thresholds_from_histograms():
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_edges = torch.linspace(-1, 1, 101)
        histograms = {
            "layers.0.mlp.gate_up": torch.ones(100),
            "layers.0.mlp.down": torch.ones(100),
        }
        path = os.path.join(tmpdir, "histograms.pt")
        torch.save(
            {"histograms": histograms, "bin_edges": bin_edges}, path
        )

        thresholds = compute_thresholds_from_histograms(path, sparsity=0.4)
        assert "layers.0.mlp.gate_up" in thresholds
        assert "layers.0.mlp.down" in thresholds
        # For uniform [-1, 1], icdf(0.7) ≈ 0.4
        assert 0.35 < thresholds["layers.0.mlp.gate_up"].item() < 0.45
