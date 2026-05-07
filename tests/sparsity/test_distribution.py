# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.sparsity.distribution import Distribution, SparsifyFn


def test_distribution_icdf():
    # Simple histogram: 100 bins from -1 to 1, uniform distribution
    bin_edges = torch.linspace(-1, 1, 101)
    histogram = torch.ones(100)
    dist = Distribution(histogram, bin_edges)

    # median should be ~0
    median = dist.icdf(0.5)
    assert abs(median) < 0.02

    # 0.75 quantile should be ~0.5
    q75 = dist.icdf(0.75)
    assert 0.4 < q75 < 0.6


def test_distribution_icdf_teal_threshold():
    """For 40% sparsity, threshold = icdf(0.5 + 0.4/2) = icdf(0.7)."""
    bin_edges = torch.linspace(-1, 1, 101)
    histogram = torch.ones(100)
    dist = Distribution(histogram, bin_edges)

    threshold = dist.icdf(0.5 + 0.4 / 2)
    # For uniform on [-1, 1], icdf(0.7) = -1 + 2*0.7 = 0.4
    assert 0.35 < threshold < 0.45


def test_sparsify_fn_zero_sparsity():
    """With threshold=0, sparsify_fn should be identity."""
    threshold = torch.tensor(0.0)
    sparsify = SparsifyFn(threshold, apply_all_tokens=True)

    x = torch.randn(10, 20)
    out = sparsify(x)
    assert torch.allclose(out, x)


def test_sparsify_fn_high_threshold():
    """With a very high threshold, everything should be zeroed."""
    threshold = torch.tensor(1e6)
    sparsify = SparsifyFn(threshold, apply_all_tokens=True)

    x = torch.randn(10, 20)
    out = sparsify(x)
    assert out.abs().max() == 0.0


def test_sparsify_fn_partial_sparsity():
    """Verify that roughly the expected fraction is zeroed."""
    torch.manual_seed(42)
    x = torch.randn(1000, 100)

    # threshold = 0.5 -> keep |x| > 0.5
    # For standard normal, P(|Z| > 0.5) ≈ 0.617
    threshold = torch.tensor(0.5)
    sparsify = SparsifyFn(threshold, apply_all_tokens=True)

    out = sparsify(x)
    sparsity = (out == 0).float().mean().item()
    # Should be roughly 1 - 0.617 = 0.383
    assert 0.30 < sparsity < 0.45


def test_sparsify_fn_moved_threshold():
    """Threshold buffer should follow module.to(device)."""
    sparsify = SparsifyFn(torch.tensor(0.5))
    sparsify.to("cpu")
    assert sparsify.threshold.device.type == "cpu"
