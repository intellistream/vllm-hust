# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.forward_context import ForwardContext, override_forward_context
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


def test_sparsify_fn_prefill_half_3d_matches_official_behavior():
    threshold = torch.tensor(1e6)
    sparsify = SparsifyFn(threshold)

    x = torch.ones(1, 4, 3)
    out = sparsify(x)
    assert torch.allclose(out[:, :2, :], x[:, :2, :])
    assert out[:, 2:, :].abs().max() == 0.0

    decode_x = torch.ones(2, 1, 3)
    decode_out = sparsify(decode_x)
    assert decode_out.abs().max() == 0.0


def test_sparsify_fn_prefill_half_2d_fallback_masks_last_half():
    threshold = torch.tensor(1e6)
    sparsify = SparsifyFn(threshold)

    x = torch.ones(4, 3)
    out = sparsify(x)
    assert torch.allclose(out[:2], x[:2])
    assert out[2:].abs().max() == 0.0

    decode_x = torch.ones(1, 3)
    decode_out = sparsify(decode_x)
    assert decode_out.abs().max() == 0.0


def test_sparsify_fn_prefill_half_2d_uses_vllm_query_slices():
    threshold = torch.tensor(1e6)
    sparsify = SparsifyFn(threshold)
    metadata = {
        "layer.0": SimpleNamespace(
            query_start_loc=torch.tensor([0, 4, 5]),
            num_actual_tokens=5,
        )
    }
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=metadata,
        slot_mapping={},
    )

    x = torch.ones(5, 3)
    with override_forward_context(context):
        out = sparsify(x)

    assert torch.allclose(out[:2], x[:2])
    assert out[2:4].abs().max() == 0.0
    assert out[4:].abs().max() == 0.0


def test_sparsify_fn_prefill_none_is_decode_only():
    threshold = torch.tensor(1e6)
    sparsify = SparsifyFn(
        threshold,
        apply_all_tokens=False,
        prefill_sparsify="none",
    )

    prefill_x = torch.ones(1, 4, 3)
    decode_x = torch.ones(1, 1, 3)
    assert torch.allclose(sparsify(prefill_x), prefill_x)
    assert sparsify(decode_x).abs().max() == 0.0
