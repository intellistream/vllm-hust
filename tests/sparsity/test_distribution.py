# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import torch

from vllm.sparsity.distribution import (
    Distribution,
    LaRosaSparsifyFn,
    SparsifyFn,
    larosa_topk,
)
from vllm.sparsity.rotation import merge_rotation_into_weight


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
    from vllm.forward_context import ForwardContext, override_forward_context

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
        cached = sparsify._get_vllm_request_slices(x)

    assert torch.allclose(out[:2], x[:2])
    assert out[2:4].abs().max() == 0.0
    assert out[4:].abs().max() == 0.0
    assert cached == [(0, 4), (4, 5)]
    assert len(context.additional_kwargs["teal_request_slices"]) == 1


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


def test_larosa_topk_matches_official_rule():
    x = torch.tensor([[1.0, -4.0, 3.0, 2.0], [0.1, -0.2, 0.3, -0.4]])

    out = larosa_topk(x, sparsity_level=0.5)

    expected = torch.tensor([[0.0, -4.0, 3.0, 0.0], [0.0, 0.0, 0.3, -0.4]])
    assert torch.allclose(out, expected)


def test_larosa_sparsify_rotates_topk_and_unrotates():
    rotation = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    sparsify = LaRosaSparsifyFn(
        sparsity_level=0.5,
        rotation=rotation,
        rotate_input=True,
    )
    x = torch.tensor([[1.0, -4.0, 3.0, 2.0]])

    out = sparsify(x)

    expected = larosa_topk(x.float() @ rotation.float(), 0.5) @ rotation.float().t()
    assert torch.allclose(out, expected.to(dtype=x.dtype))


def test_larosa_sparsify_uses_fp32_rotation_math():
    rotation = torch.eye(4, dtype=torch.float64)
    sparsify = LaRosaSparsifyFn(
        sparsity_level=0.5,
        rotation=rotation,
        rotate_input=True,
    )

    assert sparsify.rotation.dtype == torch.float32

    x = torch.tensor([[1.0, -4.0, 3.0, 2.0]], dtype=torch.bfloat16)
    out = sparsify(x)

    assert out.dtype == x.dtype


def test_larosa_sparse_linear_rotated_input_matches_unrotation_math():
    weight = torch.randn(6, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    x = torch.randn(3, 4)
    sparsify = LaRosaSparsifyFn(
        sparsity_level=0.5,
        rotation=q,
        rotate_input=True,
        use_sparse_gemv=True,
    )
    linear = SimpleNamespace(_larosa_sparse_weight_merged=True)
    merged_weight = merge_rotation_into_weight(weight, q)

    rotated_x = sparsify._sparse_linear_input(x, linear)
    threshold, inclusive = sparsify._sparse_linear_threshold(rotated_x)
    sparse_rotated_x = torch.where(
        torch.ge(rotated_x.abs(), threshold.reshape(x.shape[0], 1)),
        rotated_x,
        torch.zeros_like(rotated_x),
    )

    actual = sparse_rotated_x @ merged_weight.t()
    expected = (larosa_topk(x.float() @ q.float(), 0.5) @ q.float().t()) @ weight.t()

    assert inclusive is True
    assert torch.allclose(actual, expected, atol=1e-5)


def test_larosa_sparse_linear_rotated_input_matches_bf16_unrotation_math():
    torch.manual_seed(0)
    weight = torch.randn(6, 4, dtype=torch.bfloat16)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    sparsify = LaRosaSparsifyFn(
        sparsity_level=0.5,
        rotation=q,
        rotate_input=True,
        use_sparse_gemv=True,
    )
    linear = SimpleNamespace(_larosa_sparse_weight_merged=True)
    merged_weight = merge_rotation_into_weight(weight, q)

    rotated_x = sparsify._sparse_linear_input(x, linear)
    threshold, inclusive = sparsify._sparse_linear_threshold(rotated_x)
    sparse_rotated_x = torch.where(
        torch.ge(rotated_x.abs().to(dtype=torch.float32), threshold.reshape(3, 1)),
        rotated_x,
        torch.zeros_like(rotated_x),
    )

    actual = sparse_rotated_x @ merged_weight.t()
    dense_sparse_x = larosa_topk(x.float() @ q.float(), 0.5) @ q.float().t()
    expected = dense_sparse_x.to(dtype=x.dtype) @ weight.t()

    assert rotated_x.dtype == x.dtype
    assert threshold.dtype == torch.float32
    assert inclusive is True
    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_larosa_rotated_sparse_linear_requires_merged_weight():
    sparsify = LaRosaSparsifyFn(
        sparsity_level=0.5,
        rotation=torch.eye(4),
        rotate_input=True,
        use_sparse_gemv=True,
    )

    assert sparsify._sparse_linear_input(torch.randn(1, 4), SimpleNamespace()) is None


def test_larosa_try_apply_linear_rejects_shape_before_rotation(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_GEMV_LINEAR_POLICY", raising=False)
    monkeypatch.delenv("VLLM_SPARSE_GEMV_MIN_SPARSITY", raising=False)
    sparsify = LaRosaSparsifyFn(
        sparsity_level=0.7,
        rotation=torch.eye(5),
        rotate_input=True,
        use_sparse_gemv=True,
        expected_sparsity=0.7,
    )
    linear = SimpleNamespace(
        weight=torch.randn(4608, 4).contiguous(),
        quant_config=None,
        _larosa_sparse_weight_merged=True,
    )

    assert sparsify.try_apply_linear(torch.randn(1, 4), linear) is None


def test_sparse_linear_weight_t_cache_invalidates_after_weight_mutation():
    linear = torch.nn.Linear(4, 3, bias=False)
    sparsify = SparsifyFn(torch.tensor(0.5), use_sparse_gemv=True)

    first = sparsify._get_sparse_linear_weight_t(linear, linear.weight)
    cached = sparsify._get_sparse_linear_weight_t(linear, linear.weight)

    assert cached.data_ptr() == first.data_ptr()
    assert torch.allclose(first, linear.weight.t())

    with torch.no_grad():
        linear.weight.add_(1.0)

    updated = sparsify._get_sparse_linear_weight_t(linear, linear.weight)

    assert updated.data_ptr() != first.data_ptr()
    assert torch.allclose(updated, linear.weight.t())
    assert not torch.allclose(first, linear.weight.t())


def test_sparse_linear_auto_policy_keeps_single_batch_wide_output(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_GEMV_LINEAR_POLICY", raising=False)
    monkeypatch.delenv("VLLM_SPARSE_GEMV_MIN_SPARSITY", raising=False)
    sparsify = SparsifyFn(
        torch.tensor(0.5),
        use_sparse_gemv=True,
        expected_sparsity=0.7,
    )
    x = torch.randn(1, 3584)
    weight = torch.randn(32768, 3584)

    assert sparsify._should_use_sparse_linear_kernel(x, weight) is True


def test_sparse_linear_auto_policy_requires_expected_sparsity(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_GEMV_LINEAR_POLICY", raising=False)
    monkeypatch.delenv("VLLM_SPARSE_GEMV_MIN_SPARSITY", raising=False)
    x = torch.randn(1, 3584)
    weight = torch.randn(32768, 3584)

    unknown = SparsifyFn(torch.tensor(0.5), use_sparse_gemv=True)
    low = SparsifyFn(
        torch.tensor(0.5),
        use_sparse_gemv=True,
        expected_sparsity=0.4,
    )
    high = SparsifyFn(
        torch.tensor(0.5),
        use_sparse_gemv=True,
        expected_sparsity=0.7,
    )

    assert unknown._should_use_sparse_linear_kernel(x, weight) is False
    assert low._should_use_sparse_linear_kernel(x, weight) is False
    assert high._should_use_sparse_linear_kernel(x, weight) is True


def test_sparse_linear_auto_policy_honors_min_sparsity_env(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_GEMV_LINEAR_POLICY", raising=False)
    monkeypatch.setenv("VLLM_SPARSE_GEMV_MIN_SPARSITY", "0.75")
    x = torch.randn(1, 3584)
    weight = torch.randn(32768, 3584)

    below = SparsifyFn(
        torch.tensor(0.5),
        use_sparse_gemv=True,
        expected_sparsity=0.7,
    )
    at_cutoff = SparsifyFn(
        torch.tensor(0.5),
        use_sparse_gemv=True,
        expected_sparsity=0.75,
    )

    assert below._should_use_sparse_linear_kernel(x, weight) is False
    assert at_cutoff._should_use_sparse_linear_kernel(x, weight) is True


def test_sparse_linear_auto_policy_skips_losing_shapes(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_GEMV_LINEAR_POLICY", raising=False)
    sparsify = SparsifyFn(
        torch.tensor(0.5),
        use_sparse_gemv=True,
        expected_sparsity=0.7,
    )

    assert not sparsify._should_use_sparse_linear_kernel(
        torch.randn(2, 3584),
        torch.randn(32768, 3584),
    )
    assert not sparsify._should_use_sparse_linear_kernel(
        torch.randn(1, 3584),
        torch.randn(4608, 3584),
    )
    assert not sparsify._should_use_sparse_linear_kernel(
        torch.randn(1, 18944),
        torch.randn(3584, 18944),
    )


def test_sparse_linear_policy_overrides(monkeypatch):
    sparsify = SparsifyFn(torch.tensor(0.5), use_sparse_gemv=True)
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)

    monkeypatch.setenv("VLLM_SPARSE_GEMV_LINEAR_POLICY", "all")
    assert sparsify._should_use_sparse_linear_kernel(x, weight) is True

    monkeypatch.setenv("VLLM_SPARSE_GEMV_LINEAR_POLICY", "none")
    assert sparsify._should_use_sparse_linear_kernel(x, weight) is False


def test_sparse_linear_dense_fallback_masks_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY", raising=False)
    sparsify = SparsifyFn(torch.tensor(0.5), use_sparse_gemv=True)
    x = torch.tensor([[0.25, 1.0]])

    out = sparsify.apply_dense_fallback(x)

    assert torch.equal(out, torch.tensor([[0.0, 1.0]]))


def test_sparse_linear_identity_fallback_skips_all_row_mask(monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY", "identity")
    sparsify = SparsifyFn(torch.tensor(0.5), use_sparse_gemv=True)
    x = torch.tensor([[0.25, 1.0]])

    out = sparsify.apply_dense_fallback(x)

    assert out is x


def test_sparse_linear_single_row_identity_fallback_skips_request_slices(
    monkeypatch,
):
    from vllm.forward_context import ForwardContext, override_forward_context

    monkeypatch.setenv("VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY", "identity")
    sparsify = SparsifyFn(torch.tensor(0.5), use_sparse_gemv=True)
    metadata = {
        "layer.0": SimpleNamespace(
            query_start_loc=torch.tensor([0, 1]),
            num_actual_tokens=1,
        )
    }
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=metadata,
        slot_mapping={},
    )
    x = torch.tensor([[0.25, 1.0]])

    with override_forward_context(context):
        out = sparsify.apply_dense_fallback(x)

    assert out is x
    assert "teal_request_slices" not in context.additional_kwargs


def test_sparse_linear_identity_fallback_keeps_mixed_prefill_mask(monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY", "identity")
    sparsify = SparsifyFn(torch.tensor(0.5), use_sparse_gemv=True)
    x = torch.tensor([[0.25, 1.0], [0.25, 1.0]])

    out = sparsify.apply_dense_fallback(x)

    assert torch.equal(out, torch.tensor([[0.25, 1.0], [0.0, 1.0]]))


def test_sparse_gemv_marker_records_tensor_metadata(tmp_path, monkeypatch):
    from vllm.sparsity.kernels.sparse_gemv import (
        _record_sparse_gemv_invocation,
        reset_sparse_gemv_invocation_count,
    )

    marker_path = tmp_path / "sparse_gemv_marker.jsonl"
    monkeypatch.setenv("VLLM_SPARSE_GEMV_MARKER_PATH", str(marker_path))

    reset_sparse_gemv_invocation_count()
    _record_sparse_gemv_invocation(
        x=torch.ones(1, 4, dtype=torch.float16),
        threshold=torch.tensor(0.5, dtype=torch.float32),
        inclusive=False,
        weight_t=torch.ones(4, 3, dtype=torch.float16),
    )
    reset_sparse_gemv_invocation_count()
    _record_sparse_gemv_invocation(
        x=torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.bfloat16,
        ),
        threshold=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        inclusive=True,
    )

    records = [
        json.loads(line)
        for line in marker_path.read_text(encoding="utf-8").splitlines()
    ]
    reset_sparse_gemv_invocation_count()

    assert len(records) == 2
    assert records[0]["x_shape"] == [1, 4]
    assert records[0]["threshold_shape"] == []
    assert records[0]["threshold_numel"] == 1
    assert records[0]["inclusive"] is False
    assert records[0]["weight_t_provided"] is True
    assert records[0]["active_row_count"] == 1
    assert records[0]["active_hidden_size"] == 4
    assert records[0]["active_count_min"] == 4
    assert records[0]["active_count_max"] == 4
    assert records[0]["active_count_mean"] == 4.0
    assert records[0]["active_density_mean"] == 1.0
    assert records[0]["active_sparsity_mean"] == 0.0
    assert records[1]["x_shape"] == [3, 4]
    assert records[1]["threshold_shape"] == [3]
    assert records[1]["threshold_numel"] == 3
    assert records[1]["inclusive"] is True
    assert records[1]["weight_t_provided"] is False
    assert records[1]["active_row_count"] == 3
    assert records[1]["active_hidden_size"] == 4
    assert records[1]["active_count_min"] == 1
    assert records[1]["active_count_max"] == 3
    assert records[1]["active_count_mean"] == 2.0
    assert records[1]["active_density_min"] == 0.25
    assert records[1]["active_density_max"] == 0.75
    assert records[1]["active_density_mean"] == 0.5
    assert records[1]["active_sparsity_mean"] == 0.5


def test_larosa_sparsify_second_site_topk_direct():
    sparsify = LaRosaSparsifyFn(sparsity_level=0.5, rotate_input=False)
    x = torch.tensor([[1.0, -4.0, 3.0, 2.0]])

    out = sparsify(x)

    assert torch.allclose(out, larosa_topk(x, 0.5))
