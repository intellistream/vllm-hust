# SPDX-License-Identifier: Apache-2.0
"""TEAL and La RoSA distribution/sparsify functions."""

import torch
from torch import nn


class Distribution:
    """Histogram-based distribution for threshold calibration.

    Re-implements the core logic of TEAL's ``Distribution`` class
    without depending on the original repository.
    """

    def __init__(self, histogram: torch.Tensor, bin_edges: torch.Tensor) -> None:
        """Args:
        histogram: 1-D tensor of counts per bin.
        bin_edges: 1-D tensor of bin boundaries (length = histogram + 1).
        """
        if histogram.dim() != 1:
            raise ValueError(f"histogram must be 1-D, got {histogram.dim()}-D")
        if bin_edges.dim() != 1:
            raise ValueError(f"bin_edges must be 1-D, got {bin_edges.dim()}-D")
        if bin_edges.numel() != histogram.numel() + 1:
            raise ValueError(
                f"bin_edges length ({bin_edges.numel()}) must be "
                f"histogram length ({histogram.numel()}) + 1"
            )

        self.histogram = histogram.float()
        self.bin_edges = bin_edges.float()
        self._pdf = self.histogram / self.histogram.sum()
        self._cdf = self._pdf.cumsum(dim=0)

    def icdf(self, q: float) -> float:
        """Inverse CDF (quantile function) evaluated at ``q`` in [0, 1].

        For a desired sparsity ``s``, the TEAL threshold is computed as
        ``Distribution.icdf(0.5 + s / 2)`` because magnitudes are symmetric
        around zero.
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")

        # Find the first bin where CDF >= q
        idx = torch.searchsorted(self._cdf, torch.tensor(q), side="right").item()
        idx = min(idx, len(self.bin_edges) - 2)

        # Linear interpolation within the bin
        cdf_low = 0.0 if idx == 0 else self._cdf[idx - 1].item()
        cdf_high = self._cdf[idx].item()
        bin_width = (self.bin_edges[idx + 1] - self.bin_edges[idx]).item()

        if cdf_high - cdf_low < 1e-12:
            return self.bin_edges[idx].item()

        t = (q - cdf_low) / (cdf_high - cdf_low)
        return (self.bin_edges[idx] + t * bin_width).item()

    @classmethod
    def from_histograms(
        cls, histograms: dict[str, torch.Tensor], bin_edges: torch.Tensor
    ) -> dict[str, "Distribution"]:
        """Build a dict of Distributions from a histogram dict."""
        return {name: cls(hist, bin_edges) for name, hist in histograms.items()}


class SparsifyFn(nn.Module):
    """PyTorch module that applies a magnitude-based sparsity mask.

    For Phase 0 this uses a dense ``torch.where`` so that correctness
    can be validated without requiring a sparse GEMV kernel.
    """

    def __init__(
        self,
        threshold: torch.Tensor,
        decode_only: bool = False,
        apply_all_tokens: bool = False,
        prefill_sparsify: str = "half",
        use_sparse_gemv: bool = False,
    ) -> None:
        super().__init__()
        if prefill_sparsify not in {"half", "all", "none"}:
            raise ValueError(
                "prefill_sparsify must be one of 'half', 'all', or 'none', "
                f"got {prefill_sparsify!r}."
            )
        # Register as buffer so threshold moves with module.to(device/dtype)
        self.register_buffer("threshold", threshold)
        self.decode_only = decode_only
        self.apply_all_tokens = apply_all_tokens
        self.prefill_sparsify = prefill_sparsify
        self.use_sparse_gemv = use_sparse_gemv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_all_tokens:
            return self._apply_mask(x)

        if self.decode_only:
            return self._apply_decode_only(x)

        if self.prefill_sparsify == "all":
            return self._apply_mask(x)

        if self.prefill_sparsify == "none":
            return self._apply_decode_only(x)

        if self.prefill_sparsify == "half":
            return self._apply_prefill_half(x)

        raise ValueError(
            "prefill_sparsify must be one of 'half', 'all', or 'none', "
            f"got {self.prefill_sparsify!r}."
        )

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x.abs() > self.threshold.to(dtype=x.dtype, device=x.device),
            x,
            torch.zeros_like(x),
        )

    def try_apply_linear(
        self,
        x: torch.Tensor,
        linear_layer: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        if not self.use_sparse_gemv or x.dim() != 2:
            return None
        if not self._sparse_linear_applies_to_all_rows(x):
            return None

        weight = getattr(linear_layer, "weight", None)
        if weight is None or not weight.is_contiguous():
            return None
        if getattr(linear_layer, "quant_config", None) is not None:
            return None

        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            if get_tensor_model_parallel_world_size() != 1:
                return None
        except Exception:
            return None

        from vllm.sparsity.kernels.sparse_gemv import (
            can_use_sparse_gemv,
            sparse_gemv_impl,
        )

        if not can_use_sparse_gemv(
            tp_size=1,
            quant_config=getattr(linear_layer, "quant_config", None),
            dtype=x.dtype,
            use_sparse_gemv_flag=True,
        ):
            return None

        sparse_input = self._sparse_linear_input(x, linear_layer)
        if sparse_input is None:
            return None

        threshold, inclusive = self._sparse_linear_threshold(sparse_input)
        if threshold is None:
            return None

        weight_t = self._get_sparse_linear_weight_t(linear_layer, weight)
        output = sparse_gemv_impl(
            sparse_input,
            weight,
            threshold,
            inclusive=inclusive,
            weight_t=weight_t,
        )
        bias = getattr(linear_layer, "bias", None)
        skip_bias_add = getattr(linear_layer, "skip_bias_add", False)
        output_bias = bias if skip_bias_add else None
        if bias is not None and not skip_bias_add:
            output = output + bias
        return output, output_bias

    def _get_sparse_linear_weight_t(
        self,
        linear_layer: nn.Module,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        cache_key = (
            weight.data_ptr(),
            tuple(weight.shape),
            weight.dtype,
            weight.device,
            getattr(weight, "_version", None),
        )
        cached_key = getattr(linear_layer, "_activation_sparse_weight_t_key", None)
        cached = getattr(linear_layer, "_activation_sparse_weight_t", None)
        if cached is not None and cached_key == cache_key:
            return cached
        weight_t = weight.t().contiguous()
        linear_layer._activation_sparse_weight_t = weight_t
        linear_layer._activation_sparse_weight_t_key = cache_key
        return weight_t

    def _sparse_linear_applies_to_all_rows(self, x: torch.Tensor) -> bool:
        if self.apply_all_tokens or self.prefill_sparsify == "all":
            return True
        if self.decode_only or self.prefill_sparsify == "none":
            row_mask = self._get_vllm_decode_row_mask(x)
        else:
            row_mask = self._get_vllm_prefill_half_row_mask(x)
        if row_mask is None:
            return x.shape[0] == 1
        return bool(row_mask.all().item())

    def _sparse_linear_input(
        self,
        x: torch.Tensor,
        linear_layer: nn.Module,
    ) -> torch.Tensor | None:
        del linear_layer
        return x

    def _sparse_linear_threshold(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor | None, bool]:
        del x
        return self.threshold.to(dtype=torch.float32), False

    def _apply_rows(self, x: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        if row_mask.numel() != x.shape[0]:
            raise ValueError(
                f"row_mask length ({row_mask.numel()}) must equal "
                f"input rows ({x.shape[0]})."
            )
        masked = self._apply_mask(x)
        view_shape = (row_mask.numel(),) + (1,) * (x.dim() - 1)
        row_mask = row_mask.to(device=x.device).view(view_shape)
        return torch.where(row_mask, masked, x)

    def _apply_prefill_half(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            seq_len = x.shape[1]
            if seq_len == 1:
                return self._apply_mask(x)
            half_seq_len = seq_len // 2
            if half_seq_len == 0:
                return x
            return torch.cat(
                (
                    x[:, :-half_seq_len, :],
                    self._apply_mask(x[:, -half_seq_len:, :]),
                ),
                dim=1,
            )

        if x.dim() != 2:
            return self._apply_mask(x)

        row_mask = self._get_vllm_prefill_half_row_mask(x)
        if row_mask is None:
            if x.shape[0] == 1:
                return self._apply_mask(x)
            half_seq_len = x.shape[0] // 2
            if half_seq_len == 0:
                return x
            row_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            row_mask[-half_seq_len:] = True
        return self._apply_rows(x, row_mask)

    def _apply_decode_only(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return self._apply_mask(x) if x.shape[1] == 1 else x

        if x.dim() != 2:
            return self._apply_mask(x)

        row_mask = self._get_vllm_decode_row_mask(x)
        if row_mask is None:
            return self._apply_mask(x) if x.shape[0] == 1 else x
        return self._apply_rows(x, row_mask)

    def _get_vllm_decode_row_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        request_slices = self._get_vllm_request_slices(x)
        if request_slices is None:
            return None

        cache_key = (
            "decode",
            x.device.type,
            x.device.index,
            x.shape[0],
            tuple(request_slices),
        )
        cached = self._get_cached_forward_mask(cache_key)
        if cached is not None:
            return cached

        row_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for start, end in request_slices:
            if end - start == 1:
                row_mask[start:end] = True
        self._set_cached_forward_mask(cache_key, row_mask)
        return row_mask

    def _get_vllm_prefill_half_row_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        request_slices = self._get_vllm_request_slices(x)
        if request_slices is None:
            return None

        cache_key = (
            "prefill_half",
            x.device.type,
            x.device.index,
            x.shape[0],
            tuple(request_slices),
        )
        cached = self._get_cached_forward_mask(cache_key)
        if cached is not None:
            return cached

        row_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for start, end in request_slices:
            query_len = end - start
            if query_len == 1:
                row_mask[start:end] = True
                continue
            half_query_len = query_len // 2
            if half_query_len > 0:
                row_mask[end - half_query_len : end] = True

        self._set_cached_forward_mask(cache_key, row_mask)
        return row_mask

    def _get_cached_forward_mask(self, cache_key: tuple) -> torch.Tensor | None:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return None
        context = get_forward_context()
        cache = context.additional_kwargs.setdefault("teal_sparsify_masks", {})
        return cache.get(cache_key)

    def _set_cached_forward_mask(self, cache_key: tuple, mask: torch.Tensor) -> None:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return
        context = get_forward_context()
        cache = context.additional_kwargs.setdefault("teal_sparsify_masks", {})
        cache[cache_key] = mask

    def _get_vllm_request_slices(self, x: torch.Tensor) -> list[tuple[int, int]] | None:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return None

        context = get_forward_context()
        metadata = context.attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else None
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()), None)
        if metadata is None or not hasattr(metadata, "query_start_loc"):
            return None

        query_start_loc = metadata.query_start_loc
        if query_start_loc is None or query_start_loc.numel() < 2:
            return None

        num_actual_tokens = getattr(metadata, "num_actual_tokens", x.shape[0])
        num_actual_tokens = min(int(num_actual_tokens), x.shape[0])

        starts = query_start_loc.detach().to("cpu").tolist()
        request_slices: list[tuple[int, int]] = []
        for start, end in zip(starts, starts[1:]):
            start = max(0, min(int(start), num_actual_tokens))
            end = max(start, min(int(end), num_actual_tokens))
            if end > start:
                request_slices.append((start, end))
        return request_slices

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold.item():.4f}, "
            f"decode_only={self.decode_only}, "
            f"apply_all_tokens={self.apply_all_tokens}, "
            f"prefill_sparsify={self.prefill_sparsify}"
        )


def larosa_topk(x: torch.Tensor, sparsity_level: float) -> torch.Tensor:
    """Apply La RoSA's per-token magnitude top-k sparsity on the last dim."""
    if sparsity_level <= 0.0:
        return x
    if sparsity_level >= 1.0:
        raise ValueError(
            "La RoSA sparsity_level must be lower than 1.0, "
            f"got {sparsity_level}."
        )

    keep = int((1.0 - sparsity_level) * x.shape[-1])
    if keep < 1:
        raise ValueError(
            "La RoSA sparsity_level keeps fewer than one activation for "
            f"hidden size {x.shape[-1]}: sparsity_level={sparsity_level}."
        )

    topk_values, _ = torch.topk(torch.abs(x), keep, dim=-1)
    keep_threshold = topk_values[..., -1:]
    return torch.where(torch.abs(x) >= keep_threshold, x, torch.zeros_like(x))


class LaRosaSparsifyFn(SparsifyFn):
    """La RoSA runtime sparsifier aligned with the official HF backend.

    For the first hidden-state sparsification site (attention qkv input and
    MLP gate/up input), La RoSA rotates activations by ``Q``, applies per-token
    top-k sparsity, and rotates back by ``Q.T``. For the second site (attention
    output and MLP intermediate), it applies top-k directly. When the
    first-site linear weight has been pre-rotated as ``W <- W @ Q``, the sparse
    linear fast path can skip the explicit ``@ Q.T`` unrotation.
    """

    def __init__(
        self,
        sparsity_level: float,
        rotation: torch.Tensor | None = None,
        rotate_input: bool = False,
        decode_only: bool = False,
        apply_all_tokens: bool = True,
        prefill_sparsify: str = "all",
        use_sparse_gemv: bool = False,
    ) -> None:
        super().__init__(
            threshold=torch.tensor(0.0),
            decode_only=decode_only,
            apply_all_tokens=apply_all_tokens,
            prefill_sparsify=prefill_sparsify,
            use_sparse_gemv=use_sparse_gemv,
        )
        if rotate_input and rotation is None:
            raise ValueError("La RoSA rotate_input=True requires a rotation matrix.")
        if rotation is not None:
            if rotation.dim() != 2 or rotation.shape[0] != rotation.shape[1]:
                raise ValueError(
                    "La RoSA rotation matrix must be square, got "
                    f"shape={tuple(rotation.shape)}."
                )
            rotation = rotation.to(dtype=torch.float32)
        self.sparsity_level = float(sparsity_level)
        self.rotate_input = rotate_input
        self.register_buffer("rotation", rotation)

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity_level <= 0.0:
            return x

        original_dtype = x.dtype
        if not self.rotate_input:
            return larosa_topk(x, self.sparsity_level)

        if self.rotation is None:
            raise ValueError("La RoSA rotation matrix is not initialized.")
        if x.shape[-1] != self.rotation.shape[0]:
            raise ValueError(
                "La RoSA rotation hidden size mismatch: "
                f"input hidden={x.shape[-1]}, rotation={tuple(self.rotation.shape)}."
            )

        rotation = self.rotation.to(dtype=torch.float32, device=x.device)
        rotated_x = torch.matmul(x.to(dtype=torch.float32), rotation)
        sparse_rotated_x = larosa_topk(rotated_x, self.sparsity_level)
        unrotated_x = torch.matmul(sparse_rotated_x, rotation.t())
        return unrotated_x.to(dtype=original_dtype)

    def _sparse_linear_input(
        self,
        x: torch.Tensor,
        linear_layer: nn.Module,
    ) -> torch.Tensor | None:
        if not self.rotate_input:
            return x
        if not getattr(linear_layer, "_larosa_sparse_weight_merged", False):
            return None
        if self.rotation is None:
            raise ValueError("La RoSA rotation matrix is not initialized.")
        if x.shape[-1] != self.rotation.shape[0]:
            raise ValueError(
                "La RoSA rotation hidden size mismatch: "
                f"input hidden={x.shape[-1]}, rotation={tuple(self.rotation.shape)}."
            )
        rotation = self.rotation.to(dtype=torch.float32, device=x.device)
        rotated_x = torch.matmul(x.to(dtype=torch.float32), rotation)
        return rotated_x.to(dtype=x.dtype)

    def _sparse_linear_threshold(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor | None, bool]:
        if self.sparsity_level <= 0.0:
            return torch.zeros(x.shape[0], dtype=torch.float32, device=x.device), True
        keep = int((1.0 - self.sparsity_level) * x.shape[-1])
        if keep < 1:
            raise ValueError(
                "La RoSA sparsity_level keeps fewer than one activation for "
                f"hidden size {x.shape[-1]}: sparsity_level={self.sparsity_level}."
            )
        topk_values, _ = torch.topk(x.abs().to(dtype=torch.float32), keep, dim=-1)
        return topk_values[..., -1].contiguous(), True

    def extra_repr(self) -> str:
        rotation_shape = None if self.rotation is None else tuple(self.rotation.shape)
        return (
            f"sparsity_level={self.sparsity_level:.4f}, "
            f"rotate_input={self.rotate_input}, "
            f"rotation_shape={rotation_shape}, "
            f"decode_only={self.decode_only}, "
            f"apply_all_tokens={self.apply_all_tokens}, "
            f"prefill_sparsify={self.prefill_sparsify}"
        )
