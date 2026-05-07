# SPDX-License-Identifier: Apache-2.0
"""TEAL distribution and sparsify function."""

import torch
from torch import nn

from vllm.forward_context import get_forward_context


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
        return {
            name: cls(hist, bin_edges)
            for name, hist in histograms.items()
        }


class SparsifyFn(nn.Module):
    """PyTorch module that applies a magnitude-based sparsity mask.

    For Phase 0 this uses a dense ``torch.where`` so that correctness
    can be validated without requiring a sparse GEMV kernel.

    When ``rotation`` is provided (La RoSA mode), the forward pass becomes:
    ``x @ D -> sparsify -> x @ inv_D``.
    """

    def __init__(
        self,
        threshold: torch.Tensor,
        apply_all_tokens: bool = True,
        rotation: "RotationTransform | None" = None,
    ) -> None:
        super().__init__()
        # Register as buffer so threshold moves with module.to(device/dtype)
        self.register_buffer("threshold", threshold)
        self.apply_all_tokens = apply_all_tokens
        self.rotation = rotation
        if rotation is not None:
            # Register rotation as a submodule so it follows .to(device/dtype)
            self.add_module("_rotation", rotation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Phase 0: apply to all tokens by default.
        # Phase 1 will add decode-only detection here.
        if not self.apply_all_tokens and not self._is_decode_only_batch():
            return x

        if self.rotation is not None:
            x = self.rotation(x)

        x = torch.where(
            x.abs() > self.threshold.to(dtype=x.dtype, device=x.device),
            x,
            torch.zeros_like(x),
        )

        if self.rotation is not None:
            x = self.rotation.inverse(x).to(x.dtype)

        return x

    def _is_decode_only_batch(self) -> bool:
        """Return True if the current forward batch is decode-only.

        Phase 1 implementation: inspect ForwardContext.attn_metadata.
        For now always returns False so that ``apply_all_tokens`` controls
        the behaviour.
        """
        # TODO(Phase 1): inspect get_forward_context().attn_metadata to
        # determine whether every sequence in the batch has query_len == 1.
        return False

    def extra_repr(self) -> str:
        has_rot = self.rotation is not None
        return (
            f"threshold={self.threshold.item():.4f}, "
            f"apply_all_tokens={self.apply_all_tokens}, "
            f"has_rotation={has_rot}"
        )
