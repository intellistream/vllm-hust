# SPDX-License-Identifier: Apache-2.0
"""La RoSA rotation matrix support."""

import torch
from torch import nn


class RotationTransform(nn.Module):
    """Wrapper for La RoSA rotation matrix ``D`` and its inverse ``inv_D``.

    Applies ``x @ D`` before sparsification and ``x @ inv_D`` after.
    Both matrices are registered as buffers so they follow the module's
    device/dtype automatically.

    Phase 3 only: La RoSA is not part of the Phase 0 MVP.
    """

    def __init__(
        self,
        d_path: str | None = None,
        inv_d_path: str | None = None,
        d_matrix: torch.Tensor | None = None,
        inv_d_matrix: torch.Tensor | None = None,
    ) -> None:
        """Args:
            d_path: Path to a ``.pt`` file containing ``D``.
            inv_d_path: Path to a ``.pt`` file containing ``inv_D``.
            d_matrix: ``D`` tensor (alternative to ``d_path``).
            inv_d_matrix: ``inv_D`` tensor (alternative to ``inv_d_path``).
        """
        super().__init__()

        if d_matrix is not None:
            d = d_matrix
        elif d_path is not None:
            d = torch.load(d_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("One of d_path or d_matrix must be provided")

        if inv_d_matrix is not None:
            inv_d = inv_d_matrix
        elif inv_d_path is not None:
            inv_d = torch.load(
                inv_d_path, map_location="cpu", weights_only=True
            )
        else:
            raise ValueError(
                "One of inv_d_path or inv_d_matrix must be provided"
            )

        self.register_buffer("D", d)
        self.register_buffer("inv_D", inv_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: ``x @ D``."""
        return torch.matmul(x.to(self.D.dtype), self.D)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: ``x @ inv_D``."""
        return torch.matmul(x.to(self.inv_D.dtype), self.inv_D)

    def extra_repr(self) -> str:
        return f"D_shape={tuple(self.D.shape)}, inv_D_shape={tuple(self.inv_D.shape)}"
