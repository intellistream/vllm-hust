# SPDX-License-Identifier: Apache-2.0
"""La RoSA rotation matrix support."""

import os
from collections.abc import Callable

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class RotationTransform(nn.Module):
    """Legacy wrapper for explicit La RoSA online rotation.

    The vLLM inference path should prefer load-time weight merging via
    :func:`merge_rotation_into_weight_loader` to avoid runtime rotation FLOPs.
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


def find_rotation_matrix_path(
    calibration_path: str,
    layer_idx: int,
    proj_name: str,
) -> str | None:
    """Return the La RoSA rotation matrix path for a layer/projection."""
    rotation_dir = os.path.join(
        calibration_path,
        f"layers.{layer_idx}.{proj_name}",
    )
    for filename in ("D.pt", "Q.pt", "rotation.pt"):
        path = os.path.join(rotation_dir, filename)
        if os.path.exists(path):
            return path
    return None


def load_rotation_matrix(
    path: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load a square offline La RoSA rotation matrix."""
    rotation = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(rotation, torch.Tensor):
        raise TypeError(f"Rotation matrix at {path} is not a torch.Tensor")
    if rotation.dim() != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(
            "La RoSA rotation matrix must be square, got "
            f"shape={tuple(rotation.shape)} from {path}"
        )
    return rotation


def merge_rotation_into_weight(
    weight: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Right-multiply a linear weight by a La RoSA rotation matrix.

    vLLM linear weights are stored as ``[out_features, in_features]`` and
    applied as ``x @ weight.T``. Merging ``Q`` therefore uses ``W <- W @ Q``.
    """
    if weight.dim() != 2:
        return weight

    input_size = weight.shape[-1]
    if rotation.shape != (input_size, input_size):
        raise ValueError(
            "La RoSA rotation shape mismatch: "
            f"weight input size={input_size}, rotation shape={tuple(rotation.shape)}"
        )

    compute_dtype = torch.float32 if weight.device.type == "cpu" else weight.dtype
    rotated_weight = torch.matmul(
        weight.to(dtype=compute_dtype),
        rotation.to(device=weight.device, dtype=compute_dtype),
    )
    return rotated_weight.to(dtype=weight.dtype)


def merge_rotation_into_weight_loader(
    linear_layer: nn.Module,
    rotation: torch.Tensor,
    *,
    proj_name: str,
) -> bool:
    """Wrap a vLLM linear layer's weight loader to merge rotation at load time."""
    weight = getattr(linear_layer, "weight", None)
    if weight is None or not hasattr(weight, "weight_loader"):
        return False

    if getattr(weight, "_larosa_rotation_merged", False):
        return False

    original_loader: Callable = weight.weight_loader

    def weight_loader(param, loaded_weight, *args, **kwargs):
        merged_weight = merge_rotation_into_weight(loaded_weight, rotation)
        return original_loader(param, merged_weight, *args, **kwargs)

    weight.weight_loader = weight_loader
    weight._larosa_rotation_merged = True
    logger.debug("Merged La RoSA rotation into loader for %s", proj_name)
    return True
