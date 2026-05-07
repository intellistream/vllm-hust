# SPDX-License-Identifier: Apache-2.0
"""Layer helpers for injecting activation sparsity."""

import os
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.sparsity.config import ActivationSparsityConfig
from vllm.sparsity.distribution import SparsifyFn
from vllm.sparsity.rotation import RotationTransform
from vllm.sparsity.utils import load_threshold

logger = init_logger(__name__)


def build_sparsifier(
    sparsity_config: ActivationSparsityConfig,
    layer_idx: int,
    proj_name: str,
    device: torch.device | str = "cpu",
) -> Optional[SparsifyFn]:
    """Build a :class:`SparsifyFn` for a given layer and projection.

    Args:
        sparsity_config: The activation sparsity configuration.
        layer_idx: Layer index.
        proj_name: Projection name, e.g. ``"mlp.gate_up"``, ``"self_attn.qkv"``.
        device: Device to load the threshold onto.

    Returns:
        A :class:`SparsifyFn` instance, or ``None`` if sparsity is disabled.
    """
    if not sparsity_config.enable:
        return None

    if not sparsity_config.calibration_path:
        logger.warning_once(
            "activation_sparsity.enable=True but calibration_path is not set. "
            "Sparsity will not be applied."
        )
        return None

    # Load pre-computed threshold for this layer/proj
    threshold = load_threshold(
        sparsity_config.calibration_path,
        layer_idx,
        proj_name,
        device=str(device),
    )

    sparsify_fn = SparsifyFn(
        threshold=threshold,
        apply_all_tokens=sparsity_config.apply_all_tokens,
    )

    # La RoSA: wrap with rotation if D/inv_D exist
    if sparsity_config.method == "larosa":
        rotation_dir = os.path.join(
            sparsity_config.calibration_path,
            f"layers.{layer_idx}.{proj_name}",
        )
        d_path = os.path.join(rotation_dir, "D.pt")
        inv_d_path = os.path.join(rotation_dir, "inv_D.pt")
        if os.path.exists(d_path) and os.path.exists(inv_d_path):
            rotation = RotationTransform(d_path=d_path, inv_d_path=inv_d_path)
            # Attach rotation to the sparsify_fn module so it moves together
            sparsify_fn.rotation = rotation  # type: ignore[attr-defined]
            # Monkey-patch forward to include rotation
            _orig_forward = sparsify_fn.forward

            def _rotated_forward(x: torch.Tensor) -> torch.Tensor:
                x_rot = rotation(x)
                x_sparse = _orig_forward(x_rot)
                return rotation.inverse(x_sparse).to(x.dtype)

            sparsify_fn.forward = _rotated_forward  # type: ignore[method-assign]
            logger.debug(
                "La RoSA rotation enabled for layer %d projection %s",
                layer_idx,
                proj_name,
            )

    return sparsify_fn
