# SPDX-License-Identifier: Apache-2.0
"""Layer helpers for injecting activation sparsity."""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.sparsity.config import ActivationSparsityConfig
from vllm.sparsity.distribution import LaRosaSparsifyFn, SparsifyFn
from vllm.sparsity.rotation import (
    find_rotation_matrix_path,
    load_rotation_matrix,
    merge_rotation_into_weight_loader,
)
from vllm.sparsity.utils import load_threshold

logger = init_logger(__name__)


def build_sparsifier(
    sparsity_config: ActivationSparsityConfig,
    layer_idx: int,
    proj_name: str,
    device: torch.device | str = "cpu",
) -> SparsifyFn | None:
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

    if sparsity_config.method == "larosa":
        return _build_larosa_sparsifier(
            sparsity_config,
            layer_idx,
            proj_name,
            device,
        )

    # Load pre-computed TEAL threshold for this layer/proj
    threshold = load_threshold(
        sparsity_config.calibration_path,
        layer_idx,
        proj_name,
        device=str(device),
    )

    sparsify_fn = SparsifyFn(
        threshold=threshold,
        decode_only=sparsity_config.decode_only,
        apply_all_tokens=sparsity_config.apply_all_tokens,
        prefill_sparsify=sparsity_config.prefill_sparsify,
        use_sparse_gemv=sparsity_config.use_sparse_gemv,
        expected_sparsity=sparsity_config.uniform_sparsity,
    )

    return sparsify_fn


def _build_larosa_sparsifier(
    sparsity_config: ActivationSparsityConfig,
    layer_idx: int,
    proj_name: str,
    device: torch.device | str = "cpu",
) -> LaRosaSparsifyFn | None:
    """Build La RoSA's official runtime top-k sparsifier for a projection."""
    first_site_projs = {"self_attn.qkv", "mlp.gate_up"}
    second_site_projs = {"self_attn.o", "mlp.down"}

    if proj_name in first_site_projs:
        sparse_level = sparsity_config.uniform_sparsity * 0.8
        rotation_path = find_rotation_matrix_path(
            sparsity_config.calibration_path,
            layer_idx,
            proj_name,
        )
        if rotation_path is None:
            raise FileNotFoundError(
                "La RoSA rotation matrix not found for "
                f"layer {layer_idx}, projection '{proj_name}'. Expected "
                f"{sparsity_config.calibration_path}/histograms/"
                f"layer-{layer_idx}/self_attn/D.pt or "
                f"{sparsity_config.calibration_path}/layers."
                f"{layer_idx}.{proj_name}/D.pt."
            )
        rotation = load_rotation_matrix(rotation_path, device=device)
        return LaRosaSparsifyFn(
            sparsity_level=sparse_level,
            rotation=rotation,
            rotate_input=True,
            decode_only=sparsity_config.decode_only,
            apply_all_tokens=sparsity_config.apply_all_tokens,
            prefill_sparsify=sparsity_config.prefill_sparsify,
            use_sparse_gemv=sparsity_config.use_sparse_gemv,
            expected_sparsity=sparse_level,
        )

    if proj_name in second_site_projs:
        sparse_level = sparsity_config.uniform_sparsity * 1.2
        return LaRosaSparsifyFn(
            sparsity_level=sparse_level,
            rotation=None,
            rotate_input=False,
            decode_only=sparsity_config.decode_only,
            apply_all_tokens=sparsity_config.apply_all_tokens,
            prefill_sparsify=sparsity_config.prefill_sparsify,
            use_sparse_gemv=sparsity_config.use_sparse_gemv,
            expected_sparsity=sparse_level,
        )

    logger.warning_once("Unsupported La RoSA projection %s; skipping.", proj_name)
    return None


def merge_larosa_rotation_into_linear(
    sparsity_config: ActivationSparsityConfig | None,
    layer_idx: int,
    proj_name: str,
    linear_layer: nn.Module,
) -> bool:
    """Merge offline La RoSA rotation into a linear layer's load-time weight.

    Calibration artifacts are expected under:
    ``{calibration_path}/layers.{layer_idx}.{proj_name}/D.pt``.
    ``Q.pt`` and ``rotation.pt`` are accepted aliases for exporter convenience.
    """
    if sparsity_config is None or sparsity_config.method != "larosa":
        return False
    if not sparsity_config.enable or not sparsity_config.calibration_path:
        return False

    rotation_path = find_rotation_matrix_path(
        sparsity_config.calibration_path,
        layer_idx,
        proj_name,
    )
    if rotation_path is None:
        logger.debug(
            "No La RoSA rotation matrix for layer %d projection %s",
            layer_idx,
            proj_name,
        )
        return False

    if getattr(linear_layer, "quant_config", None) is not None:
        raise ValueError(
            "La RoSA load-time rotation merge currently supports only "
            f"unquantized linear layers; got quantized projection {proj_name}."
        )

    rotation = load_rotation_matrix(rotation_path)
    merged = merge_rotation_into_weight_loader(
        linear_layer,
        rotation,
        proj_name=f"layers.{layer_idx}.{proj_name}",
    )
    if merged:
        linear_layer._larosa_sparse_weight_merged = True
    return merged
