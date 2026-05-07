# SPDX-License-Identifier: Apache-2.0
"""Utilities for activation sparsity calibration loading."""

import glob
import json
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    get_lock,
    maybe_download_from_modelscope,
)
from vllm.sparsity.config import ActivationSparsityConfig
from vllm.sparsity.distribution import Distribution

logger = init_logger(__name__)


def load_threshold(
    calibration_path: str,
    layer_idx: int,
    proj_name: str,
    device: str = "cpu",
) -> torch.Tensor:
    """Load a pre-computed threshold tensor for a specific layer and projection.

    Args:
        calibration_path: Directory containing calibration artifacts.
        layer_idx: Layer index (e.g. 0, 1, 2).
        proj_name: Projection name (e.g. ``"mlp.gate_up"``, ``"self_attn.qkv"``).
        device: Device to load the tensor onto.

    Returns:
        A 0-D or 1-D threshold tensor.

    Raises:
        FileNotFoundError: If no threshold file is found for the layer/proj.
    """
    # Expected filenames (in order of preference):
    # layers.0.mlp.gate_up.threshold.pt
    # layers.0.self_attn.qkv.threshold.pt
    basename = f"layers.{layer_idx}.{proj_name}.threshold.pt"
    path = os.path.join(calibration_path, basename)

    if not os.path.exists(path):
        # Fallback: glob for any matching file
        pattern = os.path.join(
            calibration_path, f"layers.{layer_idx}.{proj_name}*.pt"
        )
        matches = glob.glob(pattern)
        if matches:
            path = matches[0]
        else:
            raise FileNotFoundError(
                f"Threshold file not found for layer {layer_idx}, "
                f"projection '{proj_name}' (looked for {basename})"
            )

    threshold = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor(threshold, dtype=torch.float32, device=device)
    return threshold


def compute_thresholds_from_histograms(
    histograms_path: str,
    sparsity: float,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Compute per-projection thresholds from a histograms.pt file.

    Args:
        histograms_path: Path to ``histograms.pt``.
        sparsity: Target sparsity ratio in [0, 1].
        device: Device to place the resulting tensors on.

    Returns:
        Dict mapping projection name to threshold tensor.
    """
    data = torch.load(histograms_path, map_location="cpu", weights_only=False)
    # Expected structure:
    # {
    #   "histograms": {"layers.0.mlp.gate_up": tensor, ...},
    #   "bin_edges": tensor,
    # }
    histograms = data.get("histograms", data)
    bin_edges = data.get("bin_edges", None)

    if bin_edges is None:
        raise ValueError(
            "histograms file must contain 'bin_edges' to compute thresholds"
        )

    dists = Distribution.from_histograms(histograms, bin_edges)
    q = 0.5 + sparsity / 2.0
    thresholds = {
        name: torch.tensor(d.icdf(q), dtype=torch.float32, device=device)
        for name, d in dists.items()
    }
    return thresholds


def get_activation_sparsity_config(
    model_config: Any,
    load_config: Any,
    config_filename: str = "activation_sparsity_config.json",
) -> dict[str, Any]:
    """Look for ``activation_sparsity_config.json`` in the model directory.

    Follows the same remote/local resolution pattern as
    :func:`get_sparse_attention_config`.

    Args:
        model_config: The model config (usually ``ModelConfig``).
        load_config: The load config (usually ``LoadConfig``).
        config_filename: Name of the JSON config file to look for.

    Returns:
        The parsed JSON dict, or ``{}`` if the file does not exist.
    """
    import huggingface_hub
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import DisabledTqdm

    model_name_or_path = getattr(model_config, "model", None)
    if model_name_or_path is None:
        return {}

    model_name_or_path = (
        maybe_download_from_modelscope(model_name_or_path, None)
        or model_name_or_path
    )
    is_local = os.path.isdir(model_name_or_path)

    if not is_local:
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=getattr(model_config, "revision", None),
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    config_file = os.path.join(hf_folder, config_filename)
    if not os.path.exists(config_file):
        return {}

    with open(config_file) as f:
        config = json.load(f)
    logger.info("Loaded activation sparsity config from %s", config_file)
    return config
