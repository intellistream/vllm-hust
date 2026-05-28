# SPDX-License-Identifier: Apache-2.0
"""Activation sparsity support for vLLM (TEAL / La RoSA)."""

from vllm.sparsity.config import ActivationSparsityConfig
from vllm.sparsity.distribution import Distribution, SparsifyFn
from vllm.sparsity.rotation import RotationTransform

__all__ = [
    "ActivationSparsityConfig",
    "build_sparsifier",
    "Distribution",
    "get_activation_sparsity_config",
    "load_threshold",
    "merge_larosa_rotation_into_linear",
    "RotationTransform",
    "SparsifyFn",
]


def __getattr__(name: str):
    if name in {"build_sparsifier", "merge_larosa_rotation_into_linear"}:
        from vllm.sparsity import layers

        return getattr(layers, name)
    if name in {"get_activation_sparsity_config", "load_threshold"}:
        from vllm.sparsity import utils

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
