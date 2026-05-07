# SPDX-License-Identifier: Apache-2.0
"""Configuration for activation sparsity (TEAL / La RoSA)."""

from typing import Optional

from vllm.config.utils import config, get_hash_factors, hash_factors


@config
class ActivationSparsityConfig:
    """Configuration for TEAL / La RoSA activation sparsity.

    Phase 0 (MVP) defaults are chosen for correctness verification:
    - ``apply_all_tokens=True`` to avoid prefill/decode detection complexity.
    - ``strict_unsupported_check=True`` to fail fast on TP/quant/LoRA.
    - ``use_sparse_gemv=False`` because sparse GEMV is a Phase 2 experiment.
    """

    # Master switch
    enable: bool = False
    """Whether to enable activation sparsity."""

    # Method selection
    method: str = "teal"
    """Sparsity method: ``"teal"`` or ``"larosa"``."""

    # Sparsity ratio
    uniform_sparsity: float = 0.0
    """Global uniform sparsity ratio in [0, 1]. 0.4 means 40% of activations
    are zeroed out."""

    # Calibration artifacts
    calibration_path: Optional[str] = None
    """Directory containing calibration artifacts:
    ``histograms.pt``, per-layer ``threshold.pt``, and (for La RoSA)
    ``D.pt`` / ``inv_D.pt``."""

    # Decode-only guard
    decode_only: bool = False
    """If True, only apply sparsity during decode (not prefill).
    Phase 0 defaults to False."""

    # Phase 0 safety switch
    apply_all_tokens: bool = True
    """If True, apply sparsity to all tokens regardless of prefill/decode.
    This simplifies Phase 0 correctness validation."""

    # Unsupported combination guard
    strict_unsupported_check: bool = True
    """If True, raise an error when activation sparsity is combined with
    unsupported features (tensor parallelism, quantization, LoRA)."""

    # Phase 2 experimental flag
    use_sparse_gemv: bool = False
    """Enable Triton sparse GEMV custom op. Phase 2 experimental feature."""

    def compute_hash(self) -> str:
        """Return a hash that uniquely identifies this sparsity config.

        Must be included in :meth:`VllmConfig.compute_hash` so that
        compilation caches are invalidated when sparsity settings change.
        """
        factors = get_hash_factors(self, ignored_factors=set())
        return hash_factors(factors)
