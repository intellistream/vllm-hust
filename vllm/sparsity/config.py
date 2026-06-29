# SPDX-License-Identifier: Apache-2.0
"""Configuration for activation sparsity (TEAL / La RoSA)."""

import hashlib
import json
from dataclasses import fields

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

_SUPPORTED_TARGET_PROJECTIONS = {
    "self_attn.qkv",
    "self_attn.o",
    "mlp.gate_up",
    "mlp.down",
}


@dataclass(config=ConfigDict(extra="forbid"))
class ActivationSparsityConfig:
    """Configuration for TEAL / La RoSA activation sparsity.

    Defaults are chosen to match the public TEAL HF reference:
    - ``prefill_sparsify="half"`` sparsifies only the last half of prefill
      activations while still sparsifying single-token decode activations.
    - ``strict_unsupported_check=True`` to fail fast on TP/quant/LoRA.
    - ``use_sparse_gemv=False`` because sparse GEMV is an experimental
      backend-specific path.
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
    calibration_path: str | None = None
    """Directory containing calibration artifacts:
    ``histograms.pt``, per-layer ``threshold.pt``, and (for La RoSA)
    official ``histograms/layer-{idx}/self_attn/D.pt`` rotations."""

    # Decode-only guard
    decode_only: bool = False
    """If True, only apply sparsity during decode (not prefill).
    Phase 0 defaults to False."""

    # Legacy safety switch
    apply_all_tokens: bool = False
    """If True, apply sparsity to all tokens regardless of prefill/decode.
    This preserves the original Phase 0 behaviour and overrides
    ``prefill_sparsify``."""

    # Prefill sparsification policy
    prefill_sparsify: str = "half"
    """Prefill sparsification policy when ``decode_only=False``:
    ``"half"`` matches the public TEAL HF reference by sparsifying the last
    half of each prefill query, ``"all"`` sparsifies all prefill tokens, and
    ``"none"`` sparsifies only decode tokens."""

    # Unsupported combination guard
    strict_unsupported_check: bool = True
    """If True, raise an error when activation sparsity is combined with
    unsupported features (tensor parallelism, quantization, LoRA)."""

    # Phase 2 experimental flag
    use_sparse_gemv: bool = False
    """Enable backend sparse GEMV custom op. Experimental feature."""

    # Optional projection filter
    target_projections: list[str] | None = None
    """If set, only apply activation sparsity to these projection names, e.g.
    ``["mlp.gate_up"]`` for the Ascend-friendly fused-MLP path."""

    # Optional layer filter
    target_layers: list[int] | None = None
    """If set, only apply activation sparsity to these zero-based layer ids."""

    def __post_init__(self) -> None:
        if self.method not in {"teal", "larosa"}:
            raise ValueError(
                "Activation sparsity method must be 'teal' or 'larosa', "
                f"got {self.method!r}."
            )
        if self.prefill_sparsify not in {"half", "all", "none"}:
            raise ValueError(
                "prefill_sparsify must be one of 'half', 'all', or 'none', "
                f"got {self.prefill_sparsify!r}."
            )
        if self.target_projections is not None:
            unknown = sorted(
                set(self.target_projections) - _SUPPORTED_TARGET_PROJECTIONS
            )
            if unknown:
                raise ValueError(
                    "target_projections contains unsupported projection(s): "
                    f"{unknown}. Supported projections are "
                    f"{sorted(_SUPPORTED_TARGET_PROJECTIONS)}."
                )
        if self.target_layers is not None:
            unknown_layers = [
                layer_idx
                for layer_idx in self.target_layers
                if not isinstance(layer_idx, int) or layer_idx < 0
            ]
            if unknown_layers:
                raise ValueError(
                    "target_layers must contain non-negative integer layer ids, "
                    f"got {unknown_layers}."
                )

    def compute_hash(self) -> str:
        """Return a hash that uniquely identifies this sparsity config.

        Must be included in :meth:`VllmConfig.compute_hash` so that
        compilation caches are invalidated when sparsity settings change.
        """
        factors = {field.name: getattr(self, field.name) for field in fields(self)}
        return hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()
