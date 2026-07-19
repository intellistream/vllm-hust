# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for activation sparsity (TEAL / La RoSA)."""

import hashlib
import json
import math
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
    - unsupported TP/quant/LoRA combinations always fail closed.
    - ``use_sparse_gemv=False`` because sparse GEMV is an experimental
      backend-specific path.
    """

    # Master switch
    enable: bool = False
    """Whether to explicitly enable activation sparsity."""

    # Method selection
    method: str = "teal"
    """Sparsity method: ``"teal"`` or ``"larosa"``."""

    # Sparsity ratio
    uniform_sparsity: float = 0.0
    """Global uniform sparsity ratio in [0, 1). 0.4 means 40% of activations
    are zeroed out. La RoSA additionally requires a value lower than 5/6."""

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
    """Deprecated compatibility field.

    Unsupported tensor parallelism, quantization, and LoRA combinations always
    fail closed while activation sparsity is enabled.
    """

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
        if not math.isfinite(self.uniform_sparsity):
            raise ValueError(
                f"uniform_sparsity must be finite, got {self.uniform_sparsity!r}."
            )
        if not 0.0 <= self.uniform_sparsity < 1.0:
            raise ValueError(
                "uniform_sparsity must be in the range [0, 1), "
                f"got {self.uniform_sparsity!r}."
            )
        if self.method == "larosa" and self.uniform_sparsity * 1.2 >= 1.0:
            raise ValueError(
                "La RoSA uniform_sparsity must be lower than 5/6 so its "
                "second-site sparsity remains below 1, "
                f"got {self.uniform_sparsity!r}."
            )
        if self.use_sparse_gemv and not self.enable:
            raise ValueError(
                "use_sparse_gemv=True requires activation_sparsity.enable=True."
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


def validate_activation_sparsity_compatibility(
    sparsity_config: ActivationSparsityConfig | None,
    *,
    tensor_parallel_size: int,
    has_quantization: bool,
    has_lora: bool,
) -> None:
    """Reject runtime combinations whose semantics are not implemented."""
    if sparsity_config is None or not sparsity_config.enable:
        return
    if not sparsity_config.calibration_path:
        raise ValueError(
            "activation_sparsity.enable=True requires an explicit calibration_path."
        )
    if tensor_parallel_size > 1:
        raise ValueError(
            "Activation sparsity (TEAL / La RoSA) does not yet support "
            "tensor parallelism (tp_size > 1). Please set tp_size=1."
        )
    if has_quantization:
        raise ValueError(
            "Activation sparsity (TEAL / La RoSA) does not yet support "
            "quantization. Please use an unquantized model."
        )
    if has_lora:
        raise ValueError(
            "Activation sparsity (TEAL / La RoSA) does not yet support "
            "LoRA. Please disable LoRA."
        )
