# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Strict validation helpers shared by SliceGPT checkpoint converters."""

from collections.abc import Mapping

import torch

_SLICING_DIM_KEYS = (
    "attention_input_dimensions",
    "attention_output_dimensions",
    "mlp_input_dimensions",
    "mlp_output_dimensions",
)


def normalize_slicing_conf(raw: Mapping[str, object], num_layers: int) -> dict:
    """Validate and normalize a serialized ``SlicingConfig``.

    SliceGPT serializes layer indices as JSON object keys.  Conversion must not
    accept omitted, duplicated, or out-of-range indices: doing so can produce a
    checkpoint whose parameter names load but whose residual widths are wrong.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    def layer_dims(key: str) -> list[int]:
        value = raw.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"slicing config field {key!r} must be a mapping")

        dims: list[int | None] = [None] * num_layers
        for raw_index, raw_dim in value.items():
            try:
                index = int(raw_index)
                dim = int(raw_dim)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"slicing config field {key!r} has invalid entry "
                    f"{raw_index!r}: {raw_dim!r}"
                ) from exc
            if not 0 <= index < num_layers:
                raise ValueError(
                    f"slicing config field {key!r} has out-of-range layer "
                    f"index {index}; expected [0, {num_layers})"
                )
            if dim <= 0:
                raise ValueError(
                    f"slicing config field {key!r} layer {index} must be "
                    f"positive, got {dim}"
                )
            if dims[index] is not None:
                raise ValueError(
                    f"slicing config field {key!r} defines layer {index} twice"
                )
            dims[index] = dim

        missing = [index for index, dim in enumerate(dims) if dim is None]
        if missing:
            raise ValueError(
                f"slicing config field {key!r} is missing layer indices {missing}"
            )
        return [int(dim) for dim in dims]

    embedding_dimensions = raw.get("embedding_dimensions")
    if isinstance(embedding_dimensions, Mapping):
        if len(embedding_dimensions) != 1:
            raise ValueError(
                "slicing config field 'embedding_dimensions' must contain "
                "exactly one embedding width"
            )
        embedding_dim = next(iter(embedding_dimensions.values()))
    elif isinstance(embedding_dimensions, list) and len(embedding_dimensions) == 1:
        embedding_dim = embedding_dimensions[0]
    else:
        raise ValueError(
            "slicing config field 'embedding_dimensions' must be a one-item "
            "mapping or list"
        )

    try:
        embedding_dim = int(embedding_dim)
        final_norm_dim = int(raw["head_dimension"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "slicing config must contain positive 'embedding_dimensions' and "
            "'head_dimension' values"
        ) from exc
    if embedding_dim <= 0 or final_norm_dim <= 0:
        raise ValueError("slicing config dimensions must be positive")

    normalized = {
        "embedding_dim": embedding_dim,
        "final_norm_dim": final_norm_dim,
    }
    for key in _SLICING_DIM_KEYS:
        normalized[key.replace("_dimensions", "_dims")] = layer_dims(key)
    return normalized


def validate_state_keys(state: Mapping[str, object], expected: set[str]) -> None:
    """Reject missing, unsupported, and non-tensor checkpoint entries."""

    missing = expected.difference(state)
    unexpected = {
        name
        for name, value in state.items()
        if name not in expected and not is_ignored_state_key(name, value)
    }
    non_tensor = {
        name
        for name in expected.intersection(state)
        if not torch.is_tensor(state[name])
    }
    if missing or unexpected or non_tensor:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        if non_tensor:
            details.append(f"non_tensor={sorted(non_tensor)}")
        raise ValueError("invalid SliceGPT state_dict: " + "; ".join(details))


def is_ignored_state_key(name: str, value: object) -> bool:
    """Return whether a known non-persistent SliceGPT tensor may be dropped."""
    return torch.is_tensor(value) and (
        "rotary_emb" in name
        or name.endswith(".inv_freq")
        or name.endswith("input_layernorm.weight")
        or name.endswith("post_attention_layernorm.weight")
        or name == "model.norm.weight"
    )


def validate_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    """Raise a stable user-facing error for a checkpoint shape mismatch."""
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name}: shape {actual} != expected {expected}")


def infer_attention_head_dim(q_proj: torch.Tensor, num_attention_heads: int) -> int:
    """Infer and validate the unmodified attention head dimension."""
    if num_attention_heads <= 0:
        raise ValueError(
            f"num_attention_heads must be positive, got {num_attention_heads}"
        )
    if q_proj.ndim != 2 or q_proj.shape[0] % num_attention_heads:
        raise ValueError(
            "model.layers.0.self_attn.q_proj.weight must have a row count "
            f"divisible by num_attention_heads={num_attention_heads}; got "
            f"shape {tuple(q_proj.shape)}"
        )
    head_dim = q_proj.shape[0] // num_attention_heads
    if head_dim <= 0:
        raise ValueError("inferred attention head dimension must be positive")
    return head_dim


def expected_state_names(num_layers: int, *, attention_bias: bool) -> set[str]:
    """Return every persistent tensor required by a SliceGPT checkpoint."""
    names = {"model.embed_tokens.weight", "lm_head.weight"}
    for index in range(num_layers):
        prefix = f"model.layers.{index}"
        names.update(
            {
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.attn_shortcut_Q",
                f"{prefix}.mlp_shortcut_Q",
            }
        )
        if attention_bias:
            names.update(
                {
                    f"{prefix}.self_attn.q_proj.bias",
                    f"{prefix}.self_attn.k_proj.bias",
                    f"{prefix}.self_attn.v_proj.bias",
                }
            )
    return names
