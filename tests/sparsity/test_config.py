# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.sparsity.config import (
    ActivationSparsityConfig,
    validate_activation_sparsity_compatibility,
)


def test_activation_sparsity_config_defaults():
    cfg = ActivationSparsityConfig()
    assert cfg.enable is False
    assert cfg.method == "teal"
    assert cfg.uniform_sparsity == 0.0
    assert cfg.calibration_path is None
    assert cfg.decode_only is False
    assert cfg.apply_all_tokens is False
    assert cfg.prefill_sparsify == "half"
    assert cfg.strict_unsupported_check is True
    assert cfg.use_sparse_gemv is False
    assert cfg.target_projections is None


def test_activation_sparsity_config_hash():
    cfg = ActivationSparsityConfig(enable=True, uniform_sparsity=0.4)
    h1 = cfg.compute_hash()
    h2 = cfg.compute_hash()
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64

    cfg2 = ActivationSparsityConfig(enable=True, uniform_sparsity=0.5)
    h3 = cfg2.compute_hash()
    assert h3 != h1

    cfg3 = ActivationSparsityConfig(
        enable=True,
        uniform_sparsity=0.4,
        target_projections=["mlp.gate_up"],
    )
    assert cfg3.compute_hash() != h1


def test_activation_sparsity_config_invalid_method():
    # Pydantic dataclass with extra="forbid" should reject unknown fields
    with pytest.raises((TypeError, ValidationError)):
        ActivationSparsityConfig(unknown_field=True)

    with pytest.raises((ValueError, ValidationError)):
        ActivationSparsityConfig(prefill_sparsify="last_token")

    with pytest.raises((ValueError, ValidationError)):
        ActivationSparsityConfig(target_projections=["mlp.gate"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.0])
def test_activation_sparsity_config_rejects_invalid_uniform_sparsity(value):
    with pytest.raises((ValueError, ValidationError), match="uniform_sparsity"):
        ActivationSparsityConfig(uniform_sparsity=value)


def test_larosa_rejects_second_site_sparsity_at_or_above_one():
    with pytest.raises((ValueError, ValidationError), match="5/6"):
        ActivationSparsityConfig(method="larosa", uniform_sparsity=0.84)


def test_sparse_gemv_requires_explicit_activation_sparsity_opt_in():
    with pytest.raises((ValueError, ValidationError), match="enable=True"):
        ActivationSparsityConfig(use_sparse_gemv=True)


def test_enabled_sparsity_requires_calibration_and_rejects_unsupported_modes():
    without_calibration = ActivationSparsityConfig(enable=True)
    with pytest.raises(ValueError, match="calibration_path"):
        validate_activation_sparsity_compatibility(
            without_calibration,
            tensor_parallel_size=1,
            has_quantization=False,
            has_lora=False,
        )

    config = ActivationSparsityConfig(
        enable=True,
        calibration_path="/calibration",
        strict_unsupported_check=False,
    )
    for compatibility in (
        {"tensor_parallel_size": 2, "has_quantization": False, "has_lora": False},
        {"tensor_parallel_size": 1, "has_quantization": True, "has_lora": False},
        {"tensor_parallel_size": 1, "has_quantization": False, "has_lora": True},
    ):
        with pytest.raises(ValueError):
            validate_activation_sparsity_compatibility(config, **compatibility)


def test_disabled_sparsity_does_not_reject_unrelated_runtime_features():
    validate_activation_sparsity_compatibility(
        ActivationSparsityConfig(enable=False),
        tensor_parallel_size=8,
        has_quantization=True,
        has_lora=True,
    )
