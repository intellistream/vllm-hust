# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.sparsity.config import ActivationSparsityConfig


def test_activation_sparsity_config_defaults():
    cfg = ActivationSparsityConfig()
    assert cfg.enable is False
    assert cfg.method == "teal"
    assert cfg.uniform_sparsity == 0.0
    assert cfg.calibration_path is None
    assert cfg.decode_only is False
    assert cfg.apply_all_tokens is True
    assert cfg.strict_unsupported_check is True
    assert cfg.use_sparse_gemv is False


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


def test_activation_sparsity_config_invalid_method():
    # Pydantic dataclass with extra="forbid" should reject unknown fields
    with pytest.raises(TypeError):
        ActivationSparsityConfig(unknown_field=True)
