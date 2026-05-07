# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.sparsity.config import ActivationSparsityConfig


class DummyConfig:
    """Minimal stand-in for config objects that need compute_hash."""

    def __init__(self, hash_val: str = "dummy"):
        self._hash = hash_val

    def compute_hash(self) -> str:
        return self._hash


def test_activation_sparsity_in_vllm_config_hash():
    """ActivationSparsityConfig.compute_hash() must be deterministic
    and sensitive to field changes."""
    cfg1 = ActivationSparsityConfig(enable=True, uniform_sparsity=0.4)
    cfg2 = ActivationSparsityConfig(enable=True, uniform_sparsity=0.5)

    h1 = cfg1.compute_hash()
    h2 = cfg2.compute_hash()
    assert isinstance(h1, str)
    assert h1 != h2


def test_activation_sparsity_config_validation():
    """Pydantic extra=forbid should reject unknown keys."""
    with pytest.raises(TypeError):
        ActivationSparsityConfig(unknown_field=123)
