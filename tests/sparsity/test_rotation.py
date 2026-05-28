# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.sparsity.rotation import (
    RotationTransform,
    merge_rotation_into_weight,
    merge_rotation_into_weight_loader,
)


def test_rotation_transform_roundtrip():
    """x @ D @ inv_D should recover x (up to dtype)."""
    hidden = 16
    d = torch.randn(hidden, hidden)
    inv_d = torch.linalg.inv(d)

    rot = RotationTransform(d_matrix=d, inv_d_matrix=inv_d)
    x = torch.randn(2, hidden)

    x_rot = rot(x)
    x_recovered = rot.inverse(x_rot)
    assert torch.allclose(x_recovered, x.to(x_recovered.dtype), atol=1e-4)


def test_rotation_transform_shape():
    d = torch.eye(8)
    inv_d = torch.eye(8)
    rot = RotationTransform(d_matrix=d, inv_d_matrix=inv_d)

    x = torch.randn(3, 8)
    out = rot(x)
    assert out.shape == (3, 8)


def test_merge_rotation_into_weight():
    weight = torch.randn(4, 8)
    rotation = torch.randn(8, 8)

    merged = merge_rotation_into_weight(weight, rotation)

    assert torch.allclose(merged, weight @ rotation, atol=1e-5)


def test_merge_rotation_into_weight_loader():
    class DummyWeight:
        def __init__(self):
            self.loaded = None
            self.weight_loader = self._load

        def _load(self, param, loaded_weight):
            self.loaded = loaded_weight

    class DummyLinear:
        def __init__(self):
            self.weight = DummyWeight()

    linear = DummyLinear()
    rotation = torch.eye(4) * 2
    assert merge_rotation_into_weight_loader(
        linear, rotation, proj_name="layers.0.self_attn.qkv"
    )

    loaded_weight = torch.ones(3, 4)
    linear.weight.weight_loader(linear.weight, loaded_weight)

    assert torch.allclose(linear.weight.loaded, loaded_weight @ rotation)
