# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.sparsity.rotation import RotationTransform


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
