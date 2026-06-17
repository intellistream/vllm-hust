# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.sparsity.rotation import (
    RotationTransform,
    find_rotation_matrix_path,
    load_rotation_matrix,
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


def test_merge_rotation_into_weight_absorbs_unrotation():
    weight = torch.randn(4, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    rotated_x = torch.randn(3, 8)

    merged = merge_rotation_into_weight(weight, q)

    runtime_unrotate = (rotated_x @ q.t()) @ weight.t()
    absorbed = rotated_x @ merged.t()
    assert torch.allclose(absorbed, runtime_unrotate, atol=1e-5)


def test_load_rotation_matrix_defaults_to_fp32(tmp_path):
    rotation_path = tmp_path / "D.pt"
    torch.save(torch.eye(4, dtype=torch.float64), rotation_path)

    rotation = load_rotation_matrix(str(rotation_path))

    assert rotation.dtype == torch.float32


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


def test_find_rotation_matrix_path_accepts_official_larosa_layout(tmp_path):
    rotation_dir = tmp_path / "histograms" / "layer-3" / "self_attn"
    rotation_dir.mkdir(parents=True)
    rotation_path = rotation_dir / "D.pt"
    torch.save(torch.eye(4), rotation_path)

    assert (
        find_rotation_matrix_path(str(tmp_path), 3, "mlp.gate_up")
        == str(rotation_path)
    )
