# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader


class _TestEmbedding(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape), requires_grad=False)

        def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            param.copy_(loaded_weight)

        self.weight.weight_loader = weight_loader


class _TestModel(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = _TestEmbedding(shape)


class _RepeatedLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = _TestEmbedding((4, 3))
        self.model.layers = nn.ModuleList(
            [nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=False)]
        )


def test_load_real_embedding_over_dummy_weights(tmp_path):
    weight_name = "model.embed_tokens.weight"
    expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    shard_path = tmp_path / "model.safetensors"
    save_file({weight_name: expected}, shard_path)

    loader = DummyModelLoader(
        LoadConfig(
            load_format="dummy",
            model_loader_extra_config={"embedding_weight_path": str(shard_path)},
        )
    )
    model = _TestModel(expected.shape)

    loader._load_embedding_weight(model)

    torch.testing.assert_close(model.model.embed_tokens.weight, expected)


def test_dummy_loader_rejects_unknown_extra_config():
    with pytest.raises(ValueError, match="Unsupported dummy model loader"):
        DummyModelLoader(
            LoadConfig(
                load_format="dummy",
                model_loader_extra_config={"unsupported": True},
            )
        )


def test_embedding_overlay_skips_non_owning_pipeline_rank(tmp_path):
    missing_path = tmp_path / "not-downloaded.safetensors"
    loader = DummyModelLoader(
        LoadConfig(
            load_format="dummy",
            model_loader_extra_config={
                "embedding_weight_path": str(missing_path),
            },
        )
    )

    loader._load_embedding_weight(nn.Module())


def test_share_repeated_dummy_layer_weights():
    loader = DummyModelLoader(
        LoadConfig(
            load_format="dummy",
            model_loader_extra_config={"share_dummy_weights": True},
        )
    )
    model = _RepeatedLayerModel()
    first = model.model.layers[0].weight
    second = model.model.layers[1].weight
    with torch.no_grad():
        first.fill_(1)
        second.fill_(2)

    loader._share_repeated_layer_weights(model)

    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second, torch.ones_like(second))
    assert model.model.embed_tokens.weight.data_ptr() != first.data_ptr()


def test_dummy_loader_rejects_non_boolean_weight_sharing():
    with pytest.raises(ValueError, match="share_dummy_weights must be a boolean"):
        DummyModelLoader(
            LoadConfig(
                load_format="dummy",
                model_loader_extra_config={"share_dummy_weights": "yes"},
            )
        )
