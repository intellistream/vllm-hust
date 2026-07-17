# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import re
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    get_layerwise_info,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    initialize_dummy_weights,
    initialize_single_dummy_weight,
)

logger = init_logger(__name__)

_LAYER_INDEX_RE = re.compile(r"(?<=\.layers\.)\d+(?=\.)")


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if extra_config is None:
            extra_config = {}
        if not isinstance(extra_config, dict):
            raise ValueError("Dummy model loader extra config must be a dictionary")

        supported_keys = {
            "embedding_weight_path",
            "embedding_weight_name",
            "share_dummy_weights",
        }
        unsupported_keys = extra_config.keys() - supported_keys
        if unsupported_keys:
            raise ValueError(
                "Unsupported dummy model loader extra config keys: "
                f"{sorted(unsupported_keys)}"
            )

        embedding_weight_path = extra_config.get("embedding_weight_path")
        if embedding_weight_path is not None and not isinstance(
            embedding_weight_path, str
        ):
            raise ValueError("embedding_weight_path must be a string")
        self.embedding_weight_path = (
            Path(embedding_weight_path) if embedding_weight_path else None
        )

        embedding_weight_name = extra_config.get(
            "embedding_weight_name", "model.embed_tokens.weight"
        )
        if not isinstance(embedding_weight_name, str) or not embedding_weight_name:
            raise ValueError("embedding_weight_name must be a non-empty string")
        self.embedding_weight_name = embedding_weight_name

        share_dummy_weights = extra_config.get("share_dummy_weights", False)
        if not isinstance(share_dummy_weights, bool):
            raise ValueError("share_dummy_weights must be a boolean")
        self.share_dummy_weights = share_dummy_weights

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        model = super().load_model(vllm_config, model_config, prefix)
        if self.share_dummy_weights:
            self._share_repeated_layer_weights(model)
        return model

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                self._process_online_quant_layer(layer, info)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                initialize_dummy_weights(layer, model_config)

        if self.embedding_weight_path is not None:
            self._load_embedding_weight(model)

    @torch.no_grad()
    def _share_repeated_layer_weights(self, model: nn.Module) -> None:
        """Deduplicate processed dummy weights for corresponding model layers."""
        canonical: dict[tuple, torch.Tensor] = {}
        shared_bytes = 0
        accelerator_parameter_found = False

        for name, param in model.named_parameters(remove_duplicate=False):
            normalized_name = _LAYER_INDEX_RE.sub("*", name)
            if normalized_name == name or name == self.embedding_weight_name:
                continue

            key = (
                normalized_name,
                param.dtype,
                param.device,
                tuple(param.shape),
                tuple(param.stride()),
            )
            if key not in canonical:
                canonical[key] = param.data
                continue

            param.data = canonical[key]
            shared_bytes += param.numel() * param.element_size()
            accelerator_parameter_found |= param.device.type != "cpu"

        if accelerator_parameter_found and torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        logger.info(
            "Shared %.2f GiB of repeated dummy layer parameters",
            shared_bytes / 2**30,
        )

    @torch.no_grad()
    def _load_embedding_weight(self, model: nn.Module) -> None:
        param = dict(model.named_parameters(remove_duplicate=False)).get(
            self.embedding_weight_name
        )
        if param is None:
            # Pipeline ranks that do not own the input embedding skip the shard.
            return

        assert self.embedding_weight_path is not None
        if not self.embedding_weight_path.is_file():
            raise FileNotFoundError(
                f"Embedding weight shard not found: {self.embedding_weight_path}"
            )

        weight_loader = getattr(param, "weight_loader", None)
        if not callable(weight_loader):
            raise TypeError(
                f"Parameter {self.embedding_weight_name!r} has no weight loader"
            )

        with safe_open(
            self.embedding_weight_path, framework="pt", device="cpu"
        ) as weight_file:
            available_weights = weight_file.keys()
            if self.embedding_weight_name not in available_weights:
                raise KeyError(
                    f"{self.embedding_weight_name!r} is not present in "
                    f"{self.embedding_weight_path}"
                )
            loaded_weight = weight_file.get_tensor(self.embedding_weight_name)
            weight_loader(param, loaded_weight)
        logger.info(
            "Loaded real embedding weight %s from %s over dummy model weights",
            self.embedding_weight_name,
            self.embedding_weight_path,
        )

    def _process_online_quant_layer(
        self,
        layer: nn.Module,
        info: LayerReloadingInfo,
    ) -> None:
        """Materialize, apply dummy weights, and run quantization processing."""
        materialize_layer(layer, info)

        for tensor in get_layer_tensors(layer).values():
            initialize_single_dummy_weight(tensor)

        for param in get_layer_tensors(layer).values():
            param.weight_loader = _get_original_loader(param)

        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)

        info.reset()
