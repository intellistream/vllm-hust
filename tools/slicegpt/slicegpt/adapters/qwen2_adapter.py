# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# https://www.apache.org/licenses/LICENSE-2.0

import inspect

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2RMSNorm,
)

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


def _supports_qwen2_model(model_name: str) -> bool:
    return (
        "Qwen2" in model_name
        or "Qwen2.5" in model_name
        or "qwen2" in model_name.lower()
    )


class CompressedQwen2DecoderLayer(Qwen2DecoderLayer):
    """Qwen2 decoder layer with SliceGPT residual rotations."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: tuple[Tensor] | None = None,
        use_cache: bool | None = False,
        cache_position: LongTensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_params = inspect.signature(self.self_attn.forward).parameters
        attn_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }
        if "past_key_values" in attn_params:
            attn_kwargs["past_key_values"] = past_key_values
        elif "past_key_value" in attn_params:
            attn_kwargs["past_key_value"] = past_key_values
        if "cache_position" in attn_params:
            attn_kwargs["cache_position"] = cache_position
        if "position_embeddings" in attn_params:
            attn_kwargs["position_embeddings"] = position_embeddings
        for key, value in kwargs.items():
            if key in attn_params:
                attn_kwargs[key] = value

        hidden_states = self.self_attn(**attn_kwargs)[0]
        if self.attn_shortcut_Q is not None:
            hidden_states = matmul(residual, self.attn_shortcut_Q) + hidden_states
        else:
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_shortcut_Q is not None:
            hidden_states = matmul(residual, self.mlp_shortcut_Q) + hidden_states
        else:
            hidden_states = residual + hidden_states

        return hidden_states


class Qwen2LayerAdapter(LayerAdapter):
    def __init__(self, layer: Qwen2DecoderLayer) -> None:
        super().__init__()
        self._layer: Qwen2DecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [
            self.layer.self_attn.q_proj,
            self.layer.self_attn.k_proj,
            self.layer.self_attn.v_proj,
        ]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class Qwen2ModelAdapter(ModelAdapter):
    def __init__(self, model: Qwen2ForCausalLM) -> None:
        super().__init__()
        self._model: Qwen2ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return Qwen2Config

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return Qwen2DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return Qwen2RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Qwen2LayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedQwen2DecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(
        self, layer: Module, layer_idx: int | None
    ) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(
            self.config.torch_dtype
        )
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not _supports_qwen2_model(model_name):
            return None

        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
        )
        model.config.torch_dtype = dtype
        return Qwen2ModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not _supports_qwen2_model(model_name):
            return None

        class UninitializedQwen2ForCausalLM(Qwen2ForCausalLM):
            def _init_weights(self, _) -> None:
                pass

        config = Qwen2Config.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
        )
        model = UninitializedQwen2ForCausalLM(config)
        model = model.to(dtype=dtype)
        return Qwen2ModelAdapter(model)
