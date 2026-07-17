# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only SliceGPT-compressed Qwen2/Qwen2.5 model."""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, EncoderOnlyAttention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsPP
from .utils import extract_layer_index, make_layers, maybe_prefix


class SliceGPTRMSN(nn.Module):
    def __init__(self, mean_dim: int, eps: float) -> None:
        super().__init__()
        self.mean_dim = mean_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(orig_dtype)


class SliceGPTQwen2MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        intermediate_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=input_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=output_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class SliceGPTQwen2Attention(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        input_size: int,
        output_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        cache_config,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=input_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=output_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=True,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class SliceGPTQwen2DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen2Config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        sc = config.slicing_config
        idx = extract_layer_index(prefix)

        set_default_rope_theta(config, default_theta=1000000)

        attn_in = sc["attention_input_dims"][idx]
        attn_out = sc["attention_output_dims"][idx]
        mlp_in = sc["mlp_input_dims"][idx]
        mlp_out = sc["mlp_output_dims"][idx]

        self.self_attn = SliceGPTQwen2Attention(
            config=config,
            input_size=attn_in,
            output_size=attn_out,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = SliceGPTQwen2MLP(
            input_size=mlp_in,
            output_size=mlp_out,
            intermediate_size=config.intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        orig_hidden = config.num_attention_heads * sc["attn_head_dim"]
        self.input_layernorm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)
        self.attn_shortcut_Q = nn.Parameter(torch.empty(attn_in, attn_out))
        self.mlp_shortcut_Q = nn.Parameter(torch.empty(mlp_in, mlp_out))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = torch.matmul(residual, self.attn_shortcut_Q) + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.matmul(residual, self.mlp_shortcut_Q) + hidden_states
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class SliceGPTQwen2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen2Config = vllm_config.model_config.hf_config.get_text_config()
        sc = config.slicing_config
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, sc["embedding_dim"])
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SliceGPTQwen2DecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        orig_hidden = config.num_attention_heads * sc["attn_head_dim"]
        self.norm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SliceGPTQwen2ForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen2Config = vllm_config.model_config.hf_config.get_text_config()
        sc = config.slicing_config
        self.config = config

        self.model = SliceGPTQwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            sc["final_norm_dim"],
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = lambda *a, **k: IntermediateTensors({})

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj.weight", ".q_proj.weight", "q"),
            (".qkv_proj.weight", ".k_proj.weight", "k"),
            (".qkv_proj.weight", ".v_proj.weight", "v"),
            (".qkv_proj.bias", ".q_proj.bias", "q"),
            (".qkv_proj.bias", ".k_proj.bias", "k"),
            (".qkv_proj.bias", ".v_proj.bias", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, w, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, w)
                loaded.add(name)
        return loaded
