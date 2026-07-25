# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only SliceGPT-compressed LLaMA model.

SliceGPT (https://github.com/microsoft/TransformerCompression) compresses a Llama
model by reducing the *residual-stream* (embedding) dimension. Compared to a stock
Llama this model has three differences:

  1. The residual width is reduced (e.g. 4096 -> 3072) and may differ per layer.
     All per-layer dims are read from `config.slicing_config` (embedded in config.json
     by tools/convert_slicegpt_to_vllm.py). Attention inner dims (num_heads*head_dim)
     and MLP intermediate_size are UNCHANGED.
  2. Before each residual add, the residual is rotated by a dense matrix:
        hidden = residual @ attn_shortcut_Q + attn_out
        hidden = residual @ mlp_shortcut_Q  + mlp_out
  3. The RMSNorm layers have NO affine weight (folded into the linears). CRUCIALLY,
     SliceGPT's RMSN normalizes by the *original* (pre-slice) hidden size, not the
     reduced per-layer width: `var = x.pow(2).sum(-1) / orig_hidden` even when the
     input is only `attn_in` (e.g. 3072) wide. vLLM's stock RMSNorm divides by the
     actual width, which would scale every activation by sqrt(reduced/orig) ~= 0.866
     and corrupt perplexity. We therefore use a faithful SliceGPTRMSN below with a
     fixed divisor = num_attention_heads * head_dim (the original residual width).
     These norms are called in the *un-fused* single-arg form (the residual add cannot
     be fused because the residual is rotated first).

Only tensor_parallel_size=1 / pipeline_parallel_size=1 is supported.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .llama import LlamaAttention
from .utils import extract_layer_index, make_layers, maybe_prefix


class SliceGPTRMSN(nn.Module):
    """RMSNorm without affine weight, normalizing by a FIXED divisor.

    Faithful reimplementation of SliceGPT's `slicegpt.modules.RMSN`: the variance is
    `x.pow(2).sum(-1) / mean_dim` where `mean_dim` is the ORIGINAL (pre-slice) hidden
    size, regardless of the actual (possibly reduced) width of `x`. Computed in float32
    like the reference, then cast back. There is no learnable weight.
    """

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


class SliceGPTMLP(nn.Module):
    """Llama gated MLP allowing different input/output residual dims."""

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


class SliceGPTDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: LlamaConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        sc = config.slicing_config
        idx = extract_layer_index(prefix)

        attn_in = sc["attention_input_dims"][idx]
        attn_out = sc["attention_output_dims"][idx]
        mlp_in = sc["mlp_input_dims"][idx]
        mlp_out = sc["mlp_output_dims"][idx]
        # We reuse vLLM's LlamaAttention, whose o_proj output is forced to equal its
        # hidden_size input. SliceGPT (sequential) keeps attn_in == attn_out,
        # so this holds.
        assert attn_in == attn_out, (
            f"layer {idx}: attn_in({attn_in}) != attn_out({attn_out}); "
            "LlamaAttention reuse requires equal attention in/out dims"
        )

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=attn_in,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
            quant_config=None,
            bias=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = SliceGPTMLP(
            input_size=mlp_in,
            output_size=mlp_out,
            intermediate_size=config.intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        # SliceGPT normalizes by the ORIGINAL hidden size, not the sliced width.
        orig_hidden = config.num_attention_heads * sc["attn_head_dim"]
        self.input_layernorm = SliceGPTRMSN(orig_hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SliceGPTRMSN(
            orig_hidden, eps=config.rms_norm_eps
        )

        # Dense residual rotations, used as `residual @ Q`.
        self.attn_shortcut_Q = nn.Parameter(torch.empty(attn_in, attn_out))
        self.mlp_shortcut_Q = nn.Parameter(torch.empty(mlp_in, mlp_out))

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Attention block (un-fused residual: rotate residual, then add).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = torch.matmul(residual, self.attn_shortcut_Q) + hidden_states

        # MLP block.
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
class SliceGPTLlamaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: LlamaConfig = vllm_config.model_config.hf_config
        sc = config.slicing_config
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, sc["embedding_dim"])
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SliceGPTDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # Final norm also divides by the original hidden size (final_norm_dim happens
        # to equal it when do_slice_head=false, but be explicit for correctness).
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


class SliceGPTLlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "SliceGPTLlamaForCausalLM does not support pipeline parallelism. "
                "Use pipeline_parallel_size=1."
            )
        config: LlamaConfig = vllm_config.model_config.hf_config
        self.config = config
        sc = config.slicing_config

        self.model = SliceGPTLlamaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            sc["final_norm_dim"],
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError(
                "SliceGPTLlamaForCausalLM does not support pipeline "
                "intermediate_tensors. Use pipeline_parallel_size=1."
            )
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
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
