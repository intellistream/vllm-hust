# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Convert a SliceGPT-sliced Qwen2/Qwen2.5 model (.pt + .json) into a vLLM-loadable
checkpoint directory for the custom `SliceGPTQwen2ForCausalLM` architecture.
"""

import argparse
import json
import pathlib
import shutil

import torch
from conversion_utils import (
    expected_state_names,
    infer_attention_head_dim,
    normalize_slicing_conf,
    validate_shape,
    validate_state_keys,
)
from safetensors.torch import save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sliced-dir",
        required=True,
        help="Dir with the .pt and .json produced by run_slicegpt.py",
    )
    p.add_argument("--pt-name", required=True, help="Filename of the sliced .pt")
    p.add_argument(
        "--base-model-path", required=True, help="Original HF Qwen2/Qwen2.5 dir"
    )
    p.add_argument(
        "--out-dir", required=True, help="Output dir for the vLLM checkpoint"
    )
    return p.parse_args()


def main():
    args = parse_args()
    sliced_dir = pathlib.Path(args.sliced_dir)
    base_path = pathlib.Path(args.base_model_path)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = sliced_dir / args.pt_name
    json_path = pt_path.with_suffix(".json")
    print(f"Loading state_dict: {pt_path}")
    state = torch.load(pt_path, map_location="cpu")
    raw_conf = json.loads(json_path.read_text())

    base_config = json.loads((base_path / "config.json").read_text())
    num_layers = int(base_config["num_hidden_layers"])
    num_heads = int(base_config["num_attention_heads"])
    num_kv_heads = int(base_config.get("num_key_value_heads", num_heads))
    intermediate = int(base_config["intermediate_size"])
    vocab = int(base_config["vocab_size"])

    sc = normalize_slicing_conf(raw_conf, num_layers)
    validate_state_keys(state, expected_state_names(num_layers, attention_bias=True))
    q0 = state["model.layers.0.self_attn.q_proj.weight"]
    attn_head_dim = infer_attention_head_dim(q0, num_heads)
    sc["attn_head_dim"] = attn_head_dim
    print(
        "slicing_config: "
        f"embedding_dim={sc['embedding_dim']} "
        f"final_norm_dim={sc['final_norm_dim']} "
        f"attn_head_dim={attn_head_dim}"
    )

    out_tensors: dict[str, torch.Tensor] = {}
    q_out = num_heads * attn_head_dim
    kv_out = num_kv_heads * attn_head_dim

    def put(name, t, expected_shape: tuple[int, ...]):
        validate_shape(name, t, expected_shape)
        out_tensors[name] = t.to(torch.float16).contiguous()

    put(
        "model.embed_tokens.weight",
        state["model.embed_tokens.weight"],
        (vocab, sc["embedding_dim"]),
    )
    put("lm_head.weight", state["lm_head.weight"], (vocab, sc["final_norm_dim"]))

    for i in range(num_layers):
        ai = sc["attention_input_dims"][i]
        ao = sc["attention_output_dims"][i]
        mi = sc["mlp_input_dims"][i]
        mo = sc["mlp_output_dims"][i]
        pre = f"model.layers.{i}"

        put(
            f"{pre}.self_attn.q_proj.weight",
            state[f"{pre}.self_attn.q_proj.weight"],
            (q_out, ai),
        )
        put(
            f"{pre}.self_attn.k_proj.weight",
            state[f"{pre}.self_attn.k_proj.weight"],
            (kv_out, ai),
        )
        put(
            f"{pre}.self_attn.v_proj.weight",
            state[f"{pre}.self_attn.v_proj.weight"],
            (kv_out, ai),
        )
        put(
            f"{pre}.self_attn.q_proj.bias",
            state[f"{pre}.self_attn.q_proj.bias"],
            (q_out,),
        )
        put(
            f"{pre}.self_attn.k_proj.bias",
            state[f"{pre}.self_attn.k_proj.bias"],
            (kv_out,),
        )
        put(
            f"{pre}.self_attn.v_proj.bias",
            state[f"{pre}.self_attn.v_proj.bias"],
            (kv_out,),
        )
        put(
            f"{pre}.self_attn.o_proj.weight",
            state[f"{pre}.self_attn.o_proj.weight"],
            (ao, q_out),
        )

        put(
            f"{pre}.mlp.gate_proj.weight",
            state[f"{pre}.mlp.gate_proj.weight"],
            (intermediate, mi),
        )
        put(
            f"{pre}.mlp.up_proj.weight",
            state[f"{pre}.mlp.up_proj.weight"],
            (intermediate, mi),
        )
        put(
            f"{pre}.mlp.down_proj.weight",
            state[f"{pre}.mlp.down_proj.weight"],
            (mo, intermediate),
        )

        put(f"{pre}.attn_shortcut_Q", state[f"{pre}.attn_shortcut_Q"], (ai, ao))
        put(f"{pre}.mlp_shortcut_Q", state[f"{pre}.mlp_shortcut_Q"], (mi, mo))

    print(f"Validated and collected {len(out_tensors)} tensors")

    new_config = dict(base_config)
    new_config["architectures"] = ["SliceGPTQwen2ForCausalLM"]
    new_config["model_type"] = "qwen2"
    new_config["hidden_size"] = sc["embedding_dim"]
    new_config["head_dim"] = attn_head_dim
    new_config["tie_word_embeddings"] = False
    new_config["torch_dtype"] = "float16"
    new_config["slicing_config"] = sc

    (out_dir / "config.json").write_text(json.dumps(new_config, indent=2))
    print(f"Wrote {out_dir / 'config.json'}")

    save_file(
        out_tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"}
    )
    print(f"Wrote {out_dir / 'model.safetensors'}")

    for pat in [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
    ]:
        src = base_path / pat
        if src.exists():
            shutil.copy(str(src), out_dir)
    print("Copied tokenizer files. Done.")


if __name__ == "__main__":
    main()
