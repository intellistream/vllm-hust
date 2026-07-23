# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Dry-run shape validator for the SliceGPT -> vLLM conversion.

Loads the sliced .pt state_dict + .json SlicingConfig, derives the per-layer
slicing_config exactly like tools/convert_slicegpt_to_vllm.py, and asserts every
tensor's shape against those dims -- printing each check. Writes NOTHING to disk
and does NOT import safetensors / touch the GPU.
"""

import argparse
import json
import pathlib

import torch


def normalize_slicing_conf(raw: dict, num_layers: int) -> dict:
    def as_list(d, n):
        out = [None] * n
        for k, v in d.items():
            out[int(k)] = int(v)
        assert all(x is not None for x in out), f"missing layer dims in {d}"
        return out

    emb = raw["embedding_dimensions"]
    embedding_dim = int(emb[next(iter(emb))]) if isinstance(emb, dict) else int(emb[0])
    return {
        "embedding_dim": embedding_dim,
        "final_norm_dim": int(raw["head_dimension"]),
        "attention_input_dims": as_list(raw["attention_input_dimensions"], num_layers),
        "attention_output_dims": as_list(
            raw["attention_output_dimensions"], num_layers
        ),
        "mlp_input_dims": as_list(raw["mlp_input_dimensions"], num_layers),
        "mlp_output_dims": as_list(raw["mlp_output_dimensions"], num_layers),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sliced-dir", required=True)
    p.add_argument("--pt-name", required=True)
    p.add_argument("--base-model-path", required=True)
    args = p.parse_args()

    sliced_dir = pathlib.Path(args.sliced_dir)
    base_path = pathlib.Path(args.base_model_path)
    pt_path = sliced_dir / args.pt_name
    json_path = pt_path.with_suffix(".json")

    print(f"[load] state_dict: {pt_path}")
    state = torch.load(pt_path, map_location="cpu")
    raw_conf = json.loads(json_path.read_text())
    base_config = json.loads((base_path / "config.json").read_text())

    num_layers = int(base_config["num_hidden_layers"])
    num_heads = int(base_config["num_attention_heads"])
    num_kv_heads = int(base_config.get("num_key_value_heads", num_heads))
    intermediate = int(base_config["intermediate_size"])
    vocab = int(base_config["vocab_size"])

    sc = normalize_slicing_conf(raw_conf, num_layers)
    q0 = state["model.layers.0.self_attn.q_proj.weight"]
    attn_head_dim = q0.shape[0] // num_heads
    sc["attn_head_dim"] = attn_head_dim
    qkv_out = num_heads * attn_head_dim
    kv_out = num_kv_heads * attn_head_dim

    print(
        f"[conf] embedding_dim={sc['embedding_dim']} final_norm_dim={sc['final_norm_dim']} "
        f"attn_head_dim={attn_head_dim} qkv_out={qkv_out} kv_out={kv_out}"
    )
    print(
        f"[conf] attn_in[0]={sc['attention_input_dims'][0]} attn_in[-1]={sc['attention_input_dims'][-1]} "
        f"mlp_in[-1]={sc['mlp_input_dims'][-1]} mlp_out[-1]={sc['mlp_output_dims'][-1]}"
    )

    # residual-stream continuity: mlp_out[i] must feed attn_in[i+1]; last feeds final norm/head
    print("[continuity] checking residual-stream dim continuity across layers ...")
    for i in range(num_layers - 1):
        assert sc["mlp_output_dims"][i] == sc["attention_input_dims"][i + 1], (
            f"layer {i} mlp_out({sc['mlp_output_dims'][i]}) != "
            f"layer {i + 1} attn_in({sc['attention_input_dims'][i + 1]})"
        )
    assert sc["embedding_dim"] == sc["attention_input_dims"][0], (
        f"embedding_dim({sc['embedding_dim']}) != attn_in[0]({sc['attention_input_dims'][0]})"
    )
    assert sc["mlp_output_dims"][-1] == sc["final_norm_dim"], (
        f"mlp_out[-1]({sc['mlp_output_dims'][-1]}) != final_norm_dim({sc['final_norm_dim']})"
    )
    print("[continuity] OK")

    # model code also asserts attn_in == attn_out (LlamaAttention reuse)
    for i in range(num_layers):
        assert sc["attention_input_dims"][i] == sc["attention_output_dims"][i], (
            f"layer {i}: attn_in != attn_out (model.py reuse of LlamaAttention forbids this)"
        )
    print("[attn-square] attn_in==attn_out for all layers: OK")

    errors = []
    checked = 0

    def check(name, expected):
        nonlocal checked
        if name not in state:
            errors.append(f"MISSING tensor: {name}")
            return
        got = tuple(state[name].shape)
        ok = got == tuple(expected)
        checked += 1
        if not ok:
            errors.append(f"{name}: shape {got} != expected {tuple(expected)}")

    check("model.embed_tokens.weight", (vocab, sc["embedding_dim"]))
    check("lm_head.weight", (vocab, sc["final_norm_dim"]))
    for i in range(num_layers):
        ai = sc["attention_input_dims"][i]
        ao = sc["attention_output_dims"][i]
        mi = sc["mlp_input_dims"][i]
        mo = sc["mlp_output_dims"][i]
        pre = f"model.layers.{i}"
        check(f"{pre}.self_attn.q_proj.weight", (qkv_out, ai))
        check(f"{pre}.self_attn.k_proj.weight", (kv_out, ai))
        check(f"{pre}.self_attn.v_proj.weight", (kv_out, ai))
        check(f"{pre}.self_attn.o_proj.weight", (ao, qkv_out))
        check(f"{pre}.mlp.gate_proj.weight", (intermediate, mi))
        check(f"{pre}.mlp.up_proj.weight", (intermediate, mi))
        check(f"{pre}.mlp.down_proj.weight", (mo, intermediate))
        check(f"{pre}.attn_shortcut_Q", (ai, ao))
        check(f"{pre}.mlp_shortcut_Q", (mi, mo))

    # surface any unexpected leftover tensors (besides droppable norms / rotary buffers)
    expected_names = {"model.embed_tokens.weight", "lm_head.weight"}
    for i in range(num_layers):
        pre = f"model.layers.{i}"
        for suf in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "attn_shortcut_Q",
            "mlp_shortcut_Q",
        ]:
            expected_names.add(f"{pre}.{suf}")

    def droppable(n):
        return (
            "rotary_emb" in n
            or n.endswith(".inv_freq")
            or n.endswith("input_layernorm.weight")
            or n.endswith("post_attention_layernorm.weight")
            or n == "model.norm.weight"
        )

    leftovers = [
        n
        for n in state
        if n not in expected_names and not droppable(n) and torch.is_tensor(state[n])
    ]
    dropped = [n for n in state if droppable(n)]

    print(f"[shapes] checked {checked} tensors, {len(errors)} error(s)")
    print(
        f"[drop] {len(dropped)} norm/rotary tensors will be dropped (sample): {dropped[:4]}"
    )
    if leftovers:
        print(
            f"[WARN] {len(leftovers)} UNEXPECTED leftover tensor(s) not handled by converter:"
        )
        for n in leftovers[:20]:
            print(f"        {n}  {tuple(state[n].shape)}")
    else:
        print("[leftover] none — every non-droppable tensor is accounted for")

    if errors:
        print("\n=== SHAPE ERRORS ===")
        for e in errors[:40]:
            print("  " + e)
        print(f"\nDRYRUN FAILED: {len(errors)} problem(s)")
        raise SystemExit(1)
    print("\nDRYRUN OK: all shapes match slicing_config; conversion is safe to run.")


if __name__ == "__main__":
    main()
