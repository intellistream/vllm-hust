# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _class_node(path: str, class_name: str) -> ast.ClassDef:
    module = ast.parse((ROOT / path).read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _method_node(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _keyword_names(method: ast.FunctionDef) -> set[str]:
    return {
        keyword.arg
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg is not None
    }


def test_mistral_attention_accepts_and_forwards_sparsity_config():
    class_node = _class_node(
        "vllm/model_executor/models/mistral.py", "MistralAttention"
    )
    init = _method_node(class_node, "__init__")
    forward = _method_node(class_node, "forward")

    assert "sparsity_config" in {argument.arg for argument in init.args.args}
    assert "sparsity_config" in _keyword_names(init)
    forward_names = {
        node.id for node in ast.walk(forward) if isinstance(node, ast.Name)
    }
    assert {"sparse_qkv", "sparse_o"} <= forward_names


def test_qwen3_layers_accept_and_forward_sparsity_config():
    attention = _class_node("vllm/model_executor/models/qwen3.py", "Qwen3Attention")
    decoder = _class_node("vllm/model_executor/models/qwen3.py", "Qwen3DecoderLayer")
    attention_init = _method_node(attention, "__init__")
    decoder_init = _method_node(decoder, "__init__")

    assert "sparsity_config" in {argument.arg for argument in attention_init.args.args}
    assert "sparsity_config" in {argument.arg for argument in decoder_init.args.args}
    assert "sparsity_config" in _keyword_names(decoder_init)

    forward = _method_node(attention, "forward")
    forward_names = {
        node.id for node in ast.walk(forward) if isinstance(node, ast.Name)
    }
    assert {"sparse_qkv", "sparse_o"} <= forward_names


def test_vllm_config_requires_explicit_sparsity_configuration():
    source = (ROOT / "vllm/config/vllm.py").read_text()

    assert "get_activation_sparsity_config" not in source
    assert "validate_activation_sparsity_compatibility(" in source
