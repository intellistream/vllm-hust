# SPDX-License-Identifier: Apache-2.0
"""Evaluate La RoSA PPL alignment between HF reference hooks and vLLM.

The HF hook path mirrors Alibaba EfficientAI La RoSA inference:

* q/k/v and gate/up inputs use ``x @ Q -> top_k_new -> x @ Q.T`` with
  ``sparse_level_h1 = sparsity * 0.8``.
* o/down inputs use direct ``top_k_new`` with
  ``sparse_level_h2 = sparsity * 1.2``.

Calibration artifacts follow the official layout:
``{calibration_path}/histograms/layer-{idx}/self_attn/D.pt``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_teal_ppl_alignment import (  # noqa: E402
    BackendComparison,
    assert_alignment,
    build_eval_ids,
    build_vllm,
    comparison_payload,
    empty_backend_cache,
    iter_windows,
    load_texts,
    load_tokenizer,
    print_comparison,
    score_hf_windows,
    score_vllm_windows,
    summarize_sparse_kernel_markers,
    torch_dtype,
    validate_required_sparse_kernel_markers,
)  # isort: skip

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
SUPPORTED_MODEL_TYPES = {"llama": "Llama2/3", "qwen2": "Qwen2/2.5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate La RoSA dense/sparse PPL deltas for vLLM and an "
            "HF reference implementation aligned to Alibaba EfficientAI."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument(
        "--calibration-path",
        type=Path,
        required=True,
        help=(
            "La RoSA calibration root containing "
            "histograms/layer-{idx}/self_attn/D.pt."
        ),
    )
    parser.add_argument(
        "--larosa-repo-path",
        type=Path,
        default=None,
        help=(
            "Optional Alibaba EfficientAI checkout. Used only with "
            "--hf-reference-impl official_repo."
        ),
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--dataset-name", default="wikitext")
    dataset.add_argument("--dataset-subset", default="wikitext-2-raw-v1")
    dataset.add_argument("--dataset-split", default="test")
    dataset.add_argument("--dataset-text-field", default="text")
    dataset.add_argument("--dataset-size", type=int, default=250)
    dataset.add_argument(
        "--dataset-sample",
        choices=["first", "random"],
        default="random",
        help="Select the first N rows or a seeded random subset for evaluation.",
    )
    dataset.add_argument("--dataset-seed", type=int, default=0)
    dataset.add_argument("--text-file", type=Path, default=None)
    dataset.add_argument("--inline-text", action="append", default=None)

    ppl = parser.add_argument_group("ppl windowing")
    ppl.add_argument("--context-size", type=int, default=2048)
    ppl.add_argument("--window-size", type=int, default=512)
    ppl.add_argument("--max-windows", type=int, default=None)

    backends = parser.add_argument_group("backends")
    backends.add_argument(
        "--backend",
        choices=["both", "hf", "vllm"],
        default="both",
    )
    backends.add_argument(
        "--hf-reference-impl",
        choices=["hooks", "official_repo"],
        default="hooks",
        help=(
            "hooks uses local hooks that mirror the official La RoSA math; "
            "official_repo imports Alibaba EfficientAI modeling_*_larosa.py."
        ),
    )
    backends.add_argument(
        "--vllm-use-sparse-gemv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the backend sparse GEMV kernel in the vLLM sparse run. "
            "This should be used only when the target plugin provides the "
            "activation_sparse_linear custom op."
        ),
    )
    backends.add_argument(
        "--target-projection",
        action="append",
        choices=["self_attn.qkv", "self_attn.o", "mlp.gate_up", "mlp.down"],
        default=None,
        help=(
            "Restrict sparse HF hooks and vLLM activation sparsity to this "
            "projection. Can be repeated."
        ),
    )
    backends.add_argument(
        "--vllm-allow-sparse-gemv-fallback",
        action="store_true",
        help=(
            "Allow dense masked fallback when --vllm-use-sparse-gemv is set "
            "but the backend kernel is unavailable. Leave unset for "
            "kernel-backed evidence runs."
        ),
    )
    backends.add_argument(
        "--vllm-sparse-gemv-dense-fallback-policy",
        choices=["mask", "identity"],
        default="mask",
        help=(
            "Policy for projections rejected by the sparse GEMV kernel policy. "
            "'mask' preserves La RoSA dense fallback semantics; 'identity' "
            "leaves all-row rejected projections dense for kernel-only PPL "
            "probes."
        ),
    )
    backends.add_argument(
        "--vllm-sparse-gemv-min-sparsity",
        type=float,
        default=None,
        help=(
            "Minimum expected sparsity for auto sparse GEMV dispatch. If unset, "
            "the runtime default is used."
        ),
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    runtime.add_argument(
        "--rotation-dtype",
        choices=["float32"],
        default="float32",
        help="Compute dtype for La RoSA rotation/unrotation in the HF hook path.",
    )
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument(
        "--hf-attn-implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "HF attention implementation. The hook path defaults to eager; "
            "official La RoSA scripts commonly use flash_attention_2."
        ),
    )
    runtime.add_argument("--vllm-max-model-len", type=int, default=None)
    runtime.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.75)
    runtime.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    runtime.add_argument("--vllm-shutdown-timeout", type=int, default=30)

    alignment = parser.add_argument_group("alignment")
    alignment.add_argument("--delta-nll-atol", type=float, default=2e-3)
    alignment.add_argument("--delta-nll-rtol", type=float, default=2e-2)
    alignment.add_argument("--fail-on-mismatch", action="store_true")
    alignment.add_argument("--json-output", type=Path, default=None)

    return parser.parse_args()


def get_model_config(args: argparse.Namespace) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(args.model, trust_remote_code=True)


def resolve_model_type(config: Any) -> str:
    model_type = getattr(config, "model_type", None)
    if model_type in SUPPORTED_MODEL_TYPES:
        return model_type
    raise ValueError(
        "La RoSA official support in this vLLM path is limited to "
        f"{', '.join(SUPPORTED_MODEL_TYPES.values())}; got model_type={model_type!r}."
    )


def rotation_path(calibration_path: Path, layer_idx: int) -> Path:
    return calibration_path / "histograms" / f"layer-{layer_idx}" / "self_attn" / "D.pt"


def validate_calibration_artifacts(
    calibration_path: Path,
    num_layers: int,
) -> None:
    missing = [
        str(rotation_path(calibration_path, layer_idx))
        for layer_idx in range(num_layers)
        if not rotation_path(calibration_path, layer_idx).exists()
    ]
    if missing:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(
            "Missing official La RoSA rotation artifacts. Expected one "
            "D.pt per layer under "
            f"{calibration_path}/histograms/layer-*/self_attn/.\n"
            f"Missing examples:\n{preview}"
        )


def official_topk_new(x: torch.Tensor, sparsity_level: float) -> torch.Tensor:
    if sparsity_level <= 0.0:
        return x
    if sparsity_level >= 1.0:
        raise ValueError(
            "La RoSA sparsity_level must be lower than 1.0, "
            f"got {sparsity_level}."
        )
    keep = int((1.0 - sparsity_level) * x.shape[-1])
    if keep < 1:
        raise ValueError(
            "La RoSA top-k would keep fewer than one activation: "
            f"hidden_size={x.shape[-1]}, sparsity_level={sparsity_level}."
        )
    topk_values, _ = torch.topk(torch.abs(x), keep, dim=-1)
    mask = torch.ge(torch.abs(x), topk_values[..., -1:])
    return mask * x


class HFLarosaReference:
    """Forward-hook HF reference for official La RoSA runtime sparsity."""

    def __init__(
        self,
        model: torch.nn.Module,
        calibration_path: Path,
        rotation_dtype: torch.dtype,
        target_projections: list[str] | None,
    ) -> None:
        self.model = model
        self.calibration_path = calibration_path
        self.rotation_dtype = rotation_dtype
        self.target_projections = target_projections
        self.sparsity = 0.0
        self.rotations: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        self._install_hooks()

    def _targets(self, proj_name: str) -> bool:
        return self.target_projections is None or proj_name in self.target_projections

    def set_sparsity(self, sparsity: float) -> None:
        self.sparsity = float(sparsity)

    @property
    def sparse_level_h1(self) -> float:
        return self.sparsity * 0.8

    @property
    def sparse_level_h2(self) -> float:
        return self.sparsity * 1.2

    def _install_hooks(self) -> None:
        layers = self.model.model.layers
        for layer_idx, layer in enumerate(layers):
            if self._targets("self_attn.qkv") or self._targets("mlp.gate_up"):
                q = torch.load(
                    rotation_path(self.calibration_path, layer_idx),
                    map_location="cpu",
                    weights_only=True,
                )
                self.rotations[layer_idx] = q.to(dtype=self.rotation_dtype)

            if self._targets("self_attn.qkv"):
                for module in (
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ):
                    self.handles.append(
                        module.register_forward_pre_hook(
                            self._make_h1_hook(layer_idx),
                        )
                    )

            if self._targets("mlp.gate_up"):
                for module in (layer.mlp.gate_proj, layer.mlp.up_proj):
                    self.handles.append(
                        module.register_forward_pre_hook(
                            self._make_h1_hook(layer_idx),
                        )
                    )

            if self._targets("self_attn.o"):
                self.handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        self._make_h2_hook()
                    )
                )
            if self._targets("mlp.down"):
                self.handles.append(
                    layer.mlp.down_proj.register_forward_pre_hook(
                        self._make_h2_hook()
                    )
                )

    def _make_h1_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            del module
            if self.sparse_level_h1 <= 0.0:
                return inputs
            x = inputs[0]
            q = self.rotations[layer_idx].to(device=x.device)
            rot_x = torch.matmul(x.to(dtype=q.dtype), q)
            sparse_rot_x = official_topk_new(rot_x, self.sparse_level_h1)
            topk_x = torch.matmul(sparse_rot_x, q.t()).to(dtype=x.dtype)
            return (topk_x, *inputs[1:])

        return hook

    def _make_h2_hook(self):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            del module
            if self.sparse_level_h2 <= 0.0:
                return inputs
            x = inputs[0]
            return (official_topk_new(x, self.sparse_level_h2), *inputs[1:])

        return hook


def load_hf_model(args: argparse.Namespace) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
        attn_implementation=args.hf_attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(load_tokenizer(args.model))
    if model_vocab_size != tokenizer_vocab_size:
        model.resize_token_embeddings(tokenizer_vocab_size)
    return model


def build_official_larosa_model(
    args: argparse.Namespace,
    model_type: str,
) -> torch.nn.Module:
    if args.larosa_repo_path is None:
        raise ValueError(
            "--hf-reference-impl official_repo requires --larosa-repo-path."
        )

    larosa_root = args.larosa_repo_path.resolve()
    sys.path.insert(0, str(larosa_root))

    from transformers import AutoConfig

    if model_type == "llama":
        from inference.modeling_llama_larosa import (  # type: ignore[import-not-found]
            LlamaForCausalLM,
        )

        model_cls = LlamaForCausalLM
    elif model_type == "qwen2":
        from inference.modeling_qwen2_larosa import (  # type: ignore[import-not-found]
            Qwen2ForCausalLM,
        )

        model_cls = Qwen2ForCausalLM
    else:
        raise ValueError(f"Unsupported La RoSA model_type={model_type!r}.")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.use_cache = False
    config._attn_implementation = args.hf_attn_implementation
    config.torch_dtype = args.dtype
    config.sparse_level = args.sparsity
    config.Q_path = str(args.calibration_path.resolve())

    model = model_cls.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        config=config,
        trust_remote_code=True,
    )
    model.to(args.device)
    return model


def set_official_model_sparsity(model: torch.nn.Module, sparsity: float) -> None:
    for layer in model.model.layers:
        layer.self_attn.sparse_level_h1 = sparsity * 0.8
        layer.self_attn.sparse_level_h2 = sparsity * 1.2
        layer.mlp.sparse_level_h1 = sparsity * 0.8
        layer.mlp.sparse_level_h2 = sparsity * 1.2


def run_hf_reference(
    args: argparse.Namespace,
    windows: list[list[int]],
    model_type: str,
) -> BackendComparison:
    if args.hf_reference_impl == "official_repo":
        if args.target_projection:
            raise ValueError(
                "--target-projection is supported only with "
                "--hf-reference-impl hooks."
            )
        model = build_official_larosa_model(args, model_type)
        set_official_model_sparsity(model, 0.0)
        reference_name = "official_repo"
    else:
        model = load_hf_model(args)
        larosa_ref = HFLarosaReference(
            model,
            args.calibration_path,
            torch_dtype(args.rotation_dtype),
            args.target_projection,
        )
        larosa_ref.set_sparsity(0.0)
        reference_name = "hooks"

    dense = score_hf_windows(
        model,
        windows,
        args.window_size,
        args.device,
        desc=f"HF dense ({reference_name})",
    )

    if args.hf_reference_impl == "official_repo":
        set_official_model_sparsity(model, args.sparsity)
    else:
        larosa_ref.set_sparsity(args.sparsity)

    sparse = score_hf_windows(
        model,
        windows,
        args.window_size,
        args.device,
        desc=f"HF La RoSA sparse s={args.sparsity}",
    )

    del model
    gc.collect()
    empty_backend_cache()

    return BackendComparison(
        backend=f"hf_larosa:{reference_name}:all_tokens",
        dense=dense,
        sparse=sparse,
    )


def run_vllm_reference(
    args: argparse.Namespace,
    windows: list[list[int]],
) -> BackendComparison:
    dense_llm = build_vllm(args, None)
    dense = score_vllm_windows(
        dense_llm,
        windows,
        args.window_size,
        desc="vLLM dense",
    )
    del dense_llm
    gc.collect()
    empty_backend_cache()

    sparsity_config = {
        "enable": True,
        "method": "larosa",
        "uniform_sparsity": args.sparsity,
        "calibration_path": str(args.calibration_path.resolve()),
        "decode_only": False,
        "apply_all_tokens": True,
        "prefill_sparsify": "all",
        "strict_unsupported_check": True,
        "use_sparse_gemv": args.vllm_use_sparse_gemv,
    }
    if args.target_projection:
        sparsity_config["target_projections"] = args.target_projection
    previous_require_kernel = os.environ.get("VLLM_SPARSE_GEMV_REQUIRE_KERNEL")
    previous_sparse_marker_path = os.environ.get("VLLM_SPARSE_GEMV_MARKER_PATH")
    previous_ascend_marker_path = os.environ.get(
        "VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"
    )
    previous_dense_fallback_policy = os.environ.get(
        "VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY"
    )
    previous_min_sparsity = os.environ.get("VLLM_SPARSE_GEMV_MIN_SPARSITY")
    marker_tmpdir: tempfile.TemporaryDirectory[str] | None = None
    marker_summary: dict[str, Any] | None = None
    active_min_sparsity: str | None = None
    if args.vllm_use_sparse_gemv:
        os.environ["VLLM_SPARSE_GEMV_REQUIRE_KERNEL"] = (
            "0" if args.vllm_allow_sparse_gemv_fallback else "1"
        )
        os.environ["VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY"] = (
            args.vllm_sparse_gemv_dense_fallback_policy
        )
        if args.vllm_sparse_gemv_min_sparsity is not None:
            os.environ["VLLM_SPARSE_GEMV_MIN_SPARSITY"] = str(
                args.vllm_sparse_gemv_min_sparsity
            )
        active_min_sparsity = os.environ.get(
            "VLLM_SPARSE_GEMV_MIN_SPARSITY",
            "0.70",
        )
        marker_tmpdir = tempfile.TemporaryDirectory(prefix="vllm_ppl_sparse_gemv_")
        sparse_marker_path = Path(marker_tmpdir.name) / "invocations.jsonl"
        ascend_marker_path = Path(marker_tmpdir.name) / "ascend_custom_ops.jsonl"
        os.environ["VLLM_SPARSE_GEMV_MARKER_PATH"] = str(sparse_marker_path)
        os.environ["VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"] = str(ascend_marker_path)
    try:
        sparse_llm = build_vllm(args, sparsity_config)
        sparse = score_vllm_windows(
            sparse_llm,
            windows,
            args.window_size,
            desc=f"vLLM La RoSA sparse s={args.sparsity}",
        )
        del sparse_llm
        gc.collect()
        empty_backend_cache()
        if marker_tmpdir is not None:
            marker_summary = summarize_sparse_kernel_markers(
                sparse_marker_path,
                ascend_marker_path,
            )
            if not args.vllm_allow_sparse_gemv_fallback:
                validate_required_sparse_kernel_markers(marker_summary)
    finally:
        if previous_require_kernel is None:
            os.environ.pop("VLLM_SPARSE_GEMV_REQUIRE_KERNEL", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_REQUIRE_KERNEL"] = (
                previous_require_kernel
            )
        if previous_sparse_marker_path is None:
            os.environ.pop("VLLM_SPARSE_GEMV_MARKER_PATH", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_MARKER_PATH"] = previous_sparse_marker_path
        if previous_ascend_marker_path is None:
            os.environ.pop("VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH", None)
        else:
            os.environ["VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"] = (
                previous_ascend_marker_path
            )
        if previous_dense_fallback_policy is None:
            os.environ.pop("VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY"] = (
                previous_dense_fallback_policy
            )
        if previous_min_sparsity is None:
            os.environ.pop("VLLM_SPARSE_GEMV_MIN_SPARSITY", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_MIN_SPARSITY"] = previous_min_sparsity
        if marker_tmpdir is not None:
            marker_tmpdir.cleanup()

    return BackendComparison(
        backend=(
            "vllm_larosa:official_topk:all_tokens:"
            f"use_sparse_gemv={args.vllm_use_sparse_gemv}:"
            "sparse_gemv_dense_fallback="
            f"{args.vllm_sparse_gemv_dense_fallback_policy}:"
            f"sparse_gemv_min_sparsity={active_min_sparsity}"
        ),
        dense=dense,
        sparse=sparse,
        sparse_kernel_markers=marker_summary,
    )


def main() -> int:
    args = parse_args()
    args.calibration_path = args.calibration_path.resolve()

    if args.sparsity < 0.0 or args.sparsity >= 1.0:
        raise ValueError("--sparsity must be in [0, 1).")
    if args.sparsity * 1.2 >= 1.0:
        raise ValueError(
            "Official La RoSA h2 sparsity is sparsity * 1.2 and must be < 1. "
            f"Got sparsity={args.sparsity}."
        )
    if (
        args.vllm_sparse_gemv_min_sparsity is not None
        and not 0.0 <= args.vllm_sparse_gemv_min_sparsity <= 1.0
    ):
        raise ValueError("--vllm-sparse-gemv-min-sparsity must be in [0, 1]")

    config = get_model_config(args)
    model_type = resolve_model_type(config)
    num_layers = int(config.num_hidden_layers)
    validate_calibration_artifacts(args.calibration_path, num_layers)

    tokenizer = load_tokenizer(args.model)
    texts = load_texts(args)
    token_ids = build_eval_ids(tokenizer, texts, args.window_size)
    windows = iter_windows(
        token_ids,
        args.context_size,
        args.window_size,
        args.max_windows,
    )

    print(
        "setup: "
        f"model={args.model}, model_type={model_type}, "
        f"sparsity={args.sparsity}, windows={len(windows)}, "
        f"tokens={len(token_ids)}, calibration_path={args.calibration_path}"
    )

    hf_result = None
    vllm_result = None
    if args.backend in {"both", "hf"}:
        hf_result = run_hf_reference(args, windows, model_type)
        print_comparison(hf_result)
    if args.backend in {"both", "vllm"}:
        vllm_result = run_vllm_reference(args, windows)
        print_comparison(vllm_result)

    aligned = assert_alignment(hf_result, vllm_result, args)

    if args.json_output:
        payload = {
            "setup": {
                "model": args.model,
                "model_type": model_type,
                "official_supported_family": SUPPORTED_MODEL_TYPES[model_type],
                "sparsity": args.sparsity,
                "context_size": args.context_size,
                "window_size": args.window_size,
                "tokens": len(token_ids),
                "windows": len(windows),
                "hf_reference_impl": args.hf_reference_impl,
                "rotation_dtype": args.rotation_dtype,
                "vllm_sparse_gemv_dense_fallback_policy": (
                    args.vllm_sparse_gemv_dense_fallback_policy
                ),
                "vllm_sparse_gemv_min_sparsity": (
                    args.vllm_sparse_gemv_min_sparsity
                ),
                "calibration_path": str(args.calibration_path),
            },
            "hf": comparison_payload(hf_result),
            "vllm": comparison_payload(vllm_result),
            "aligned": aligned,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")

    if not aligned and args.fail_on_mismatch:
        return 2
    if (
        hf_result is not None
        and vllm_result is not None
        and math.copysign(1.0, hf_result.delta_nll)
        != math.copysign(1.0, vllm_result.delta_nll)
    ):
        print(
            "warning: HF and vLLM PPL deltas have different signs; "
            "inspect calibration artifacts and model/runtime parity."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
