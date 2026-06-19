# SPDX-License-Identifier: Apache-2.0
"""Compare TEAL PPL deltas between vLLM and an HF reference path.

The default HF reference mirrors the public FasterDecoding/TEAL implementation:
thresholds are computed with ``icdf(0.5 + sparsity / 2)`` from per-layer
histograms, then applied before q/k/v/o and gate/up/down projections. Passing
``--hf-reference-impl official_repo`` instead loads the model class from a local
FasterDecoding/TEAL checkout.

Three HF reference modes are provided:
* ``official_half``: match TEAL's HF SparsifyFn default, which sparsifies only
  the last half of a prefill sequence.
* ``all_tokens``: sparsify every token, matching the legacy vLLM Phase 0 path.
* ``decode_only``: leave prefill dense and sparsify single-token decode only.

The vLLM sparse path defaults to ``prefill_sparsify="half"`` so its PPL delta
can be checked against ``official_half`` directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

DEFAULT_MODEL = "meta-llama/Llama-2-7b-hf"
DEFAULT_ARTIFACT_ROOT = Path(".cache/teal_validation/llama2_7b_s0.50")
SUPPORTED_PROJECTIONS = (
    "self_attn.qkv",
    "self_attn.o",
    "mlp.gate_up",
    "mlp.down",
)


@dataclass
class PPLStats:
    nll: float
    ppl: float
    token_weighted_nll: float
    token_weighted_ppl: float
    windows: int
    tokens: int


@dataclass
class BackendComparison:
    backend: str
    dense: PPLStats
    sparse: PPLStats
    sparse_kernel_markers: dict[str, Any] | None = None

    @property
    def delta_nll(self) -> float:
        return self.sparse.nll - self.dense.nll

    @property
    def delta_ppl(self) -> float:
        return self.sparse.ppl - self.dense.ppl

    @property
    def ppl_ratio(self) -> float:
        return self.sparse.ppl / self.dense.ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TEAL dense/sparse PPL deltas for vLLM and an "
            "HF reference implementation."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Directory containing manifest/config plus official_histograms.",
    )
    parser.add_argument(
        "--hf-histogram-path",
        type=Path,
        default=None,
        help=(
            "Official TEAL histogram root, e.g. "
            "models/Llama-2-7B/histograms. Defaults to "
            "<artifact-root>/official_histograms."
        ),
    )
    parser.add_argument(
        "--vllm-calibration-path",
        type=Path,
        default=None,
        help=(
            "vLLM threshold root. Defaults to <artifact-root>/thresholds or "
            "activation_sparsity_config.json calibration_path."
        ),
    )
    parser.add_argument(
        "--teal-repo-path",
        type=Path,
        default=None,
        help=(
            "Optional FasterDecoding/TEAL checkout. Used only for a threshold "
            "spot-check against the official Distribution class."
        ),
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "HF dataset name. For a paper-style sampled subset, use tatsu-lab/alpaca."
        ),
    )
    dataset.add_argument("--dataset-subset", default=None)
    dataset.add_argument("--dataset-split", default="train")
    dataset.add_argument("--dataset-text-field", default="text")
    dataset.add_argument("--dataset-size", type=int, default=250)
    dataset.add_argument(
        "--dataset-sample",
        choices=["first", "random"],
        default="first",
        help="Select the first N rows or a seeded random subset for evaluation.",
    )
    dataset.add_argument("--dataset-seed", type=int, default=0)
    dataset.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Plain-text file; paragraphs are split on blank lines.",
    )
    dataset.add_argument(
        "--inline-text",
        action="append",
        default=None,
        help="Inline sample text. Can be passed multiple times.",
    )

    ppl = parser.add_argument_group("ppl windowing")
    ppl.add_argument("--context-size", type=int, default=2048)
    ppl.add_argument("--window-size", type=int, default=512)
    ppl.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=(
            "Limit windows for sampled subset checks. Full eval should leave "
            "this unset."
        ),
    )

    backends = parser.add_argument_group("backends")
    backends.add_argument(
        "--backend",
        choices=["both", "hf", "vllm"],
        default="both",
        help="Which backend(s) to evaluate.",
    )
    backends.add_argument(
        "--hf-reference-mode",
        choices=["official_half", "all_tokens", "decode_only"],
        default="official_half",
        help=(
            "official_half matches FasterDecoding/TEAL HF prefill behavior; "
            "all_tokens matches the legacy all-token sparsity path."
        ),
    )
    backends.add_argument(
        "--hf-reference-impl",
        choices=["hooks", "official_repo"],
        default="hooks",
        help=(
            "hooks uses local forward pre-hooks that mirror TEAL; "
            "official_repo imports the FasterDecoding/TEAL sparse model class "
            "from --teal-repo-path."
        ),
    )
    backends.add_argument(
        "--vllm-prefill-sparsify",
        choices=["half", "all", "none"],
        default="half",
        help=(
            "vLLM prefill policy: half matches FasterDecoding/TEAL, all "
            "sparsifies every token, none sparsifies decode tokens only."
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
            "'mask' preserves TEAL dense fallback semantics; 'identity' leaves "
            "all-row rejected projections dense for kernel-only PPL probes."
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
    backends.add_argument(
        "--hf-attn-implementation",
        default="sdpa",
        help="HF attention implementation for the local reference model.",
    )
    backends.add_argument("--device", default="cuda")
    backends.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    backends.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    backends.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Defaults to context_size + window_size + 1.",
    )
    backends.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    backends.add_argument(
        "--vllm-shutdown-timeout",
        type=int,
        default=5,
    )

    compare = parser.add_argument_group("comparison")
    compare.add_argument("--delta-nll-atol", type=float, default=0.02)
    compare.add_argument("--delta-nll-rtol", type=float, default=0.05)
    compare.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when vLLM delta NLL is outside tolerance.",
    )
    compare.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable result JSON.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def empty_backend_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    npu = getattr(torch, "npu", None)
    if npu is None:
        return
    try:
        if npu.is_available():
            npu.empty_cache()
    except RuntimeError:
        pass


def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )
    return tokenizer


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.inline_text:
        return args.inline_text

    if args.text_file:
        text = args.text_file.read_text()
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if p]

    if args.dataset_name:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required for --dataset-name. Install it with "
                "`uv pip install --python .venv/bin/python datasets`."
            ) from exc
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_subset,
            split=args.dataset_split,
        )
        if args.dataset_size is not None:
            size = min(args.dataset_size, len(dataset))
            if getattr(args, "dataset_sample", "first") == "random":
                dataset = dataset.shuffle(
                    seed=getattr(args, "dataset_seed", 0)
                ).select(range(size))
            else:
                dataset = dataset.select(range(size))
        return [str(sample[args.dataset_text_field]) for sample in dataset]

    return [
        "TEAL applies magnitude-based activation sparsity to transformer "
        "hidden states without retraining.",
        "Perplexity alignment is checked by scoring the same target tokens "
        "under dense and sparse execution paths.",
        "The vLLM path should match the HF reference once threshold mapping "
        "and prefill sparsification semantics are the same.",
    ]


def build_eval_ids(
    tokenizer: Any,
    texts: Iterable[str],
    window_size: int,
) -> list[int]:
    text = "\n\n".join(texts)
    encodings = tokenizer(text, return_tensors="pt")
    token_ids = encodings.input_ids[0].tolist()
    seq_len = len(token_ids) - (len(token_ids) % window_size)
    if seq_len < 2:
        raise ValueError(
            f"Need at least 2 eval tokens after window truncation, got {seq_len}."
        )
    return token_ids[:seq_len]


def iter_windows(
    token_ids: list[int],
    context_size: int,
    window_size: int,
    max_windows: int | None = None,
) -> list[list[int]]:
    max_length = context_size + window_size
    windows: list[list[int]] = []
    for begin_loc in range(0, len(token_ids), window_size):
        end_loc = begin_loc + max_length
        window = token_ids[begin_loc:end_loc]
        if len(window) >= 2:
            windows.append(window)
        if max_windows is not None and len(windows) >= max_windows:
            break
        if end_loc >= len(token_ids):
            break
    if not windows:
        raise ValueError("No PPL windows were produced.")
    return windows


def target_start(window_len: int, window_size: int) -> int:
    return max(1, window_len - window_size)


def official_icdf_from_histogram(
    histogram_path: Path,
    hidden_type: str,
    sparsity: float,
) -> torch.Tensor:
    histogram = torch.load(histogram_path, map_location="cpu", weights_only=False)
    centers = histogram[f"{hidden_type}_centers"].float()
    counts = histogram[hidden_type].float()
    total_count = counts.sum()
    cumulative_counts = torch.cumsum(counts, dim=0)
    target_count = (0.5 + sparsity / 2.0) * total_count
    idx = torch.searchsorted(cumulative_counts, target_count)

    if idx == 0:
        value = centers[0]
    elif idx == len(centers):
        value = centers[-1]
    else:
        lower_count = cumulative_counts[idx - 1]
        upper_count = cumulative_counts[idx]
        lower_value = centers[idx - 1]
        upper_value = centers[idx]
        denom = upper_count - lower_count
        if denom.abs() < 1e-12:
            value = lower_value
        else:
            fraction = (target_count - lower_count) / denom
            value = lower_value + fraction * (upper_value - lower_value)
    return value.detach().cpu().to(torch.float32).reshape(())


class TealMask(torch.nn.Module):
    def __init__(self, reference_mode: str) -> None:
        super().__init__()
        self.reference_mode = reference_mode
        self.register_buffer("threshold", torch.tensor(0.0, dtype=torch.float32))

    def set_threshold(self, threshold: torch.Tensor | float) -> None:
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, dtype=torch.float32)
        self.threshold = threshold.detach().cpu().to(torch.float32).reshape(())

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        threshold = self.threshold.to(device=x.device, dtype=x.dtype)
        return x.abs().gt(threshold) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reference_mode == "all_tokens":
            return self._apply_mask(x)

        if x.dim() < 3:
            if self.reference_mode == "decode_only":
                return self._apply_mask(x) if x.size(0) == 1 else x
            half_seq_len = x.size(0) // 2
            return torch.cat((x[:-half_seq_len], self._apply_mask(x[-half_seq_len:])))

        seq_len = x.size(1)
        if self.reference_mode == "decode_only":
            return self._apply_mask(x) if seq_len == 1 else x

        if seq_len > 1:
            half_seq_len = seq_len // 2
            return torch.cat(
                (x[:, :-half_seq_len, :], self._apply_mask(x[:, -half_seq_len:, :])),
                dim=1,
            )
        return self._apply_mask(x)


class HFTealReference:
    def __init__(
        self,
        model: torch.nn.Module,
        histogram_path: Path,
        reference_mode: str,
    ) -> None:
        self.model = model
        self.histogram_path = histogram_path
        self.reference_mode = reference_mode
        self.masks: dict[str, TealMask] = {}
        self._install_hooks()

    def _register_mask(
        self,
        module: torch.nn.Module,
        name: str,
        mask: TealMask,
    ) -> None:
        self.masks[name] = mask

        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs:
                return inputs
            return (mask(inputs[0]), *inputs[1:])

        handle = module.register_forward_pre_hook(hook)
        hooks = getattr(self.model, "_teal_alignment_hooks", [])
        hooks.append(handle)
        self.model._teal_alignment_hooks = hooks

    def _install_hooks(self) -> None:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("HF reference currently expects model.model.layers.")

        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            mlp = layer.mlp

            attn_h1 = TealMask(self.reference_mode)
            attn_h2 = TealMask(self.reference_mode)
            mlp_h1 = TealMask(self.reference_mode)
            mlp_h2 = TealMask(self.reference_mode)

            for proj_name in ("q_proj", "k_proj", "v_proj"):
                self._register_mask(
                    getattr(attn, proj_name),
                    f"layers.{layer_idx}.self_attn.{proj_name}",
                    attn_h1,
                )
            self._register_mask(
                attn.o_proj,
                f"layers.{layer_idx}.self_attn.o_proj",
                attn_h2,
            )
            for proj_name in ("gate_proj", "up_proj"):
                self._register_mask(
                    getattr(mlp, proj_name),
                    f"layers.{layer_idx}.mlp.{proj_name}",
                    mlp_h1,
                )
            self._register_mask(
                mlp.down_proj,
                f"layers.{layer_idx}.mlp.down_proj",
                mlp_h2,
            )

    def set_sparsity(self, sparsity: float) -> None:
        layers = self.model.model.layers
        for layer_idx, _layer in enumerate(layers):
            mlp_path = self.histogram_path / f"layer-{layer_idx}" / "mlp"
            attn_path = self.histogram_path / f"layer-{layer_idx}" / "self_attn"
            if sparsity == 0:
                mlp_h1 = mlp_h2 = attn_h1 = attn_h2 = torch.tensor(0.0)
            else:
                mlp_h1 = official_icdf_from_histogram(
                    mlp_path / "histograms.pt", "h1", sparsity
                )
                mlp_h2 = official_icdf_from_histogram(
                    mlp_path / "histograms.pt", "h2", sparsity
                )
                attn_h1 = official_icdf_from_histogram(
                    attn_path / "histograms.pt", "h1", sparsity
                )
                attn_h2 = official_icdf_from_histogram(
                    attn_path / "histograms.pt", "h2", sparsity
                )

            for name, mask in self.masks.items():
                prefix = f"layers.{layer_idx}."
                if not name.startswith(prefix):
                    continue
                if (
                    ".self_attn.q_proj" in name
                    or ".self_attn.k_proj" in name
                    or ".self_attn.v_proj" in name
                ):
                    mask.set_threshold(attn_h1)
                elif ".self_attn.o_proj" in name:
                    mask.set_threshold(attn_h2)
                elif ".mlp.gate_proj" in name or ".mlp.up_proj" in name:
                    mask.set_threshold(mlp_h1)
                elif ".mlp.down_proj" in name:
                    mask.set_threshold(mlp_h2)


def build_official_teal_model(
    args: argparse.Namespace,
    histogram_path: Path,
) -> torch.nn.Module:
    if args.teal_repo_path is None:
        raise ValueError("--hf-reference-impl official_repo requires --teal-repo-path.")
    if args.hf_reference_mode == "all_tokens":
        raise ValueError(
            "The official TEAL HF backend supports half-prefill "
            "(apply_prefill=True) and decode-only (apply_prefill=False), "
            "but not the synthetic all_tokens reference mode."
        )

    teal_repo_path = args.teal_repo_path.resolve()
    sys.path.insert(0, str(teal_repo_path))
    try:
        from teal.model import (  # type: ignore[import-not-found]
            LlamaSparseConfig,
            LlamaSparseForCausalLM,
            MistralSparseConfig,
            MistralSparseForCausalLM,
        )
    finally:
        sys.path.pop(0)

    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if base_config.model_type == "llama":
        sparse_config = LlamaSparseConfig.from_dict(base_config.to_dict())
        model_cls = LlamaSparseForCausalLM
    elif base_config.model_type == "mistral":
        sparse_config = MistralSparseConfig.from_dict(base_config.to_dict())
        model_cls = MistralSparseForCausalLM
    else:
        raise ValueError(
            "Official TEAL HF backend currently supports llama and mistral "
            f"configs, got model_type={base_config.model_type!r}."
        )

    model = model_cls.from_pretrained(
        args.model,
        config=sparse_config,
        histogram_path=str(histogram_path),
        uniform_sparsity=0.0,
        apply_prefill=args.hf_reference_mode == "official_half",
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
    )
    patch_official_teal_transformers_compat(model)
    model.to(args.device)
    return model


def patch_official_teal_transformers_compat(model: torch.nn.Module) -> None:
    """Fill attributes expected by the TEAL monkeypatch on newer HF Llama."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    config = getattr(model, "config", None)
    if layers is None or config is None:
        return

    rotary_emb = getattr(getattr(model, "model", None), "rotary_emb", None)
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if not hasattr(attn, "num_heads") and hasattr(config, "num_attention_heads"):
            attn.num_heads = config.num_attention_heads
        if not hasattr(attn, "num_key_value_heads") and hasattr(
            config, "num_key_value_heads"
        ):
            attn.num_key_value_heads = config.num_key_value_heads
        if not hasattr(attn, "hidden_size") and hasattr(config, "hidden_size"):
            attn.hidden_size = config.hidden_size
        if not hasattr(attn, "rotary_emb") and rotary_emb is not None:
            attn.rotary_emb = rotary_emb


def validate_artifacts(
    histogram_path: Path,
    calibration_path: Path,
    num_layers: int,
) -> None:
    missing: list[str] = []
    for layer_idx in range(num_layers):
        for rel in (
            f"layer-{layer_idx}/mlp/histograms.pt",
            f"layer-{layer_idx}/self_attn/histograms.pt",
        ):
            if not (histogram_path / rel).exists():
                missing.append(str(histogram_path / rel))
        for proj in SUPPORTED_PROJECTIONS:
            filename = f"layers.{layer_idx}.{proj}.threshold.pt"
            if not (calibration_path / filename).exists():
                missing.append(str(calibration_path / filename))
    if missing:
        raise FileNotFoundError(
            "Missing TEAL calibration artifacts:\n" + "\n".join(missing[:20])
        )


def spot_check_official_distribution(
    teal_repo_path: Path | None,
    histogram_path: Path,
    sparsity: float,
) -> None:
    if teal_repo_path is None:
        return

    sys.path.insert(0, str(teal_repo_path.resolve()))
    try:
        from utils.utils import Distribution
    finally:
        sys.path.pop(0)

    for layer_idx, part, hidden_type in (
        (0, "mlp", "h1"),
        (0, "self_attn", "h2"),
    ):
        path = histogram_path / f"layer-{layer_idx}" / part
        official = Distribution(str(path), hidden_type)
        official_value = float(official.icdf(0.5 + sparsity / 2.0))
        local_value = float(
            official_icdf_from_histogram(path / "histograms.pt", hidden_type, sparsity)
        )
        if not math.isclose(official_value, local_value, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError(
                "Threshold computation mismatch against official TEAL "
                f"Distribution for {path} {hidden_type}: "
                f"{local_value} != {official_value}"
            )


def score_hf_windows(
    model: torch.nn.Module,
    windows: list[list[int]],
    window_size: int,
    device: str,
    desc: str,
) -> PPLStats:
    model.eval()
    nlls: list[float] = []
    weighted_nll_sum = 0.0
    token_count = 0
    for window in tqdm(windows, desc=desc):
        input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        labels = input_ids.clone()
        labels[:, :-window_size] = -100
        start = target_start(input_ids.size(1), window_size)
        count = input_ids.size(1) - start
        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss.float()
        nll = float(loss.item())
        nlls.append(nll)
        weighted_nll_sum += nll * count
        token_count += count

    mean_nll = sum(nlls) / len(nlls)
    token_weighted_nll = weighted_nll_sum / token_count
    return PPLStats(
        nll=mean_nll,
        ppl=math.exp(mean_nll),
        token_weighted_nll=token_weighted_nll,
        token_weighted_ppl=math.exp(token_weighted_nll),
        windows=len(nlls),
        tokens=token_count,
    )


def run_hf_reference(
    args: argparse.Namespace,
    windows: list[list[int]],
    histogram_path: Path,
) -> BackendComparison:
    if args.hf_reference_impl == "official_repo":
        model = build_official_teal_model(args, histogram_path)
        model.set_uniform_sparsity(0.0)
    else:
        from transformers import AutoModelForCausalLM

        dtype = torch_dtype(args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=args.hf_attn_implementation,
            low_cpu_mem_usage=True,
        )
        model.to(args.device)

        model_vocab_size = model.get_input_embeddings().weight.size(0)
        tokenizer = load_tokenizer(args.model)
        tokenizer_vocab_size = len(tokenizer)
        if model_vocab_size != tokenizer_vocab_size:
            model.resize_token_embeddings(tokenizer_vocab_size)

        teal_ref = HFTealReference(
            model=model,
            histogram_path=histogram_path,
            reference_mode=args.hf_reference_mode,
        )
        teal_ref.set_sparsity(0.0)

    dense = score_hf_windows(
        model,
        windows,
        args.window_size,
        args.device,
        desc=f"HF dense ({args.hf_reference_mode})",
    )
    if args.hf_reference_impl == "official_repo":
        model.set_uniform_sparsity(args.sparsity)
    else:
        teal_ref.set_sparsity(args.sparsity)
    sparse = score_hf_windows(
        model,
        windows,
        args.window_size,
        args.device,
        desc=f"HF sparse s={args.sparsity}",
    )

    del model
    gc.collect()
    empty_backend_cache()
    return BackendComparison(
        backend=f"hf_reference:{args.hf_reference_impl}:{args.hf_reference_mode}",
        dense=dense,
        sparse=sparse,
    )


def prompt_logprob(logprobs: Any, position: int, token_id: int) -> float:
    entry = logprobs[position]
    if entry is None:
        raise ValueError(f"No prompt logprob at position {position}.")
    item = entry.get(token_id)
    if item is None:
        raise KeyError(
            f"Token id {token_id} missing from prompt_logprobs at {position}."
        )
    return float(item.logprob)


def score_vllm_windows(
    llm: Any,
    windows: list[list[int]],
    window_size: int,
    desc: str,
) -> PPLStats:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
    )
    nlls: list[float] = []
    weighted_nll_sum = 0.0
    token_count = 0
    for window in tqdm(windows, desc=desc):
        outputs = llm.generate(
            [{"prompt_token_ids": window}],
            sampling_params,
            use_tqdm=False,
        )
        prompt_logprobs = outputs[0].prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt_logprobs.")
        start = target_start(len(window), window_size)
        nll = 0.0
        count = 0
        for position in range(start, len(window)):
            nll -= prompt_logprob(prompt_logprobs, position, window[position])
            count += 1
        window_nll = nll / count
        nlls.append(window_nll)
        weighted_nll_sum += nll
        token_count += count

    mean_nll = sum(nlls) / len(nlls)
    token_weighted_nll = weighted_nll_sum / token_count
    return PPLStats(
        nll=mean_nll,
        ppl=math.exp(mean_nll),
        token_weighted_nll=token_weighted_nll,
        token_weighted_ppl=math.exp(token_weighted_nll),
        windows=len(nlls),
        tokens=token_count,
    )


def build_vllm(
    args: argparse.Namespace,
    activation_sparsity_config: dict[str, Any] | None,
) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.vllm_max_model_len
        or (args.context_size + args.window_size + 1),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enforce_eager": args.vllm_enforce_eager,
        "shutdown_timeout": args.vllm_shutdown_timeout,
    }
    if activation_sparsity_config is not None:
        kwargs["activation_sparsity_config"] = activation_sparsity_config
    return LLM(**kwargs)


def read_marker_records(marker_path: Path) -> list[dict[str, Any]]:
    if not marker_path.exists():
        return []
    records = []
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        records.append(json.loads(line))
    return records


def unique_marker_values(
    records: list[dict[str, Any]],
    key: str,
) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for record in records:
        if key not in record:
            continue
        value = record[key]
        marker = json.dumps(value, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        values.append(value)
    return values


def summarize_sparse_kernel_markers(
    sparse_marker_path: Path,
    ascend_marker_path: Path,
) -> dict[str, Any]:
    sparse_records = read_marker_records(sparse_marker_path)
    ascend_records = read_marker_records(ascend_marker_path)
    summary: dict[str, Any] = {
        "sparse_gemv_marker_records": len(sparse_records),
        "sparse_gemv_marker_pids": sorted(
            {
                int(record["pid"])
                for record in sparse_records
                if "pid" in record
            }
        ),
        "sparse_gemv_marker_x_shapes": unique_marker_values(
            sparse_records,
            "x_shape",
        ),
        "sparse_gemv_marker_threshold_shapes": unique_marker_values(
            sparse_records,
            "threshold_shape",
        ),
        "sparse_gemv_marker_threshold_numels": unique_marker_values(
            sparse_records,
            "threshold_numel",
        ),
        "sparse_gemv_marker_inclusive": unique_marker_values(
            sparse_records,
            "inclusive",
        ),
        "sparse_gemv_marker_weight_t_provided": unique_marker_values(
            sparse_records,
            "weight_t_provided",
        ),
        "ascend_sparse_linear_marker_records": len(ascend_records),
        "ascend_sparse_linear_marker_pids": sorted(
            {
                int(record["pid"])
                for record in ascend_records
                if "pid" in record
            }
        ),
        "ascend_sparse_linear_marker_ops": sorted(
            {
                str(record["op"])
                for record in ascend_records
                if "op" in record
            }
        ),
        "ascend_sparse_linear_marker_x_shapes": unique_marker_values(
            ascend_records,
            "x_shape",
        ),
        "ascend_sparse_linear_marker_threshold_shapes": unique_marker_values(
            ascend_records,
            "threshold_shape",
        ),
        "ascend_sparse_linear_marker_threshold_numels": unique_marker_values(
            ascend_records,
            "threshold_numel",
        ),
        "ascend_sparse_linear_marker_inclusive": unique_marker_values(
            ascend_records,
            "inclusive",
        ),
        "ascend_sparse_linear_marker_weight_t_provided": unique_marker_values(
            ascend_records,
            "weight_t_provided",
        ),
    }
    return summary


def validate_required_sparse_kernel_markers(
    marker_summary: dict[str, Any] | None,
) -> None:
    if marker_summary is None:
        raise RuntimeError("sparse kernel marker summary was not collected.")
    if marker_summary["sparse_gemv_marker_records"] <= 0:
        raise RuntimeError("vLLM sparse GEMV marker records are missing.")
    if marker_summary["ascend_sparse_linear_marker_records"] <= 0:
        raise RuntimeError("Ascend sparse linear marker records are missing.")
    accepted_ops = {
        "activation_sparse_linear",
        "activation_sparse_linear_direct_t",
        "activation_sparse_linear_packed_t",
    }
    if not any(
        op in accepted_ops
        for op in marker_summary["ascend_sparse_linear_marker_ops"]
    ):
        raise RuntimeError(
            "Ascend sparse linear marker did not record a supported sparse "
            "linear custom op."
        )


def run_vllm_reference(
    args: argparse.Namespace,
    windows: list[list[int]],
    calibration_path: Path,
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
        "method": "teal",
        "uniform_sparsity": args.sparsity,
        "calibration_path": str(calibration_path),
        "decode_only": False,
        "apply_all_tokens": False,
        "prefill_sparsify": args.vllm_prefill_sparsify,
        "strict_unsupported_check": True,
        "use_sparse_gemv": args.vllm_use_sparse_gemv,
    }
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
            desc=f"vLLM sparse s={args.sparsity}",
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
            f"vllm:prefill_sparsify={args.vllm_prefill_sparsify}:"
            f"use_sparse_gemv={args.vllm_use_sparse_gemv}:"
            "sparse_gemv_dense_fallback="
            f"{args.vllm_sparse_gemv_dense_fallback_policy}:"
            f"sparse_gemv_min_sparsity={active_min_sparsity}"
        ),
        dense=dense,
        sparse=sparse,
        sparse_kernel_markers=marker_summary,
    )


def resolve_artifact_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    artifact_root = args.artifact_root.resolve()
    histogram_path = (
        args.hf_histogram_path.resolve()
        if args.hf_histogram_path is not None
        else (artifact_root / "official_histograms").resolve()
    )

    if args.vllm_calibration_path is not None:
        calibration_path = args.vllm_calibration_path.resolve()
    else:
        config_path = artifact_root / "activation_sparsity_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            calibration_path = Path(config["calibration_path"]).resolve()
        else:
            calibration_path = (artifact_root / "thresholds").resolve()
    return histogram_path, calibration_path


def print_comparison(result: BackendComparison) -> None:
    print(
        f"{result.backend}: dense_ppl={result.dense.ppl:.6f}, "
        f"sparse_ppl={result.sparse.ppl:.6f}, "
        f"delta_ppl={result.delta_ppl:.6f}, "
        f"delta_nll={result.delta_nll:.6f}, "
        f"ppl_ratio={result.ppl_ratio:.6f}, "
        f"windows={result.dense.windows}, tokens={result.dense.tokens}"
    )


def comparison_payload(result: BackendComparison | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = asdict(result)
    payload.update(
        {
            "delta_nll": result.delta_nll,
            "delta_ppl": result.delta_ppl,
            "ppl_ratio": result.ppl_ratio,
        }
    )
    return payload


def assert_alignment(
    hf_result: BackendComparison | None,
    vllm_result: BackendComparison | None,
    args: argparse.Namespace,
) -> bool:
    if hf_result is None or vllm_result is None:
        return True
    aligned = math.isclose(
        hf_result.delta_nll,
        vllm_result.delta_nll,
        rel_tol=args.delta_nll_rtol,
        abs_tol=args.delta_nll_atol,
    )
    print(
        "alignment: "
        f"hf_delta_nll={hf_result.delta_nll:.6f}, "
        f"vllm_delta_nll={vllm_result.delta_nll:.6f}, "
        f"abs_diff={abs(hf_result.delta_nll - vllm_result.delta_nll):.6f}, "
        f"status={'PASS' if aligned else 'FAIL'}"
    )
    return aligned


def main() -> int:
    args = parse_args()
    if (
        args.vllm_sparse_gemv_min_sparsity is not None
        and not 0.0 <= args.vllm_sparse_gemv_min_sparsity <= 1.0
    ):
        raise ValueError("--vllm-sparse-gemv-min-sparsity must be in [0, 1]")
    histogram_path, calibration_path = resolve_artifact_paths(args)
    tokenizer = load_tokenizer(args.model)
    texts = load_texts(args)
    token_ids = build_eval_ids(tokenizer, texts, args.window_size)
    windows = iter_windows(
        token_ids,
        context_size=args.context_size,
        window_size=args.window_size,
        max_windows=args.max_windows,
    )

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = int(hf_config.num_hidden_layers)
    validate_artifacts(histogram_path, calibration_path, num_layers)
    spot_check_official_distribution(args.teal_repo_path, histogram_path, args.sparsity)

    print(
        "eval setup: "
        f"model={args.model}, sparsity={args.sparsity}, "
        f"tokens={len(token_ids)}, windows={len(windows)}, "
        f"context={args.context_size}, window={args.window_size}, "
        f"hf_impl={args.hf_reference_impl}, "
        f"hf_mode={args.hf_reference_mode}, "
        f"vllm_prefill={args.vllm_prefill_sparsify}"
    )
    print(f"histogram_path={histogram_path}")
    print(f"vllm_calibration_path={calibration_path}")

    hf_result: BackendComparison | None = None
    vllm_result: BackendComparison | None = None

    if args.backend in ("both", "hf"):
        hf_result = run_hf_reference(args, windows, histogram_path)
        print_comparison(hf_result)

    if args.backend in ("both", "vllm"):
        vllm_result = run_vllm_reference(args, windows, calibration_path)
        print_comparison(vllm_result)

    aligned = assert_alignment(hf_result, vllm_result, args)

    if args.json_output:
        payload = {
            "setup": {
                "model": args.model,
                "sparsity": args.sparsity,
                "context_size": args.context_size,
                "window_size": args.window_size,
                "tokens": len(token_ids),
                "windows": len(windows),
                "hf_reference_impl": args.hf_reference_impl,
                "hf_reference_mode": args.hf_reference_mode,
                "vllm_prefill_sparsify": args.vllm_prefill_sparsify,
                "vllm_sparse_gemv_dense_fallback_policy": (
                    args.vllm_sparse_gemv_dense_fallback_policy
                ),
                "vllm_sparse_gemv_min_sparsity": (
                    args.vllm_sparse_gemv_min_sparsity
                ),
                "histogram_path": str(histogram_path),
                "vllm_calibration_path": str(calibration_path),
            },
            "hf": comparison_payload(hf_result),
            "vllm": comparison_payload(vllm_result),
            "aligned": aligned,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")

    if not aligned and args.fail_on_mismatch:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
