# SPDX-License-Identifier: Apache-2.0
"""Capture La RoSA activation statistics and rotations for Qwen-style models.

This script generates artifacts compatible with the official La RoSA layout:

    {output_path}/histograms/layer-{idx}/self_attn/D.pt

It can also write the MLP rotation under ``mlp/D.pt`` for inspection and future
experiments. The current official Qwen2.5 La RoSA inference path reuses the
per-layer ``self_attn/D.pt`` rotation for both attention and MLP first-site
sparsification, so that path remains the required vLLM artifact.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate La RoSA covariance/eigenvector artifacts from sampled "
            "activation windows."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Artifact root; rotations are saved under <root>/histograms/.",
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--dataset-name", default="wikitext")
    dataset.add_argument("--dataset-subset", default="wikitext-2-raw-v1")
    dataset.add_argument("--dataset-split", default="train")
    dataset.add_argument("--dataset-text-field", default="text")
    dataset.add_argument("--dataset-size", type=int, default=500)
    dataset.add_argument(
        "--dataset-sample",
        choices=["first", "random"],
        default="random",
        help="Use the first N rows or a seeded random subset.",
    )
    dataset.add_argument("--dataset-seed", type=int, default=0)
    dataset.add_argument("--text-file", type=Path, default=None)
    dataset.add_argument("--inline-text", action="append", default=None)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--sequence-length", type=int, default=2048)
    runtime.add_argument("--batch-size", type=int, default=1)
    runtime.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional sampled sequence cap. Leave unset for all built windows.",
    )
    runtime.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    runtime.add_argument(
        "--device",
        default="auto",
        help="auto, npu, cuda, cpu, or a concrete torch device like npu:0.",
    )
    runtime.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    runtime.add_argument(
        "--stat-device",
        default="cpu",
        help="Where covariance matrices are accumulated, usually cpu.",
    )
    runtime.add_argument("--damp-percent", type=float, default=0.01)
    runtime.add_argument(
        "--save-covariance",
        action="store_true",
        help="Also save large covariance.pt files for each layer/projection.",
    )
    runtime.add_argument(
        "--capture-mlp",
        action="store_true",
        help="Also capture and eigensolve layer MLP covariance matrices.",
    )

    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.inline_text:
        return args.inline_text

    if args.text_file:
        text = args.text_file.read_text()
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if p]

    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_subset,
        split=args.dataset_split,
    )
    if args.dataset_size is not None:
        size = min(args.dataset_size, len(dataset))
        if args.dataset_sample == "random":
            dataset = dataset.shuffle(seed=args.dataset_seed).select(range(size))
        else:
            dataset = dataset.select(range(size))
    return [str(sample[args.dataset_text_field]) for sample in dataset]


def build_token_windows(
    tokenizer: Any,
    texts: Iterable[str],
    sequence_length: int,
    max_sequences: int | None,
) -> list[list[int]]:
    text = "\n\n".join(t for t in texts if t.strip())
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    num_sequences = len(token_ids) // sequence_length
    if max_sequences is not None:
        num_sequences = min(num_sequences, max_sequences)
    if num_sequences <= 0:
        raise ValueError(
            "No full activation-capture windows were produced. "
            f"Need at least {sequence_length} tokens."
        )
    return [
        token_ids[i * sequence_length : (i + 1) * sequence_length]
        for i in range(num_sequences)
    ]


class CovarianceCollector:
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        stat_device: torch.device,
        capture_mlp: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.capture_mlp = capture_mlp
        self.attn_cov = [
            torch.zeros(
                (hidden_size, hidden_size),
                dtype=torch.float32,
                device=stat_device,
            )
            for _ in range(num_layers)
        ]
        self.mlp_cov = (
            [
                torch.zeros(
                    (hidden_size, hidden_size),
                    dtype=torch.float32,
                    device=stat_device,
                )
                for _ in range(num_layers)
            ]
            if capture_mlp
            else []
        )
        self.attn_tokens = [0 for _ in range(num_layers)]
        self.mlp_tokens = [0 for _ in range(num_layers)] if capture_mlp else []

    def hook(self, layer_idx: int, projection: str):
        def _hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            del module
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(
                device=self.attn_cov[layer_idx].device,
                dtype=torch.float32,
            )
            cov = x.t().matmul(x)
            if projection == "self_attn":
                self.attn_cov[layer_idx].add_(cov)
                self.attn_tokens[layer_idx] += x.shape[0]
            elif projection == "mlp":
                if not self.capture_mlp:
                    return
                self.mlp_cov[layer_idx].add_(cov)
                self.mlp_tokens[layer_idx] += x.shape[0]
            else:
                raise ValueError(f"Unknown projection {projection!r}.")

        return _hook


def save_rotation_from_covariance(
    cov: torch.Tensor,
    output_dir: Path,
    *,
    damp_percent: float,
    token_count: int,
    save_covariance: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cov_cpu = cov.detach().to(device="cpu", dtype=torch.float32)
    diag_mean = torch.mean(torch.diag(cov_cpu))
    damp = damp_percent * diag_mean
    idx = torch.arange(cov_cpu.shape[-1])
    cov_cpu[idx, idx] += damp

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_cpu)
    rotation = eigenvectors.contiguous().to(dtype=torch.float32)
    inverse = rotation.t().contiguous()

    torch.save(rotation, output_dir / "D.pt")
    torch.save(inverse, output_dir / "inv_D.pt")
    torch.save(eigenvalues.to(dtype=torch.float32), output_dir / "eigenvalues.pt")
    if save_covariance:
        torch.save(cov_cpu, output_dir / "covariance.pt")

    return {
        "tokens": token_count,
        "hidden_size": cov_cpu.shape[-1],
        "damp_percent": damp_percent,
        "damp": float(damp.item()),
        "diag_mean": float(diag_mean.item()),
        "rotation_dtype": "float32",
    }


def install_hooks(
    model: torch.nn.Module,
    collector: CovarianceCollector,
) -> list[Any]:
    handles: list[Any] = []
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        handles.append(
            layer.self_attn.q_proj.register_forward_pre_hook(
                collector.hook(layer_idx, "self_attn")
            )
        )
        if collector.capture_mlp:
            handles.append(
                layer.mlp.gate_proj.register_forward_pre_hook(
                    collector.hook(layer_idx, "mlp")
                )
            )
    return handles


def main() -> int:
    args = parse_args()
    args.output_path = args.output_path.resolve()
    device = resolve_device(args.device)
    stat_device = torch.device(args.stat_device)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if getattr(config, "model_type", None) != "qwen2":
        raise ValueError(
            "This capture script is intended for Qwen2/Qwen2.5 models; "
            f"got model_type={getattr(config, 'model_type', None)!r}."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )

    texts = load_texts(args)
    windows = build_token_windows(
        tokenizer,
        texts,
        args.sequence_length,
        args.max_sequences,
    )

    print(
        "setup: "
        f"model={args.model}, device={device}, stat_device={stat_device}, "
        f"dtype={args.dtype}, sequence_length={args.sequence_length}, "
        f"sequences={len(windows)}, output_path={args.output_path}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    model.config.use_cache = False

    collector = CovarianceCollector(
        num_layers=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        stat_device=stat_device,
        capture_mlp=args.capture_mlp,
    )
    handles = install_hooks(model, collector)

    try:
        for start in tqdm(
            range(0, len(windows), args.batch_size),
            desc="capture activation covariance",
        ):
            batch = windows[start : start + args.batch_size]
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
            del input_ids
            empty_device_cache(device)
    finally:
        for handle in handles:
            handle.remove()

    metadata: dict[str, Any] = {
        "model": args.model,
        "model_type": getattr(config, "model_type", None),
        "dataset": {
            "name": args.dataset_name,
            "subset": args.dataset_subset,
            "split": args.dataset_split,
            "size": args.dataset_size,
            "sample": args.dataset_sample,
            "seed": args.dataset_seed,
            "text_file": str(args.text_file) if args.text_file else None,
        },
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "sequences": len(windows),
        "dtype": args.dtype,
        "device": str(device),
        "stat_device": str(stat_device),
        "capture_mlp": args.capture_mlp,
        "layers": {},
    }

    for layer_idx in tqdm(range(int(config.num_hidden_layers)), desc="save rotations"):
        layer_root = args.output_path / "histograms" / f"layer-{layer_idx}"
        layer_metadata = {
            "self_attn": save_rotation_from_covariance(
                collector.attn_cov[layer_idx],
                layer_root / "self_attn",
                damp_percent=args.damp_percent,
                token_count=collector.attn_tokens[layer_idx],
                save_covariance=args.save_covariance,
            ),
        }
        if args.capture_mlp:
            layer_metadata["mlp"] = save_rotation_from_covariance(
                collector.mlp_cov[layer_idx],
                layer_root / "mlp",
                damp_percent=args.damp_percent,
                token_count=collector.mlp_tokens[layer_idx],
                save_covariance=args.save_covariance,
            )
        metadata["layers"][str(layer_idx)] = layer_metadata

    args.output_path.mkdir(parents=True, exist_ok=True)
    (args.output_path / "activation_stats_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    del model
    gc.collect()
    empty_device_cache(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
