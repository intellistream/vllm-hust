# SPDX-License-Identifier: Apache-2.0
"""Capture TEAL activation histograms and thresholds for Qwen-style models."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
PROJECTION_THRESHOLDS = {
    "self_attn.qkv": ("self_attn", "h1"),
    "self_attn.o": ("self_attn", "h2"),
    "mlp.gate_up": ("mlp", "h1"),
    "mlp.down": ("mlp", "h2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TEAL histogram/threshold artifacts from activations."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, default=0.4)

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
    )
    dataset.add_argument("--dataset-seed", type=int, default=0)
    dataset.add_argument("--text-file", type=Path, default=None)
    dataset.add_argument("--inline-text", action="append", default=None)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--sequence-length", type=int, default=2048)
    runtime.add_argument("--batch-size", type=int, default=1)
    runtime.add_argument("--max-sequences", type=int, default=None)
    runtime.add_argument("--histogram-bins", type=int, default=2048)
    runtime.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    runtime.add_argument("--device", default="auto")
    runtime.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
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
            "No full TEAL histogram windows were produced. "
            f"Need at least {sequence_length} tokens."
        )
    return [
        token_ids[i * sequence_length : (i + 1) * sequence_length]
        for i in range(num_sequences)
    ]


def site_key(layer_idx: int, part: str, hidden_type: str) -> tuple[int, str, str]:
    return (layer_idx, part, hidden_type)


class RangeCollector:
    def __init__(self, num_layers: int) -> None:
        self.max_abs = {
            site_key(layer_idx, part, hidden_type): 0.0
            for layer_idx in range(num_layers)
            for part in ("self_attn", "mlp")
            for hidden_type in ("h1", "h2")
        }

    def hook(self, layer_idx: int, part: str, hidden_type: str):
        key = site_key(layer_idx, part, hidden_type)

        def _hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            del module
            x = inputs[0].detach()
            value = float(x.abs().max().to(device="cpu", dtype=torch.float32).item())
            self.max_abs[key] = max(self.max_abs[key], value)

        return _hook


class HistogramCollector:
    def __init__(
        self,
        max_abs: dict[tuple[int, str, str], float],
        bins: int,
    ) -> None:
        self.bins = bins
        self.ranges = {key: max(value * 1.001, 1e-6) for key, value in max_abs.items()}
        self.counts = {key: torch.zeros(bins, dtype=torch.float32) for key in max_abs}
        self.tokens = {key: 0 for key in max_abs}

    def centers(self, key: tuple[int, str, str]) -> torch.Tensor:
        edge = self.ranges[key]
        width = (2.0 * edge) / self.bins
        return torch.linspace(
            -edge + width / 2.0,
            edge - width / 2.0,
            self.bins,
            dtype=torch.float32,
        )

    def hook(self, layer_idx: int, part: str, hidden_type: str):
        key = site_key(layer_idx, part, hidden_type)

        def _hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            del module
            x = inputs[0].detach().reshape(-1).to(dtype=torch.float32)
            edge = self.ranges[key]
            hist = torch.histc(x, bins=self.bins, min=-edge, max=edge)
            self.counts[key].add_(hist.to(device="cpu", dtype=torch.float32))
            self.tokens[key] += int(x.numel())

        return _hook


def install_hooks(model: torch.nn.Module, collector: Any) -> list[Any]:
    handles: list[Any] = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(
            layer.self_attn.q_proj.register_forward_pre_hook(
                collector.hook(layer_idx, "self_attn", "h1")
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                collector.hook(layer_idx, "self_attn", "h2")
            )
        )
        handles.append(
            layer.mlp.gate_proj.register_forward_pre_hook(
                collector.hook(layer_idx, "mlp", "h1")
            )
        )
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                collector.hook(layer_idx, "mlp", "h2")
            )
        )
    return handles


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def run_windows(
    model: torch.nn.Module,
    windows: list[list[int]],
    batch_size: int,
    device: torch.device,
    desc: str,
) -> None:
    for start in tqdm(range(0, len(windows), batch_size), desc=desc):
        batch = windows[start : start + batch_size]
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
        del input_ids


def official_icdf_from_counts(
    centers: torch.Tensor,
    counts: torch.Tensor,
    sparsity: float,
) -> torch.Tensor:
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


def save_artifacts(
    collector: HistogramCollector,
    output_path: Path,
    num_layers: int,
    sparsity: float,
) -> dict[str, Any]:
    hist_root = output_path / "official_histograms"
    threshold_root = output_path / "thresholds"
    threshold_root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"layers": {}, "thresholds": {}}

    for layer_idx in range(num_layers):
        metadata["layers"][str(layer_idx)] = {}
        for part in ("self_attn", "mlp"):
            part_dir = hist_root / f"layer-{layer_idx}" / part
            part_dir.mkdir(parents=True, exist_ok=True)
            payload = {}
            for hidden_type in ("h1", "h2"):
                key = site_key(layer_idx, part, hidden_type)
                centers = collector.centers(key)
                counts = collector.counts[key]
                payload[f"{hidden_type}_centers"] = centers
                payload[hidden_type] = counts
                metadata["layers"][str(layer_idx)][f"{part}.{hidden_type}"] = {
                    "tokens": collector.tokens[key],
                    "bins": collector.bins,
                    "range": collector.ranges[key],
                    "count": float(counts.sum().item()),
                }
            torch.save(payload, part_dir / "histograms.pt")

        for projection, (part, hidden_type) in PROJECTION_THRESHOLDS.items():
            key = site_key(layer_idx, part, hidden_type)
            threshold = official_icdf_from_counts(
                collector.centers(key),
                collector.counts[key],
                sparsity,
            )
            filename = f"layers.{layer_idx}.{projection}.threshold.pt"
            torch.save(threshold, threshold_root / filename)
            metadata["thresholds"][filename] = float(threshold.item())

    (output_path / "activation_sparsity_config.json").write_text(
        json.dumps(
            {
                "enable": True,
                "method": "teal",
                "uniform_sparsity": sparsity,
                "calibration_path": str(threshold_root.resolve()),
                "decode_only": False,
                "apply_all_tokens": False,
                "prefill_sparsify": "half",
                "strict_unsupported_check": True,
                "use_sparse_gemv": False,
            },
            indent=2,
        )
        + "\n"
    )
    return metadata


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.sparsity) or not 0.0 <= args.sparsity <= 1.0:
        raise ValueError(f"--sparsity must be in [0, 1], got {args.sparsity}")

    args.output_path = args.output_path.resolve()
    device = resolve_device(args.device)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if getattr(config, "model_type", None) != "qwen2":
        raise ValueError(
            "This capture script is intended for Qwen2/Qwen2.5 models; "
            f"got model_type={getattr(config, 'model_type', None)!r}."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
        f"model={args.model}, device={device}, dtype={args.dtype}, "
        f"sequence_length={args.sequence_length}, sequences={len(windows)}, "
        f"bins={args.histogram_bins}, output_path={args.output_path}"
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

    range_collector = RangeCollector(num_layers=int(config.num_hidden_layers))
    handles = install_hooks(model, range_collector)
    try:
        run_windows(model, windows, args.batch_size, device, "capture TEAL ranges")
    finally:
        remove_hooks(handles)

    histogram_collector = HistogramCollector(
        range_collector.max_abs,
        bins=args.histogram_bins,
    )
    handles = install_hooks(model, histogram_collector)
    try:
        run_windows(model, windows, args.batch_size, device, "capture TEAL histograms")
    finally:
        remove_hooks(handles)

    args.output_path.mkdir(parents=True, exist_ok=True)
    metadata = save_artifacts(
        histogram_collector,
        args.output_path,
        num_layers=int(config.num_hidden_layers),
        sparsity=args.sparsity,
    )
    metadata.update(
        {
            "model": args.model,
            "model_type": config.model_type,
            "sparsity": args.sparsity,
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
            "histogram_bins": args.histogram_bins,
        }
    )
    (args.output_path / "teal_histogram_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    del model
    gc.collect()
    empty_device_cache(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
