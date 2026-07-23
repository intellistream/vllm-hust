# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit rank-local PP optimization cost models from profiling CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    workload: str
    profile_step: int
    pp_rank: int
    layer_num: int
    request_num: int
    aggregated_ctx_len: int
    forward_ns: float
    total_ns: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--trim-extremes", type=int, default=1)
    return parser.parse_args()


def _expected_request_counts(input_dir: Path) -> dict[str, int]:
    checkpoint_path = input_dir / "checkpoint.json"
    with checkpoint_path.open(encoding="utf-8") as source:
        checkpoint = json.load(source)
    expected = {}
    for entry in checkpoint.get("completed", {}).values():
        profile_dir = Path(entry["profile_dir"])
        try:
            workload = str(profile_dir.relative_to(input_dir))
        except ValueError:
            workload = profile_dir.name
        expected[workload] = int(entry["request_num"])
    if not expected:
        raise ValueError(f"no completed workloads in {checkpoint_path}")
    return expected


def _load_samples(input_dir: Path) -> tuple[list[Sample], dict[str, int]]:
    expected = _expected_request_counts(input_dir)
    grouped: dict[tuple, dict[str, float]] = {}
    raw_rows = 0
    excluded_rows = 0
    for profile_path in sorted(input_dir.rglob("profile_pp*_tp*.csv")):
        if "profile_stream" in profile_path.parts:
            continue
        workload = str(profile_path.parent.relative_to(input_dir))
        if workload not in expected:
            continue
        with profile_path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                raw_rows += 1
                request_num = int(row["request_num"])
                if request_num != expected[workload]:
                    excluded_rows += 1
                    continue
                layer_num = int(row["layer_num"])
                ctx_len = int(row["aggregated_ctx_len"])
                if request_num <= 0 or layer_num <= 0 or ctx_len <= 0:
                    excluded_rows += 1
                    continue
                key = (
                    workload,
                    int(row["profile_step"]),
                    int(row["microbatch_id"]),
                    int(row["pp_rank"]),
                    layer_num,
                    request_num,
                    ctx_len,
                )
                timestamps = grouped.setdefault(
                    key,
                    {
                        "t1": float(row["t1_ns"]),
                        "t2": float(row["t2_ns"]),
                        "t3": float(row["t3_ns"]),
                        "t5": float(row["t5_ns"]),
                    },
                )
                timestamps["t1"] = min(timestamps["t1"], float(row["t1_ns"]))
                timestamps["t2"] = min(timestamps["t2"], float(row["t2_ns"]))
                timestamps["t3"] = max(timestamps["t3"], float(row["t3_ns"]))
                timestamps["t5"] = max(timestamps["t5"], float(row["t5_ns"]))

    samples = []
    for key, timestamps in grouped.items():
        workload, step, _, pp_rank, layer_num, request_num, ctx_len = key
        forward_ns = timestamps["t3"] - timestamps["t2"]
        total_ns = timestamps["t5"] - timestamps["t1"]
        if forward_ns <= 0 or total_ns <= 0:
            continue
        samples.append(
            Sample(
                workload=workload,
                profile_step=step,
                pp_rank=pp_rank,
                layer_num=layer_num,
                request_num=request_num,
                aggregated_ctx_len=ctx_len,
                forward_ns=forward_ns,
                total_ns=total_ns,
            )
        )
    return samples, {"raw_rows": raw_rows, "excluded_rows": excluded_rows}


def _filter_samples(
    samples: list[Sample], warmup_fraction: float, trim_extremes: int
) -> tuple[list[Sample], dict[str, int | float]]:
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup fraction must be in [0, 1)")
    if trim_extremes < 0:
        raise ValueError("trim extremes must be non-negative")

    warmup_excluded: set[int] = set()
    for workload in {sample.workload for sample in samples}:
        steps = sorted(
            {sample.profile_step for sample in samples if sample.workload == workload}
        )
        skipped_steps = set(steps[: int(len(steps) * warmup_fraction)])
        warmup_excluded.update(
            index
            for index, sample in enumerate(samples)
            if sample.workload == workload and sample.profile_step in skipped_steps
        )
    warmed = [
        sample for index, sample in enumerate(samples) if index not in warmup_excluded
    ]

    trim_excluded: set[int] = set()
    if trim_extremes:
        for workload in {sample.workload for sample in warmed}:
            workload_indices = [
                index
                for index, sample in enumerate(warmed)
                if sample.workload == workload
            ]
            for field in ("forward_ns", "total_ns"):
                ordered = sorted(
                    workload_indices,
                    key=lambda index: getattr(warmed[index], field),
                )
                count = min(trim_extremes, len(ordered) // 2)
                if count:
                    trim_excluded.update(ordered[:count])
                    trim_excluded.update(ordered[-count:])
    filtered = [
        sample for index, sample in enumerate(warmed) if index not in trim_excluded
    ]
    return filtered, {
        "warmup_fraction": warmup_fraction,
        "warmup_excluded": len(warmup_excluded),
        "trim_extremes": trim_extremes,
        "trim_excluded": len(trim_excluded),
        "final_samples": len(filtered),
    }


def _fit_rank(samples: list[Sample], cost: str, pp_rank: int) -> dict:
    features = np.asarray(
        [
            (
                sample.request_num * sample.layer_num,
                sample.request_num,
                sample.aggregated_ctx_len * sample.layer_num,
                sample.aggregated_ctx_len,
                sample.layer_num,
                1.0,
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    target = np.asarray(
        [getattr(sample, f"{cost}_ns") for sample in samples], dtype=np.float64
    )
    coefficients, _, _, _ = np.linalg.lstsq(features, target, rcond=None)
    prediction = features @ coefficients
    residual = target - prediction
    sse = float(np.sum(residual**2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "cost": cost,
        "pp_rank": pp_rank,
        "layer_num": max(sample.layer_num for sample in samples),
        "profiled_layer_nums": sorted({sample.layer_num for sample in samples}),
        "formula": (
            "cost = p0 * request_num * layer_num + p1 * request_num + "
            "p2 * aggregated_ctx_len * layer_num + p3 * aggregated_ctx_len + "
            "p4 * layer_num + p5"
        ),
        "coefficients": {
            f"p{index}": float(value) for index, value in enumerate(coefficients)
        },
        "sample_count": len(samples),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - sse / sst if sst else 1.0,
        "success": True,
        "message": "numpy.linalg.lstsq completed",
        "optimizer": {"method": "numpy.linalg.lstsq"},
    }


def _portable_input_dir(input_dir: Path) -> str:
    resolved = input_dir.resolve()
    benchmark_dir = Path(__file__).resolve().parent
    try:
        return str(resolved.relative_to(benchmark_dir))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    samples, load_stats = _load_samples(args.input_dir)
    samples, filter_stats = _filter_samples(
        samples, args.warmup_fraction, args.trim_extremes
    )
    if not samples:
        raise SystemExit(f"no valid profile samples found in {args.input_dir}")

    models = []
    for pp_rank in sorted({sample.pp_rank for sample in samples}):
        rank_samples = [sample for sample in samples if sample.pp_rank == pp_rank]
        for cost in ("forward", "total"):
            models.append(_fit_rank(rank_samples, cost, pp_rank))

    result = {
        "metadata": {
            "model_name": args.model_name,
            "deployment": args.deployment,
            "hardware": args.hardware,
            "input_dir": _portable_input_dir(args.input_dir),
        },
        "cost_definitions": {
            "forward": "t3_ns-t2_ns",
            "total": "t5_ns-t1_ns",
        },
        "filters": {"loading": load_stats, "preprocessing": filter_stats},
        "sample_count": len(samples),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"Wrote fit result to {args.output}")


if __name__ == "__main__":
    main()
