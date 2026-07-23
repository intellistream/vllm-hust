# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate deterministic microbatch calibration traces."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

REQUEST_COUNTS = (16, 32, 64, 128, 192, 256)
KV_UTILIZATIONS = (0.6, 0.7, 0.8, 0.9)
DISTRIBUTIONS = ("uniform", "normal")
OUTPUT_TOKENS = 100
MIN_INPUT_TOKENS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kv-cache-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _weights(distribution: str, count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    if distribution == "uniform":
        return [rng.uniform(0.5, 1.5) for _ in range(count)]
    if distribution == "normal":
        return [max(0.2, rng.gauss(1.0, 0.25)) for _ in range(count)]
    raise ValueError(f"unsupported distribution: {distribution}")


def _allocate_lengths(
    count: int,
    target_tokens: int,
    max_model_len: int,
    weights: list[float],
) -> list[int]:
    minimum = MIN_INPUT_TOKENS + OUTPUT_TOKENS
    maximum = max_model_len
    if target_tokens < count * minimum or target_tokens > count * maximum:
        raise ValueError(
            f"target {target_tokens} is outside [{count * minimum}, "
            f"{count * maximum}] for {count} requests"
        )

    lengths = [minimum] * count
    remaining = target_tokens - count * minimum
    active = set(range(count))
    while remaining and active:
        weight_sum = sum(weights[index] for index in active)
        allocations: dict[int, int] = {}
        for index in active:
            capacity = maximum - lengths[index]
            share = max(1, math.floor(remaining * weights[index] / weight_sum))
            allocations[index] = min(capacity, share)

        allocated = min(remaining, sum(allocations.values()))
        if allocated == 0:
            break
        for index in sorted(active):
            amount = min(allocations[index], allocated)
            lengths[index] += amount
            allocated -= amount
            remaining -= amount
            if allocated == 0:
                break
        active = {index for index in active if lengths[index] < maximum}

    index = 0
    while remaining:
        if lengths[index] < maximum:
            lengths[index] += 1
            remaining -= 1
        index = (index + 1) % count
    return lengths


def main() -> None:
    args = parse_args()
    if args.kv_cache_size <= 0 or args.max_model_len <= OUTPUT_TOKENS:
        raise SystemExit("KV cache size and model length must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    generated = 0
    for distribution_index, distribution in enumerate(DISTRIBUTIONS):
        for request_count in REQUEST_COUNTS:
            for utilization in KV_UTILIZATIONS:
                target_tokens = int(args.kv_cache_size * utilization)
                seed = (
                    distribution_index * 1_000_000
                    + request_count * 1_000
                    + int(utilization * 100)
                )
                total_lengths = _allocate_lengths(
                    request_count,
                    target_tokens,
                    args.max_model_len,
                    _weights(distribution, request_count, seed),
                )
                name = (
                    f"{distribution}_req{request_count}_"
                    f"kv{args.kv_cache_size}_util{int(utilization * 100):03d}.csv"
                )
                output_path = args.output_dir / name
                if output_path.exists() and not args.overwrite:
                    raise SystemExit(f"workload already exists: {output_path}")
                with output_path.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(
                        output,
                        fieldnames=("timestamp", "input_length", "total_length"),
                    )
                    writer.writeheader()
                    for total_length in total_lengths:
                        writer.writerow(
                            {
                                "timestamp": 0,
                                "input_length": total_length - OUTPUT_TOKENS,
                                "total_length": total_length,
                            }
                        )
                summary_rows.append(
                    {
                        "path": name,
                        "distribution": distribution,
                        "request_count": request_count,
                        "kv_utilization": utilization,
                        "total_tokens": sum(total_lengths),
                        "min_total_length": min(total_lengths),
                        "max_total_length": max(total_lengths),
                    }
                )
                generated += 1

    summary_path = args.output_dir / "profile_workload_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Generated {generated} calibration workloads in {args.output_dir}")


if __name__ == "__main__":
    main()
