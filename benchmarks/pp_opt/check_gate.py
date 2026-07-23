# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail unless each PP-opt regression-gate result matches or beats baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    parser.add_argument("--expected-comparisons", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.comparison.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != args.expected_comparisons:
        raise SystemExit(
            f"gate has {len(rows)} complete comparisons; "
            f"expected {args.expected_comparisons}"
        )

    failed = []
    for row in rows:
        speedup = float(row["speedup_factor"])
        print(f"{row['model']} {row['trace']}: {speedup:.3f}x")
        if speedup < args.minimum_speedup:
            failed.append((row["model"], row["trace"], speedup))
    if failed:
        raise SystemExit(f"PP-opt regression gate failed: {failed}")
    print("PP-opt regression gate passed.")


if __name__ == "__main__":
    main()
