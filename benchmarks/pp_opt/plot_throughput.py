# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot baseline and PP-opt generation throughput for the benchmark matrix."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "results" / "pp4tp2"
DEFAULT_INPUT = DEFAULT_RESULT_DIR / "throughput.csv"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "throughput.png"
DEFAULT_SMOOTH_WINDOW_SECONDS = 30.0
PANELS = (
    ("qwen3_32b", "conversation", "Qwen3-32B PP4+TP2 / conversation"),
    ("qwen3_32b", "burstgpt", "Qwen3-32B PP4+TP2 / BurstGPT"),
    ("qwen3_235b", "conversation", "Qwen3-235B PP4+TP2 / conversation"),
    ("qwen3_235b", "burstgpt", "Qwen3-235B PP4+TP2 / BurstGPT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smooth-window-seconds",
        type=float,
        default=DEFAULT_SMOOTH_WINDOW_SECONDS,
        help="centered moving-average window; use 0 to plot raw samples",
    )
    return parser.parse_args()


def centered_moving_average(
    x_values: list[float], y_values: list[float], window_seconds: float
) -> list[float]:
    if window_seconds <= 0 or len(x_values) < 2:
        return y_values

    half_window = window_seconds / 2
    prefix_sum = [0.0]
    for value in y_values:
        prefix_sum.append(prefix_sum[-1] + value)

    smoothed = []
    for timestamp in x_values:
        left = bisect.bisect_left(x_values, timestamp - half_window)
        right = bisect.bisect_right(x_values, timestamp + half_window)
        smoothed.append((prefix_sum[right] - prefix_sum[left]) / (right - left))
    return smoothed


def main() -> None:
    args = parse_args()
    if args.smooth_window_seconds < 0:
        raise SystemExit("--smooth-window-seconds must be non-negative")
    series = defaultdict(lambda: ([], []))
    with args.input.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = (row["model"], row["trace"], row["mode"])
            series[key][0].append(float(row["time_s"]))
            series[key][1].append(float(row["generation_tokens_per_s"]))

    present_panels = [
        panel
        for panel in PANELS
        if any(series[(panel[0], panel[1], mode)][0] for mode in ("baseline", "pp_opt"))
    ]
    if not present_panels:
        raise SystemExit(f"no throughput series found in {args.input}")

    column_count = min(2, len(present_panels))
    row_count = math.ceil(len(present_panels) / column_count)
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 3.5 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    styles = {
        "baseline": {"label": "Baseline", "color": "#59636e", "linewidth": 1.2},
        "pp_opt": {"label": "PP-opt", "color": "#007f73", "linewidth": 1.4},
    }
    for axis, (model, trace, title) in zip(axes.flat, present_panels):
        for mode in ("baseline", "pp_opt"):
            x_values, y_values = series[(model, trace, mode)]
            if x_values:
                points = sorted(zip(x_values, y_values))
                sorted_x, sorted_y = map(list, zip(*points))
                plotted_y = centered_moving_average(
                    sorted_x, sorted_y, args.smooth_window_seconds
                )
                axis.plot(sorted_x, plotted_y, **styles[mode])
        axis.set_title(title)
        axis.set_xlabel("Workload elapsed time (s)")
        axis.set_ylabel("Generation throughput (token/s)")
        axis.grid(True, color="#d9dde1", linewidth=0.6, alpha=0.8)
        axis.legend(frameon=False)
    for axis in axes.flat[len(present_panels) :]:
        axis.set_visible(False)

    figure.suptitle(
        "vLLM-HUST Pipeline-Parallel Throughput", fontsize=14, weight="bold"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
