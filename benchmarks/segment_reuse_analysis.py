"""Analyze segment reuse evaluation results.

Reads benchmark result JSON files from the three experimental conditions
(baseline, reorder, stitch) and produces:
- Comparison table (throughput, TTFT, ITL, cache hit rate)
- Per-family breakdown
- Paper-ready LaTeX table output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(path: Path) -> dict | None:
    """Load a benchmark result JSON file."""
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(result: dict) -> dict:
    """Extract key metrics from a benchmark result."""
    if result is None:
        return {}

    # vLLM benchmark output format.
    metrics = {}

    # Throughput.
    metrics["throughput_req_s"] = result.get("request_throughput", 0)
    metrics["throughput_tok_s"] = result.get("output_throughput", 0)
    metrics["total_input_tokens"] = result.get("total_input_tokens", 0)
    metrics["total_output_tokens"] = result.get("total_output_tokens", 0)

    # Latency.
    metrics["mean_ttft_ms"] = result.get("mean_ttft_ms", 0)
    metrics["median_ttft_ms"] = result.get("median_ttft_ms", 0)
    metrics["p99_ttft_ms"] = result.get("p99_ttft_ms", 0)
    metrics["mean_itl_ms"] = result.get("mean_itl_ms", 0)
    metrics["median_itl_ms"] = result.get("median_itl_ms", 0)
    metrics["p99_itl_ms"] = result.get("p99_itl_ms", 0)
    metrics["mean_tpot_ms"] = result.get("mean_tpot_ms", 0)

    # Cache stats.
    metrics["cache_hit_rate"] = result.get("cache_hit_rate", 0)
    metrics["prefix_cache_hit_rate"] = result.get(
        "prefix_cache_hit_rate",
        result.get("cache_hit_rate", 0)
    )

    # Request count.
    metrics["num_requests"] = result.get("num_prompts", 0)
    metrics["completed_requests"] = result.get("completed", 0)

    return metrics


def format_table(conditions: dict[str, dict]) -> str:
    """Format comparison table as plain text."""
    lines = []
    header = f"{'Metric':<30}"
    for name in conditions:
        header += f"  {name:>15}"
    lines.append(header)
    lines.append("-" * len(header))

    all_metrics = set()
    for m in conditions.values():
        all_metrics.update(m.keys())

    for metric in sorted(all_metrics):
        row = f"{metric:<30}"
        for name in conditions:
            val = conditions[name].get(metric, "N/A")
            if isinstance(val, float):
                row += f"  {val:>15.2f}"
            elif isinstance(val, int):
                row += f"  {val:>15d}"
            else:
                row += f"  {str(val):>15}"
        lines.append(row)

    return "\n".join(lines)


def format_latex_table(conditions: dict[str, dict]) -> str:
    """Format comparison table as LaTeX."""
    names = list(conditions.keys())
    col_spec = "l" + "r" * len(names)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Segment Reuse Evaluation Results}",
        "\\label{tab:segment-reuse-results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header.
    header = "Metric"
    for name in names:
        header += f" & {name}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Key metrics rows.
    key_metrics = [
        ("Throughput (req/s)", "throughput_req_s", ".2f"),
        ("Throughput (tok/s)", "throughput_tok_s", ".1f"),
        ("Mean TTFT (ms)", "mean_ttft_ms", ".1f"),
        ("Median TTFT (ms)", "median_ttft_ms", ".1f"),
        ("P99 TTFT (ms)", "p99_ttft_ms", ".1f"),
        ("Mean ITL (ms)", "mean_itl_ms", ".2f"),
        ("Cache Hit Rate", "cache_hit_rate", ".3f"),
    ]

    for label, key, fmt in key_metrics:
        row = label
        vals = []
        for name in names:
            val = conditions[name].get(key, 0)
            vals.append(f"{val:{fmt}}")
        row += " & " + " & ".join(vals)
        row += " \\\\"
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze segment reuse evaluation results."
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing benchmark result JSON files.",
    )
    parser.add_argument(
        "--timestamp", type=str, required=True,
        help="Timestamp suffix to match result files.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file for comparison data.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ts = args.timestamp

    # Load results for each condition.
    conditions = {}
    for name in ["baseline", "reorder", "stitch"]:
        result_path = results_dir / f"{name}_{ts}.json"
        result = load_result(result_path)
        metrics = extract_metrics(result)
        conditions[name] = metrics

    # Print comparison table.
    print("\n=== Comparison Table ===")
    print(format_table(conditions))

    # Print LaTeX table.
    print("\n=== LaTeX Table ===")
    print(format_latex_table(conditions))

    # Compute relative improvements.
    if "baseline" in conditions and "stitch" in conditions:
        baseline = conditions["baseline"]
        stitch = conditions["stitch"]
        print("\n=== Stitch vs Baseline ===")
        for key in ["throughput_req_s", "mean_ttft_ms", "cache_hit_rate"]:
            b = baseline.get(key, 0)
            s = stitch.get(key, 0)
            if b > 0:
                pct = ((s - b) / b) * 100
                print(f"  {key}: {b:.3f} -> {s:.3f} ({pct:+.1f}%)")

    if "reorder" in conditions and "stitch" in conditions:
        reorder = conditions["reorder"]
        stitch = conditions["stitch"]
        print("\n=== Stitch vs Reorder ===")
        for key in ["throughput_req_s", "mean_ttft_ms", "cache_hit_rate"]:
            r = reorder.get(key, 0)
            s = stitch.get(key, 0)
            if r > 0:
                pct = ((s - r) / r) * 100
                print(f"  {key}: {r:.3f} -> {s:.3f} ({pct:+.1f}%)")

    # Save comparison JSON.
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(conditions, f, indent=2)
        print(f"\nSaved comparison -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
