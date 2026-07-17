# SPDX-License-Identifier: Apache-2.0
"""Summarize PP-opt client results and extract vLLM throughput samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "results" / "pp4tp2"
DEFAULT_RAW_DIR = DEFAULT_RESULT_DIR / "raw"
DEFAULT_SUMMARY = DEFAULT_RESULT_DIR / "summary.csv"
DEFAULT_COMPARISON = DEFAULT_RESULT_DIR / "comparison.csv"
DEFAULT_THROUGHPUT = DEFAULT_RESULT_DIR / "throughput.csv"
SERVER_METRIC_RE = re.compile(
    r"INFO (?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<clock>\d{2}:\d{2}:\d{2}).*?"
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs"
)
SUMMARY_FIELDS = (
    "experiment_id",
    "model",
    "trace",
    "mode",
    "pp_size",
    "tp_size",
    "successful_requests",
    "failed_requests",
    "duration_s",
    "actual_output_tokens",
    "output_goodput_tokens_per_s",
    "request_throughput_per_s",
    "mean_server_generation_tokens_per_s",
    "peak_server_generation_tokens_per_s",
)
THROUGHPUT_FIELDS = (
    "experiment_id",
    "model",
    "trace",
    "mode",
    "time_s",
    "prompt_tokens_per_s",
    "generation_tokens_per_s",
    "running_requests",
    "waiting_requests",
)
COMPARISON_FIELDS = (
    "model",
    "trace",
    "baseline_output_goodput_tokens_per_s",
    "pp_opt_output_goodput_tokens_per_s",
    "speedup_factor",
    "speedup_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--throughput", type=Path, default=DEFAULT_THROUGHPUT)
    return parser.parse_args()


def server_timestamp(match: re.Match[str], year: int) -> float:
    value = datetime.strptime(
        f"{year}-{match['month']}-{match['day']} {match['clock']}",
        "%Y-%m-%d %H:%M:%S",
    )
    return time.mktime(value.timetuple())


def load_server_samples(
    path: Path,
    metadata: dict,
    first_send_s: float,
    final_completion_s: float,
) -> list[dict[str, str | float | int]]:
    year = datetime.fromtimestamp(float(metadata["start_timestamp_s"])).year
    samples = []
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = SERVER_METRIC_RE.search(line)
            if match is None:
                continue
            timestamp_s = server_timestamp(match, year)
            if timestamp_s < first_send_s - 1 or timestamp_s > final_completion_s + 1:
                continue
            samples.append(
                {
                    "experiment_id": metadata["experiment_id"],
                    "model": metadata["model_key"],
                    "trace": metadata["trace_key"],
                    "mode": metadata["mode"],
                    "time_s": max(0.0, timestamp_s - first_send_s),
                    "prompt_tokens_per_s": float(match["prompt"]),
                    "generation_tokens_per_s": float(match["generation"]),
                    "running_requests": int(match["running"]),
                    "waiting_requests": int(match["waiting"]),
                }
            )
    return samples


def summarize(run_dir: Path) -> tuple[dict, list[dict]]:
    with (run_dir / "metadata.json").open(encoding="utf-8") as source:
        metadata = json.load(source)
    with (run_dir / "client_results.json").open(encoding="utf-8") as source:
        client = json.load(source)

    results = client["results"]
    successful = [result for result in results if result["success"]]
    first_send_s = min(result["actual_send_timestamp_s"] for result in results)
    final_completion_s = max(result["completion_timestamp_s"] for result in results)
    duration_s = final_completion_s - first_send_s
    actual_output_tokens = sum(result["actual_output_length"] for result in successful)
    server_samples = load_server_samples(
        run_dir / "server.log",
        metadata,
        first_send_s,
        final_completion_s,
    )
    generations = [
        float(sample["generation_tokens_per_s"]) for sample in server_samples
    ]
    stats = client["statistics"]
    summary = {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["model_key"],
        "trace": metadata["trace_key"],
        "mode": metadata["mode"],
        "pp_size": metadata["pp_size"],
        "tp_size": metadata["tp_size"],
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "duration_s": duration_s,
        "actual_output_tokens": actual_output_tokens,
        "output_goodput_tokens_per_s": (
            actual_output_tokens / duration_s if duration_s > 0 else 0.0
        ),
        "request_throughput_per_s": stats.get("throughput_req_per_sec", 0.0),
        "mean_server_generation_tokens_per_s": (
            sum(generations) / len(generations) if generations else 0.0
        ),
        "peak_server_generation_tokens_per_s": max(generations, default=0.0),
    }
    return summary, server_samples


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_modes(summaries: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for summary in summaries:
        key = (summary["model"], summary["trace"])
        grouped.setdefault(key, {})[summary["mode"]] = summary

    comparisons = []
    for (model, trace), modes in sorted(grouped.items()):
        if "baseline" not in modes or "pp_opt" not in modes:
            continue
        baseline = modes["baseline"]
        pp_opt = modes["pp_opt"]
        if baseline["failed_requests"] or pp_opt["failed_requests"]:
            continue
        baseline_goodput = baseline["output_goodput_tokens_per_s"]
        pp_opt_goodput = pp_opt["output_goodput_tokens_per_s"]
        speedup = pp_opt_goodput / baseline_goodput
        comparisons.append(
            {
                "model": model,
                "trace": trace,
                "baseline_output_goodput_tokens_per_s": baseline_goodput,
                "pp_opt_output_goodput_tokens_per_s": pp_opt_goodput,
                "speedup_factor": speedup,
                "speedup_percent": (speedup - 1.0) * 100.0,
            }
        )
    return comparisons


def main() -> None:
    args = parse_args()
    summaries = []
    samples = []
    for run_dir in sorted(args.raw_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        required = ("metadata.json", "client_results.json", "server.log")
        if not all((run_dir / name).is_file() for name in required):
            print(f"Skipping incomplete run: {run_dir.name}")
            continue
        summary, run_samples = summarize(run_dir)
        summaries.append(summary)
        samples.extend(run_samples)

    comparisons = compare_modes(summaries)
    write_csv(args.summary, SUMMARY_FIELDS, summaries)
    write_csv(args.comparison, COMPARISON_FIELDS, comparisons)
    write_csv(args.throughput, THROUGHPUT_FIELDS, samples)
    print(f"Wrote {len(summaries)} summaries to {args.summary}")
    print(f"Wrote {len(comparisons)} comparisons to {args.comparison}")
    print(f"Wrote {len(samples)} throughput samples to {args.throughput}")


if __name__ == "__main__":
    main()
