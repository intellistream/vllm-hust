#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare PP-stage compactness from Ascend PyTorch Profiler traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

SCOPE_RE = re.compile(
    r"^pp_forward\|pp=(?P<pp>\d+)\|mb=(?P<mb>-?\d+)"
    r"\|reqs=(?P<reqs>\d+)\|ctx=(?P<ctx>\d+)\|tokens=(?P<tokens>\d+)$"
)


@dataclass(frozen=True)
class Interval:
    start_us: float
    end_us: float


@dataclass(frozen=True)
class ForwardScope:
    stage: int
    microbatch: int
    requests: int
    context_tokens: int
    scheduled_tokens: int
    start_us: float
    end_us: float


@dataclass(frozen=True)
class CubeSample:
    start_us: float
    end_us: float
    utilization: float


@dataclass
class StageTrace:
    stage: int
    rank: int
    compute: list[Interval]
    communication: list[Interval]
    communication_not_overlapped: list[Interval]
    free: list[Interval]
    scopes: list[ForwardScope]
    cube_samples: list[CubeSample]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: item.start_us):
        if interval.end_us <= interval.start_us:
            continue
        if merged and interval.start_us <= merged[-1].end_us:
            previous = merged[-1]
            merged[-1] = Interval(
                previous.start_us, max(previous.end_us, interval.end_us)
            )
        else:
            merged.append(interval)
    return merged


def _clip(
    intervals: Iterable[Interval], start_us: float, end_us: float
) -> list[Interval]:
    return _merge(
        Interval(max(interval.start_us, start_us), min(interval.end_us, end_us))
        for interval in intervals
        if interval.end_us > start_us and interval.start_us < end_us
    )


def _duration(intervals: Iterable[Interval]) -> float:
    return sum(interval.end_us - interval.start_us for interval in intervals)


def _find_trace(profile_root: Path, rank: int) -> Path:
    matches = sorted(
        profile_root.glob(f"rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one parsed trace for global rank {rank} under {profile_root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_cube_samples(trace_path: Path) -> list[CubeSample]:
    kernel_path = trace_path.with_name("kernel_details.csv")
    samples: list[CubeSample] = []
    with kernel_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["Accelerator Core"] != "MIX_AIC":
                continue
            utilization = float(row["cube_utilization(%)"]) / 100.0
            duration_us = float(row["Duration(us)"])
            start_us = float(row["Start Time(us)"])
            if utilization > 0.0 and duration_us > 0.0:
                samples.append(
                    CubeSample(start_us, start_us + duration_us, utilization)
                )
    return samples


def _load_stage(profile_root: Path, stage: int, tp_size: int) -> StageTrace:
    rank = stage * tp_size
    trace_path = _find_trace(profile_root, rank)
    with trace_path.open(encoding="utf-8") as file:
        events = json.load(file)
    if isinstance(events, dict):
        events = events["traceEvents"]

    process_names = {
        event["pid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    thread_names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    overlap_pids = [
        pid for pid, name in process_names.items() if name == "Overlap Analysis"
    ]
    if len(overlap_pids) != 1:
        raise RuntimeError(
            f"rank {rank} has {len(overlap_pids)} Overlap Analysis lanes"
        )
    overlap_pid = overlap_pids[0]

    lanes: dict[str, list[Interval]] = defaultdict(list)
    scopes: list[ForwardScope] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        start_us = float(event["ts"])
        end_us = start_us + float(event.get("dur", 0.0))
        if event.get("pid") == overlap_pid:
            lane = thread_names.get((overlap_pid, event.get("tid")), "unknown")
            lanes[lane].append(Interval(start_us, end_us))

        match = SCOPE_RE.match(str(event.get("name", "")))
        if match:
            fields = {name: int(value) for name, value in match.groupdict().items()}
            if fields["pp"] != stage:
                raise RuntimeError(
                    f"rank {rank} contains PP stage {fields['pp']} scope, "
                    f"expected {stage}"
                )
            scopes.append(
                ForwardScope(
                    stage=stage,
                    microbatch=fields["mb"],
                    requests=fields["reqs"],
                    context_tokens=fields["ctx"],
                    scheduled_tokens=fields["tokens"],
                    start_us=start_us,
                    end_us=end_us,
                )
            )

    if not scopes:
        raise RuntimeError(
            f"rank {rank} has no pp_forward scopes; set "
            "VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 while capturing"
        )
    return StageTrace(
        stage=stage,
        rank=rank,
        compute=_merge(lanes["Computing"]),
        communication=_merge(lanes["Communication"]),
        communication_not_overlapped=_merge(lanes["Communication(Not Overlapped)"]),
        free=_merge(lanes["Free"]),
        scopes=sorted(scopes, key=lambda scope: scope.start_us),
        cube_samples=_load_cube_samples(trace_path),
    )


def _cube_metrics(
    samples: Iterable[CubeSample], start_us: float, end_us: float
) -> tuple[float, float]:
    weighted_utilization_us = 0.0
    sampled_us = 0.0
    for sample in samples:
        overlap_us = max(
            min(sample.end_us, end_us) - max(sample.start_us, start_us), 0.0
        )
        weighted_utilization_us += overlap_us * sample.utilization
        sampled_us += overlap_us
    active_utilization = weighted_utilization_us / sampled_us if sampled_us else 0.0
    wall_clock_proxy = weighted_utilization_us / (end_us - start_us)
    return active_utilization, wall_clock_proxy


def _occupancy_distribution(
    stages: list[StageTrace], start_us: float, end_us: float
) -> dict[int, float]:
    boundaries: list[tuple[float, int]] = []
    for stage in stages:
        for interval in _clip(stage.compute, start_us, end_us):
            boundaries.append((interval.start_us, 1))
            boundaries.append((interval.end_us, -1))
    boundaries.sort(key=lambda item: (item[0], item[1]))

    occupied_us: dict[int, float] = defaultdict(float)
    active = 0
    previous = start_us
    for timestamp, delta in boundaries:
        timestamp = min(max(timestamp, start_us), end_us)
        if timestamp > previous:
            occupied_us[active] += timestamp - previous
            previous = timestamp
        active += delta
    occupied_us[active] += max(end_us - previous, 0.0)
    window_us = end_us - start_us
    return {count: occupied_us[count] / window_us for count in range(len(stages) + 1)}


def _coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values) if values else 0.0
    return statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0


def _profile_metrics(
    label: str, profile_root: Path, pp_size: int, tp_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[StageTrace]]:
    stages = [_load_stage(profile_root, stage, tp_size) for stage in range(pp_size)]
    start_us = min(scope.start_us for stage in stages for scope in stage.scopes)
    end_us = max(scope.end_us for stage in stages for scope in stage.scopes)
    window_us = end_us - start_us
    occupancy = _occupancy_distribution(stages, start_us, end_us)

    stage_rows: list[dict[str, Any]] = []
    compute_duties: list[float] = []
    active_cube_utilizations: list[float] = []
    wall_clock_cube_proxies: list[float] = []
    for stage in stages:
        compute_us = _duration(_clip(stage.compute, start_us, end_us))
        communication_us = _duration(
            _clip(stage.communication_not_overlapped, start_us, end_us)
        )
        free_us = _duration(_clip(stage.free, start_us, end_us))
        scope_durations_ms = [
            (scope.end_us - scope.start_us) / 1000.0 for scope in stage.scopes
        ]
        compute_duty = compute_us / window_us
        active_cube_utilization, wall_clock_cube_proxy = _cube_metrics(
            stage.cube_samples, start_us, end_us
        )
        compute_duties.append(compute_duty)
        active_cube_utilizations.append(active_cube_utilization)
        wall_clock_cube_proxies.append(wall_clock_cube_proxy)
        stage_rows.append(
            {
                "profile": label,
                "stage": stage.stage,
                "global_rank": stage.rank,
                "window_ms": window_us / 1000.0,
                "compute_ms": compute_us / 1000.0,
                "compute_duty": compute_duty,
                "active_cube_utilization": active_cube_utilization,
                "wall_clock_cube_utilization_proxy": wall_clock_cube_proxy,
                "communication_not_overlapped_ms": communication_us / 1000.0,
                "communication_not_overlapped_fraction": communication_us / window_us,
                "reported_free_ms": free_us / 1000.0,
                "forward_scopes": len(stage.scopes),
                "scope_duration_p50_ms": _percentile(scope_durations_ms, 0.50),
                "scope_duration_p95_ms": _percentile(scope_durations_ms, 0.95),
            }
        )

    scope_counts = [len(stage.scopes) for stage in stages]
    wave_skews_ms: list[float] = []
    wave_spans_ms: list[float] = []
    if len(set(scope_counts)) == 1:
        for index in range(scope_counts[0]):
            wave = [stage.scopes[index] for stage in stages]
            wave_skews_ms.append(
                (
                    max(scope.start_us for scope in wave)
                    - min(scope.start_us for scope in wave)
                )
                / 1000.0
            )
            wave_spans_ms.append(
                (
                    max(scope.end_us for scope in wave)
                    - min(scope.start_us for scope in wave)
                )
                / 1000.0
            )

    reference_scopes = stages[0].scopes
    context_sizes = [scope.context_tokens for scope in reference_scopes]
    request_sizes = [scope.requests for scope in reference_scopes]
    summary = {
        "profile": label,
        "profile_root": str(profile_root.resolve()),
        "window_ms": window_us / 1000.0,
        "forward_scopes_per_stage": scope_counts,
        "microbatch_scopes": sum(scope.microbatch >= 0 for scope in reference_scopes),
        "context_tokens_p50": _percentile(context_sizes, 0.50),
        "context_tokens_p95": _percentile(context_sizes, 0.95),
        "context_tokens_cv": _coefficient_of_variation(context_sizes),
        "requests_p50": _percentile(request_sizes, 0.50),
        "mean_stage_compute_duty": statistics.fmean(compute_duties),
        "stage_compute_duty_cv": _coefficient_of_variation(compute_duties),
        "mean_active_cube_utilization": statistics.fmean(active_cube_utilizations),
        "mean_wall_clock_cube_utilization_proxy": statistics.fmean(
            wall_clock_cube_proxies
        ),
        "average_active_compute_stages": sum(
            count * fraction for count, fraction in occupancy.items()
        ),
        "all_stage_compute_idle_fraction": occupancy[0],
        "all_stages_computing_fraction": occupancy[pp_size],
        "at_least_two_stages_computing_fraction": sum(
            fraction for count, fraction in occupancy.items() if count >= 2
        ),
        "compute_occupancy_fraction": {
            str(count): fraction for count, fraction in occupancy.items()
        },
        "wave_start_skew_p50_ms": _percentile(wave_skews_ms, 0.50),
        "wave_start_skew_p95_ms": _percentile(wave_skews_ms, 0.95),
        "wave_span_p50_ms": _percentile(wave_spans_ms, 0.50),
        "wave_span_p95_ms": _percentile(wave_spans_ms, 0.95),
        "window_start_us": start_us,
        "window_end_us": end_us,
    }
    return summary, stage_rows, stages


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_timeline(
    output_path: Path,
    profiles: list[tuple[dict[str, Any], list[StageTrace]]],
    plot_window_ms: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(
        len(profiles), 1, figsize=(13, 2.6 * len(profiles)), sharex=False
    )
    if len(profiles) == 1:
        axes = [axes]
    compute_color = "#0072B2"
    wait_color = "#D55E00"

    for axis, (summary, stages) in zip(axes, profiles):
        start_us = summary["window_start_us"]
        available_ms = summary["window_ms"]
        visible_ms = min(plot_window_ms, available_ms)
        end_us = start_us + visible_ms * 1000.0
        for stage in stages:
            y = len(stages) - 1 - stage.stage
            compute = _clip(stage.compute, start_us, end_us)
            waits = _clip(stage.communication_not_overlapped, start_us, end_us)
            axis.broken_barh(
                [
                    (
                        (interval.start_us - start_us) / 1000.0,
                        (interval.end_us - interval.start_us) / 1000.0,
                    )
                    for interval in waits
                ],
                (y - 0.34, 0.68),
                facecolors=wait_color,
                alpha=0.42,
                linewidth=0,
            )
            axis.broken_barh(
                [
                    (
                        (interval.start_us - start_us) / 1000.0,
                        (interval.end_us - interval.start_us) / 1000.0,
                    )
                    for interval in compute
                ],
                (y - 0.34, 0.68),
                facecolors=compute_color,
                linewidth=0,
            )
        axis.set_yticks(
            range(len(stages)),
            [f"PP {stage}" for stage in reversed(range(len(stages)))],
        )
        axis.set_xlim(0, visible_ms)
        axis.set_ylabel("Stage")
        axis.set_title(
            f"{summary['profile']}: avg active stages "
            f"{summary['average_active_compute_stages']:.2f}, "
            f"all-stage idle {summary['all_stage_compute_idle_fraction']:.1%}"
        )
        axis.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("Time from profiled window start (ms)")
    figure.legend(
        handles=[
            Patch(facecolor=compute_color, label="NPU computing"),
            Patch(
                facecolor=wait_color, alpha=0.42, label="Communication not overlapped"
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _comparison(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    higher_is_better = [
        "mean_stage_compute_duty",
        "mean_active_cube_utilization",
        "mean_wall_clock_cube_utilization_proxy",
        "average_active_compute_stages",
        "all_stages_computing_fraction",
        "at_least_two_stages_computing_fraction",
    ]
    lower_is_better = [
        "stage_compute_duty_cv",
        "all_stage_compute_idle_fraction",
        "wave_start_skew_p50_ms",
        "wave_start_skew_p95_ms",
        "wave_span_p50_ms",
        "wave_span_p95_ms",
    ]
    result: dict[str, Any] = {}
    for metric in higher_is_better:
        before = baseline[metric]
        after = optimized[metric]
        result[metric] = {
            "baseline": before,
            "optimized": after,
            "relative_change": (after / before - 1.0) if before else None,
        }
    for metric in lower_is_better:
        before = baseline[metric]
        after = optimized[metric]
        result[metric] = {
            "baseline": before,
            "optimized": after,
            "reduction": (1.0 - after / before) if before else None,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--optimized-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--plot-window-ms", type=float, default=2000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline, baseline_rows, baseline_stages = _profile_metrics(
        "Baseline", args.baseline_profile, args.pp_size, args.tp_size
    )
    optimized, optimized_rows, optimized_stages = _profile_metrics(
        "PP-opt", args.optimized_profile, args.pp_size, args.tp_size
    )
    output = {
        "baseline": baseline,
        "optimized": optimized,
        "comparison": _comparison(baseline, optimized),
        "notes": [
            "Device intervals come from CANN Overlap Analysis, not CPU launch time.",
            "One TP rank (TP rank 0) represents each PP stage.",
            "This reports pipeline occupancy, not theoretical FLOP-based MFU.",
            "Wall-clock Cube utilization is a CANN PipeUtilization-derived MFU proxy.",
        ],
    }
    with (args.output_dir / "pipeline_compactness.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    _write_csv(
        args.output_dir / "stage_compactness.csv", baseline_rows + optimized_rows
    )
    _plot_timeline(
        args.output_dir / "pipeline_timeline.png",
        [(baseline, baseline_stages), (optimized, optimized_stages)],
        args.plot_window_ms,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
