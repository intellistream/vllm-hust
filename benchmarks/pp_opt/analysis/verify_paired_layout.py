#!/usr/bin/env python3
"""Build and compare correctness manifests for layout-paired PP runs."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

LAYOUT_PROFILE_FIELDS = (
    "layer_num",
    "request_num",
    "aggregated_ctx_len",
    "total_scheduled_tokens",
)
CONCURRENT_REQUIREMENTS = {
    "minimum_request_count": 2,
    "minimum_candidate_microbatch_count": 2,
    "minimum_block_reuse_after_finish_count": 1,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_profile(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [
            {field: int(row[field]) for field in LAYOUT_PROFILE_FIELDS}
            for row in csv.DictReader(source)
        ]


def _load_attention_events(
    run_dir: Path,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    events: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for raw_path in sorted(glob.glob(str(run_dir / "state_flow.pid-*.jsonl"))):
        with open(raw_path, encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event.get(
                    "event"
                ) != "attention_input_before_forward" or not event.get("request_ids"):
                    continue
                rank = (int(event["pp_rank"]), int(event["tp_rank"]))
                events.setdefault(rank, []).append(event)
    return events


def _canonical_layout_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    request_names: dict[str, str] = {}
    steps = []
    for ordinal, event in enumerate(events):
        members = []
        scheduled = []
        for request_id in event["request_ids"]:
            canonical = request_names.setdefault(
                request_id, f"request_{len(request_names)}"
            )
            members.append(canonical)
            scheduled.append(
                [canonical, int(event["num_scheduled_tokens"][request_id])]
            )
        steps.append(
            {
                "ordinal": ordinal,
                "members": members,
                "scheduled_tokens": scheduled,
                "batch_size": len(members),
                "input_shape": event["input_ids"]["shape"],
                "position_shape": event["positions"]["shape"],
                "position_values": event["positions"].get("values"),
            }
        )
    return steps


def _load_scheduler_events(run_dir: Path) -> list[dict[str, Any]]:
    events = []
    for raw_path in sorted(glob.glob(str(run_dir / "state_flow.pid-*.jsonl"))):
        with open(raw_path, encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event.get("event") == "scheduler_exit":
                    events.append(event)
    return sorted(events, key=lambda event: int(event["event_seq"]))


def _scheduler_coverage(events: list[dict[str, Any]]) -> dict[str, Any]:
    request_names: dict[str, str] = {}
    last_owner: dict[int, str] = {}
    finished: set[str] = set()
    completion_ordinals: dict[str, int] = {}
    reuse_events = []
    microbatch_ids = set()

    def canonical(request_id: str) -> str:
        return request_names.setdefault(request_id, f"request_{len(request_names)}")

    for ordinal, event in enumerate(events):
        microbatch_id = event.get("microbatch_id")
        if isinstance(microbatch_id, int) and microbatch_id >= 0:
            microbatch_ids.add(microbatch_id)
        newly_finished = set(event.get("finished_req_ids", []))
        scheduler_output = event.get("scheduler_output") or {}
        newly_finished.update(scheduler_output.get("finished_req_ids", []))
        for request_id in newly_finished:
            name = canonical(request_id)
            finished.add(name)
            completion_ordinals.setdefault(name, ordinal)

        for state in event.get("request_states", []):
            owner = canonical(state["request_id"])
            block_ids = {
                int(block_id)
                for group in state.get("block_ids", [])
                for block_id in group
                if int(block_id) != 0
            }
            for block_id in sorted(block_ids):
                previous = last_owner.get(block_id)
                if previous is not None and previous != owner and previous in finished:
                    reuse_events.append(
                        {
                            "ordinal": ordinal,
                            "previous_owner": previous,
                            "new_owner": owner,
                            "previous_completion_ordinal": completion_ordinals[
                                previous
                            ],
                        }
                    )
                last_owner[block_id] = owner

    return {
        "request_count": len(request_names),
        "candidate_microbatch_ids": sorted(microbatch_ids),
        "candidate_microbatch_count": len(microbatch_ids),
        "completed_requests": sorted(finished),
        "completion_count": len(finished),
        "block_reuse_after_finish_events": reuse_events,
        "block_reuse_after_finish_count": len(reuse_events),
    }


def _layout_sha256(steps: list[dict[str, Any]]) -> str:
    encoded = json.dumps(steps, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    profiles = sorted(run_dir.glob("pp_profile_pp*_tp*.csv"))
    if not profiles:
        raise ValueError(f"no PP profile CSV files found in {run_dir}")
    profile_rows = {path.name: _read_profile(path) for path in profiles}
    representative_profile = profile_rows.get("pp_profile_pp0_tp0.csv")
    if representative_profile is None:
        raise ValueError("pp_profile_pp0_tp0.csv is required")
    for name, rows in profile_rows.items():
        if rows != representative_profile:
            raise ValueError(f"profile layout differs across ranks: {name}")

    attention_by_rank = _load_attention_events(run_dir)
    expected_ranks = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if set(attention_by_rank) != expected_ranks:
        raise ValueError(
            f"attention ranks are {sorted(attention_by_rank)}, "
            f"expected {sorted(expected_ranks)}"
        )
    layouts = {
        rank: _canonical_layout_steps(events)
        for rank, events in attention_by_rank.items()
    }
    representative_layout = layouts[(0, 0)]
    for rank, layout in layouts.items():
        if layout != representative_layout:
            raise ValueError(f"state-flow layout differs across ranks: {rank}")
    if len(representative_profile) != len(representative_layout):
        raise ValueError(
            "profile/state-flow step count differs: "
            f"{len(representative_profile)} != {len(representative_layout)}"
        )
    for index, (profile, layout) in enumerate(
        zip(representative_profile, representative_layout)
    ):
        if profile["request_num"] != layout["batch_size"]:
            raise ValueError(f"batch size differs at step {index}")
        if profile["total_scheduled_tokens"] != sum(
            value for _, value in layout["scheduled_tokens"]
        ):
            raise ValueError(f"scheduled-token count differs at step {index}")

    with (run_dir / "client_results.json").open(encoding="utf-8") as source:
        client = json.load(source)
    results = sorted(client["results"], key=lambda item: int(item["request_id"]))
    outputs = [
        {
            "request_index": index,
            "output_length": int(result["output_length"]),
            "actual_output_length": int(result["actual_output_length"]),
            "output_token_ids_sha256": result.get("output_token_ids_sha256"),
            "output_token_ids": result.get("output_token_ids"),
        }
        for index, result in enumerate(results)
    ]
    coverage = _scheduler_coverage(_load_scheduler_events(run_dir))

    raw_files = [
        run_dir / "metadata.json",
        run_dir / "client_results.json",
        *profiles,
        *sorted(run_dir.glob("state_flow.pid-*.jsonl")),
    ]
    return {
        "schema": "vllm.pp_paired_layout_manifest.v1",
        "run_dir": str(run_dir),
        "layout_definition": (
            "Canonical first-seen request membership, per-request scheduled-token "
            "counts, batch size, input shape, and positions for every non-warmup "
            "model step. Raw microbatch labels and process-local request IDs are "
            "excluded because they do not affect layout."
        ),
        "step_count": len(representative_layout),
        "layout_sha256": _layout_sha256(representative_layout),
        "layout_steps": representative_layout,
        "profile_layout": representative_profile,
        "outputs": outputs,
        "coverage": coverage,
        "source_sha256": {path.name: file_sha256(path) for path in raw_files},
    }


def _concurrent_coverage_checks(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_coverage = baseline.get("coverage", {})
    candidate_coverage = candidate.get("coverage", {})
    checks = {
        "baseline_request_count": (
            baseline_coverage.get("request_count", 0)
            >= CONCURRENT_REQUIREMENTS["minimum_request_count"]
        ),
        "candidate_request_count": (
            candidate_coverage.get("request_count", 0)
            >= CONCURRENT_REQUIREMENTS["minimum_request_count"]
        ),
        "candidate_has_distinct_microbatches": (
            candidate_coverage.get("candidate_microbatch_count", 0)
            >= CONCURRENT_REQUIREMENTS["minimum_candidate_microbatch_count"]
        ),
        "baseline_reuses_block_after_finish": (
            baseline_coverage.get("block_reuse_after_finish_count", 0)
            >= CONCURRENT_REQUIREMENTS["minimum_block_reuse_after_finish_count"]
        ),
        "candidate_reuses_block_after_finish": (
            candidate_coverage.get("block_reuse_after_finish_count", 0)
            >= CONCURRENT_REQUIREMENTS["minimum_block_reuse_after_finish_count"]
        ),
    }
    return {
        "requirements": CONCURRENT_REQUIREMENTS,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_manifests(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    scope: str = "single_request",
) -> dict[str, Any]:
    if scope not in {"single_request", "concurrent_recycle"}:
        raise ValueError(f"unsupported gate scope: {scope}")
    layout_match = baseline["layout_sha256"] == candidate["layout_sha256"]
    result: dict[str, Any] = {
        "schema": "vllm.pp_paired_layout_comparison.v1",
        "baseline_layout_sha256": baseline["layout_sha256"],
        "candidate_layout_sha256": candidate["layout_sha256"],
        "layout_match": layout_match,
        "scope": scope,
        "verdict": "incomparable_layout",
        "output_checks": [],
    }
    if not layout_match:
        return result

    baseline_outputs = baseline["outputs"]
    candidate_outputs = candidate["outputs"]
    if len(baseline_outputs) != len(candidate_outputs):
        result["verdict"] = "failed_output_count"
        return result
    checks = []
    for expected, actual in zip(baseline_outputs, candidate_outputs):
        same_length = (
            expected["actual_output_length"]
            == actual["actual_output_length"]
            == expected["output_length"]
            == actual["output_length"]
        )
        same_tokens = (
            expected["output_token_ids_sha256"] == actual["output_token_ids_sha256"]
        )
        checks.append(
            {
                "request_index": expected["request_index"],
                "same_length": same_length,
                "same_full_token_sha256": same_tokens,
            }
        )
    result["output_checks"] = checks
    output_passed = all(
        check["same_length"] and check["same_full_token_sha256"] for check in checks
    )
    if not output_passed:
        result["verdict"] = "failed_exact"
        return result
    if scope == "single_request":
        result["verdict"] = "passed_exact_narrow"
        return result

    coverage = _concurrent_coverage_checks(baseline, candidate)
    result["concurrent_coverage"] = coverage
    result["verdict"] = (
        "passed_exact_concurrent_recycle"
        if coverage["passed"]
        else "insufficient_concurrent_coverage"
    )
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("run_dir", type=Path)
    manifest_parser.add_argument("output", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("output", type=Path)
    compare_parser.add_argument(
        "--scope",
        choices=("single_request", "concurrent_recycle"),
        default="single_request",
    )
    args = parser.parse_args()

    if args.command == "manifest":
        payload = build_manifest(args.run_dir)
    else:
        with args.baseline.open(encoding="utf-8") as source:
            baseline = json.load(source)
        with args.candidate.open(encoding="utf-8") as source:
            candidate = json.load(source)
        payload = compare_manifests(baseline, candidate, scope=args.scope)
    _write_json(args.output, payload)


if __name__ == "__main__":
    main()
