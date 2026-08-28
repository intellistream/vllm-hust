#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import regex as re

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MATCHED_FIELDS = (
    "vllm_ascend_hust_sha",
    "benchmark_runner_sha",
    "runtime_manager_sha",
    "hardware_chip_model",
    "cann_version",
    "torch_version",
    "torch_npu_version",
)


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def validate(
    baseline_metadata: Path,
    candidate_provenance: Path,
    expected_target_sha: str,
) -> None:
    manifest = load_object(baseline_metadata)
    baseline = manifest.get("provenance")
    if not isinstance(baseline, dict):
        raise ValueError(f"{baseline_metadata}: missing provenance object")

    candidate = load_object(candidate_provenance)
    if candidate.get("schema_version") != "perfgate-runtime-provenance/v1":
        raise ValueError(
            f"{candidate_provenance}: unsupported runtime provenance schema"
        )

    expected_sha = expected_target_sha.strip().lower()
    if not SHA_PATTERN.fullmatch(expected_sha):
        raise ValueError("expected target SHA must be a full 40-character Git SHA")
    candidate_sha = str(candidate.get("vllm_hust_sha") or "").strip().lower()
    if candidate_sha != expected_sha:
        raise ValueError(
            "candidate Core SHA mismatch: "
            f"expected {expected_sha}, got {candidate_sha or 'unset'}"
        )

    mismatches = []
    for field in MATCHED_FIELDS:
        expected = str(baseline.get(field) or "").strip()
        actual = str(candidate.get(field) or "").strip()
        if field == "runtime_manager_sha" and (
            not SHA_PATTERN.fullmatch(expected.lower())
            or not SHA_PATTERN.fullmatch(actual.lower())
        ):
            mismatches.append(f"{field}: baseline={expected!r}, candidate={actual!r}")
            continue
        if expected != actual:
            mismatches.append(f"{field}: baseline={expected!r}, candidate={actual!r}")
    if mismatches:
        raise ValueError(
            "candidate runtime provenance does not match exact baseline: "
            + "; ".join(mismatches)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--candidate-provenance", type=Path, required=True)
    parser.add_argument("--expected-target-sha", required=True)
    args = parser.parse_args(argv)
    try:
        validate(
            args.baseline_metadata,
            args.candidate_provenance,
            args.expected_target_sha,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print("Perfgate candidate runtime provenance matches the exact baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
