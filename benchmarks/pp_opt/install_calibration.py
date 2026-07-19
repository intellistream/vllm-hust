# SPDX-License-Identifier: Apache-2.0
"""Validate and install a topology-specific PP optimization cost model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as source:
        config = json.load(source)
    with args.fit_result.open(encoding="utf-8") as source:
        fit = json.load(source)
    model_config = args.model_config
    if model_config is None:
        relative_model_config = Path(config["model_dir"]) / "config.json"
        model_config = next(
            (
                parent / relative_model_config
                for parent in args.config.resolve().parents
                if (parent / relative_model_config).is_file()
            ),
            None,
        )
    if model_config is None or not model_config.is_file():
        raise SystemExit(f"model config is missing: {model_config}")

    metadata = fit.setdefault("metadata", {})
    expected_metadata = {
        "model_name": config["model_name"],
        "deployment": config["deployment"],
        "hardware": config["hardware"],
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise SystemExit(f"calibration metadata mismatch: {mismatches}")

    pp_size = int(config["pipeline_parallel_size"])
    layer_partition = [int(value) for value in config["layer_partition"]]
    models = fit.get("models", [])
    by_key = {
        (entry.get("cost"), int(entry.get("pp_rank", -1))): entry for entry in models
    }
    missing = [
        (cost, rank)
        for cost in ("forward", "total")
        for rank in range(pp_size)
        if (cost, rank) not in by_key
    ]
    if missing:
        raise SystemExit(f"calibration is missing cost models: {missing}")

    wrong_layers = []
    for cost in ("forward", "total"):
        for rank, expected_layers in enumerate(layer_partition):
            actual_layers = int(by_key[(cost, rank)].get("layer_num", -1))
            if actual_layers != expected_layers:
                wrong_layers.append((cost, rank, actual_layers, expected_layers))
    if wrong_layers:
        raise SystemExit(f"calibration layer partition mismatch: {wrong_layers}")

    failed_models = [
        (entry.get("cost"), entry.get("pp_rank"), entry.get("message"))
        for entry in models
        if not entry.get("success", False)
    ]
    if failed_models:
        raise SystemExit(f"cost model fitting failed: {failed_models}")

    metadata.update(
        {
            "config_key": config["config_key"],
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "model_config_sha256": hashlib.sha256(
                model_config.read_bytes()
            ).hexdigest(),
            "pipeline_parallel_size": pp_size,
            "tensor_parallel_size": int(config["tensor_parallel_size"]),
            "layer_partition": layer_partition,
            "calibration_run_id": args.run_id,
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(fit, output, indent=2)
        output.write("\n")
    shutil.move(temporary, args.output)
    print(f"Installed calibrated cost model: {args.output}")


if __name__ == "__main__":
    main()
