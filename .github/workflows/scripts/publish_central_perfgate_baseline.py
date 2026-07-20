from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a trusted main benchmark to the central baseline branch."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-git-repository", type=Path, required=True)
    parser.add_argument("--benchmark-git-repository", type=Path, required=True)
    parser.add_argument("--central-remote", required=True)
    parser.add_argument("--branch", default="benchmark-baselines")
    parser.add_argument("--main-ref", default="origin/main")
    parser.add_argument("--hardware-chip-model", required=True)
    parser.add_argument("--cann-version", required=True)
    parser.add_argument("--update-latest-pointer", action="store_true")
    return parser.parse_args(argv)


def _object(payload: dict[str, Any], key: str, *, field: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"baseline artifact is missing object key {field}")
    return value


def _string(payload: dict[str, Any], key: str, *, field: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"baseline artifact is missing string key {field}")
    return value


def _extract_artifact_fields(payload: dict[str, Any]) -> dict[str, str]:
    metadata = _object(payload, "metadata", field="metadata")
    same_spec = _object(payload, "same_spec", field="same_spec")
    runtime = _object(
        metadata, "runtime_provenance", field="metadata.runtime_provenance"
    )
    engine = _object(runtime, "engine", field="metadata.runtime_provenance.engine")
    plugin = _object(runtime, "plugin", field="metadata.runtime_provenance.plugin")
    return {
        "target_repository": _string(
            metadata, "github_repository", field="metadata.github_repository"
        ),
        "target_sha": _string(metadata, "git_commit", field="metadata.git_commit"),
        "scenario": _string(same_spec, "scenario", field="same_spec.scenario"),
        "spec_id": _string(same_spec, "spec_id", field="same_spec.spec_id"),
        "spec_hash": _string(
            same_spec,
            "resolved_spec_hash",
            field="same_spec.resolved_spec_hash",
        ),
        "vllm_hust_sha": _string(
            engine, "commit", field="metadata.runtime_provenance.engine.commit"
        ),
        "vllm_ascend_hust_sha": _string(
            plugin, "commit", field="metadata.runtime_provenance.plugin.commit"
        ),
    }


def _git_sha(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"unable to resolve Git SHA for {repository}: {detail}")
    return result.stdout.strip()


def _package_version(distribution: str, module: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        try:
            imported = importlib.import_module(module)
        except ImportError as error:
            raise ValueError(
                f"unable to import runtime package {distribution}"
            ) from error
        version = str(getattr(imported, "__version__", "")).strip()
        if not version:
            raise ValueError(
                f"unable to resolve runtime version for {distribution}"
            ) from None
        return version


def _load_protocol() -> Any:
    try:
        from vllm_hust_benchmark import perfgate_baselines
    except ImportError as error:
        raise ValueError(
            "central baseline protocol is unavailable; merge and check out "
            "vllm-hust-benchmark#66 before running the producer"
        ) from error
    return perfgate_baselines


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = json.loads(args.source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("baseline artifact must contain a JSON object")
        fields = _extract_artifact_fields(payload)
        protocol = _load_protocol()
        identity = protocol.BaselineIdentity(
            target_repository=fields["target_repository"],
            target_sha=fields["target_sha"],
            scenario=fields["scenario"],
            spec_id=fields["spec_id"],
            spec_hash=fields["spec_hash"],
        )
        provenance = protocol.BaselineProvenance(
            vllm_hust_sha=fields["vllm_hust_sha"],
            vllm_ascend_hust_sha=fields["vllm_ascend_hust_sha"],
            benchmark_runner_sha=_git_sha(args.benchmark_git_repository),
            hardware_chip_model=args.hardware_chip_model,
            cann_version=args.cann_version,
            torch_version=_package_version("torch", "torch"),
            torch_npu_version=_package_version("torch-npu", "torch_npu"),
        )
        result = protocol.publish_baseline(
            args.central_remote,
            args.branch,
            args.source,
            args.target_git_repository,
            args.main_ref,
            identity,
            provenance,
            update_latest_pointer=args.update_latest_pointer,
        )
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
