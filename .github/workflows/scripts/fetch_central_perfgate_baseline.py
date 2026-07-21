from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ascend_same_spec_compat import apply_ascend_compatibility_overlay


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and validate one exact central perfgate baseline."
    )
    parser.add_argument("--central-repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-repository", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--spec-file", type=Path, required=True)
    parser.add_argument("--benchmark-git-repository", type=Path, required=True)
    parser.add_argument("--plugin-git-repository", type=Path, required=True)
    parser.add_argument("--hardware-chip-model", required=True)
    parser.add_argument("--cann-version", required=True)
    parser.add_argument("--github-env", type=Path)
    return parser.parse_args(argv)


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


def _load_benchmark_modules(benchmark_repository: Path) -> tuple[Any, Any]:
    source = (benchmark_repository / "src").resolve()
    if not source.is_dir():
        raise ValueError(f"benchmark repository source directory not found: {source}")
    for module_name in list(sys.modules):
        if module_name == "vllm_hust_benchmark" or module_name.startswith(
            "vllm_hust_benchmark."
        ):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(source))
    try:
        from vllm_hust_benchmark import perfgate_baselines, same_spec
    except ImportError as error:
        raise ValueError(
            "central baseline protocol is unavailable from the pinned "
            f"benchmark checkout: {source}"
        ) from error
    return perfgate_baselines, same_spec


def _string(payload: dict[str, Any], key: str, *, field: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"resolved same-spec payload is missing {field}")
    return value


def _resolve_spec_identity(
    spec_file: Path,
    expected_scenario: str,
    same_spec: Any,
) -> tuple[str, str]:
    source_spec = same_spec.load_benchmark_spec(spec_file)
    effective_spec = apply_ascend_compatibility_overlay(source_spec)
    payload = same_spec.build_same_spec_payload(
        effective_spec,
        spec_source=spec_file,
    )
    scenario = _string(payload, "scenario", field="scenario")
    if scenario != expected_scenario:
        raise ValueError(
            "resolved same-spec scenario mismatch: "
            f"expected {expected_scenario}, got {scenario}"
        )
    return (
        _string(payload, "spec_id", field="spec_id"),
        _string(payload, "resolved_spec_hash", field="resolved_spec_hash"),
    )


def _write_github_env(path: Path | None, values: dict[str, str]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for name, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"GitHub environment value contains a newline: {name}")
            handle.write(f"{name}={value}\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        protocol, same_spec = _load_benchmark_modules(args.benchmark_git_repository)
        spec_id, spec_hash = _resolve_spec_identity(
            args.spec_file,
            args.scenario,
            same_spec,
        )
        identity = protocol.BaselineIdentity(
            target_repository=args.target_repository,
            target_sha=args.target_sha,
            scenario=args.scenario,
            spec_id=spec_id,
            spec_hash=spec_hash,
        )
        provenance = protocol.BaselineProvenance(
            vllm_hust_sha=args.target_sha,
            vllm_ascend_hust_sha=_git_sha(args.plugin_git_repository),
            benchmark_runner_sha=_git_sha(args.benchmark_git_repository),
            hardware_chip_model=args.hardware_chip_model,
            cann_version=args.cann_version,
            torch_version=_package_version("torch", "torch"),
            torch_npu_version=_package_version("torch-npu", "torch_npu"),
        )
        output = protocol.fetch_baseline(
            args.central_repository_root,
            args.output,
            identity,
            expected_provenance=provenance,
        )
        _write_github_env(
            args.github_env,
            {
                "PERFGATE_BASELINE_FILE": str(output),
                "PERFGATE_BASELINE_AVAILABLE": "1",
                "PERFGATE_BASELINE_COMMIT": args.target_sha,
                "PERFGATE_BASELINE_SOURCE": "exact",
                "PERFGATE_BASELINE_SPEC_ID": spec_id,
                "PERFGATE_BASELINE_SPEC_HASH": spec_hash,
            },
        )
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "baseline": str(output),
                "source": "exact",
                "target_sha": args.target_sha,
                "spec_id": spec_id,
                "spec_hash": spec_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
