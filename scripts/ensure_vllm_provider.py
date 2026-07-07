# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import importlib
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    tomllib = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure that the active Python environment exposes the top-level "
            "vllm package from exactly one installed distribution."
        )
    )
    parser.add_argument(
        "--module-name",
        default="vllm",
        help="Top-level import name to validate (default: vllm).",
    )
    parser.add_argument(
        "--expected-distribution",
        default="vllm-hust",
        help=(
            "Distribution name that is expected to provide the top-level "
            "module (default: vllm-hust)."
        ),
    )
    parser.add_argument(
        "--remove-conflicts",
        action="store_true",
        help=(
            "Uninstall extra distributions that also provide the same top-level "
            "module, keeping only --expected-distribution."
        ),
    )
    return parser.parse_args()


def providers_for_module(module_name: str) -> list[str]:
    return sorted(set(importlib_metadata.packages_distributions().get(module_name, [])))


def distribution_metadata_path(distribution_name: str) -> Path:
    distribution = importlib_metadata.distribution(distribution_name)
    return Path(str(distribution._path)).resolve()


def canonicalize_distribution_name(distribution_name: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution_name).lower()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_declared_distribution() -> str | None:
    if tomllib is None:
        return None

    pyproject_path = repo_root() / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = payload.get("project") or {}
    name = project.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def resolve_expected_distribution(
    providers: list[str],
    expected_distribution: str,
) -> str:
    if expected_distribution in providers:
        return expected_distribution

    declared_distribution = repo_declared_distribution()
    if not declared_distribution:
        return expected_distribution

    declared_distribution_canonical = canonicalize_distribution_name(declared_distribution)
    for provider in providers:
        if canonicalize_distribution_name(provider) == declared_distribution_canonical:
            return provider

    return expected_distribution


def filter_repo_local_shadow_metadata(
    providers: list[str],
    expected_distribution: str,
) -> list[str]:
    current_repo_root = repo_root()
    metadata_paths = {
        provider: distribution_metadata_path(provider) for provider in providers
    }
    expected_path = metadata_paths.get(expected_distribution)

    if expected_path is None or current_repo_root not in expected_path.parents:
        return providers

    filtered_providers = []
    for provider in providers:
        provider_path = metadata_paths[provider]
        if provider != expected_distribution and current_repo_root in provider_path.parents:
            continue
        filtered_providers.append(provider)

    return filtered_providers


def uninstall_distribution(distribution_name: str) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        distribution_name,
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    raw_providers = providers_for_module(args.module_name)
    effective_expected_distribution = resolve_expected_distribution(
        raw_providers,
        args.expected_distribution,
    )
    providers = filter_repo_local_shadow_metadata(
        raw_providers,
        effective_expected_distribution,
    )

    if not providers:
        print(
            f"No installed distribution provides top-level {args.module_name!r} ",
            f"for interpreter {sys.executable}",
            file=sys.stderr,
            sep="",
        )
        return 1

    if effective_expected_distribution not in providers:
        print(
            f"Top-level {args.module_name!r} is provided by {providers}, but ",
            f"expected distribution {effective_expected_distribution!r} is missing",
            file=sys.stderr,
            sep="",
        )
        return 1

    conflicts = [
        provider for provider in providers if provider != effective_expected_distribution
    ]
    if conflicts and args.remove_conflicts:
        for provider in conflicts:
            print(
                f"Removing conflicting distribution {provider!r} because it also "
                f"provides top-level {args.module_name!r}"
            )
            uninstall_distribution(provider)
        raw_providers = providers_for_module(args.module_name)
        effective_expected_distribution = resolve_expected_distribution(
            raw_providers,
            args.expected_distribution,
        )
        providers = filter_repo_local_shadow_metadata(
            raw_providers,
            effective_expected_distribution,
        )
        conflicts = [
            provider for provider in providers if provider != effective_expected_distribution
        ]

    if conflicts:
        print(
            f"Conflicting distributions still provide top-level {args.module_name!r}: ",
            f"{providers}. Remove all but {effective_expected_distribution!r}.",
            file=sys.stderr,
            sep="",
        )
        return 1

    module = importlib.import_module(args.module_name)
    module_path = getattr(module, "__file__", None)
    if not module_path:
        print(
            f"Imported {args.module_name!r} without a concrete module path",
            file=sys.stderr,
        )
        return 1

    print(f"module={args.module_name}")
    print(f"provider={effective_expected_distribution}")
    print(f"import_path={module_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
