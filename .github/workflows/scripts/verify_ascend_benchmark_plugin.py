#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the Ascend plugin checkout and installed entry point."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_CHECKOUT_FILES = (
    "setup.py",
    "scripts/install_local_ascend_plugin.sh",
    "scripts/use_single_ascend_env.sh",
    "scripts/hust_ascend_manager_helper.sh",
)
HEX_DIGITS = frozenset("0123456789abcdef")


def _is_full_git_sha(value: str) -> bool:
    return len(value) == 40 and set(value).issubset(HEX_DIGITS)


def _module_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"{module_name} is not importable")
    return Path(spec.origin).resolve()


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def verify_checkout(plugin_repo: Path, expected_sha: str) -> dict[str, Any]:
    plugin_repo = plugin_repo.resolve()
    missing = [
        relative_path
        for relative_path in REQUIRED_CHECKOUT_FILES
        if not (plugin_repo / relative_path).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"plugin checkout is missing required files: {', '.join(missing)}"
        )
    if (
        not (plugin_repo / "scripts/install_local_ascend_plugin.sh").stat().st_mode
        & 0o111
    ):
        raise RuntimeError("plugin install script is not executable")

    setup_text = (plugin_repo / "setup.py").read_text(encoding="utf-8")
    normalized_setup = "".join(setup_text.split())
    if (
        "vllm.platform_plugins" not in setup_text
        or "ascend=vllm_ascend:register" not in normalized_setup
    ):
        raise RuntimeError(
            "plugin checkout does not declare the Ascend platform entry point"
        )

    head_sha = subprocess.run(
        ["git", "-C", str(plugin_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _is_full_git_sha(head_sha):
        raise RuntimeError("plugin checkout HEAD is not a full Git SHA")
    if expected_sha:
        if not _is_full_git_sha(expected_sha):
            raise RuntimeError("expected plugin SHA is not a full immutable Git SHA")
        if head_sha != expected_sha:
            raise RuntimeError(
                f"plugin checkout HEAD does not match expected SHA: {head_sha}"
            )
    return {"mode": "checkout", "status": "passed", "plugin_sha": head_sha}


def verify_installed(core_repo: Path, plugin_repo: Path) -> dict[str, Any]:
    core_repo = core_repo.resolve()
    plugin_repo = plugin_repo.resolve()
    vllm_path = _module_path("vllm")
    plugin_path = _module_path("vllm_ascend")
    _module_path("xxhash")
    if not _under(vllm_path, core_repo):
        raise RuntimeError(
            f"vllm is not loaded from the checked-out Core repo: {vllm_path}"
        )
    if not _under(plugin_path, plugin_repo):
        raise RuntimeError(
            f"vllm_ascend is not loaded from the checked-out plugin repo: {plugin_path}"
        )

    entry_points = [
        entry_point
        for entry_point in metadata.entry_points(group="vllm.platform_plugins")
        if entry_point.name == "ascend"
    ]
    if len(entry_points) != 1:
        raise RuntimeError(
            "expected exactly one installed Ascend platform entry point, "
            f"found {len(entry_points)}"
        )
    entry_point = entry_points[0]
    if entry_point.value != "vllm_ascend:register":
        raise RuntimeError(
            f"unexpected Ascend platform entry point: {entry_point.value}"
        )
    if not callable(entry_point.load()):
        raise RuntimeError("Ascend platform entry point does not resolve to a callable")

    return {
        "mode": "installed",
        "status": "passed",
        "vllm_module": str(vllm_path),
        "vllm_ascend_module": str(plugin_path),
        "platform_entry_point": entry_point.value,
        "xxhash_version": metadata.version("xxhash"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("checkout", "installed"))
    parser.add_argument("--core-repo", type=Path)
    parser.add_argument("--plugin-repo", type=Path, required=True)
    parser.add_argument("--expected-sha", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "checkout":
            payload = verify_checkout(args.plugin_repo, args.expected_sha)
        else:
            if args.core_repo is None:
                raise RuntimeError("--core-repo is required in installed mode")
            payload = verify_installed(args.core_repo, args.plugin_repo)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        payload = {"mode": args.mode, "status": "failed", "reason": str(error)}
        status = 2
    else:
        status = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Ascend plugin {args.mode} preflight: {payload['status']}")
    if status:
        print(payload["reason"])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
