#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify that a built vLLM wheel carries the Extension Bundle v1 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

REQUIRED_CONTRACT_FILES = frozenset(
    {
        "vllm/control_bridge/host_config.schema.json",
        "vllm/plugins/control_action.schema.json",
        "vllm/plugins/control_receipt.schema.json",
        "vllm/plugins/kv_connector_runtime_config.schema.json",
        "vllm/plugins/kv_connector_selection.schema.json",
        "vllm/plugins/manifest.schema.json",
    }
)
REQUIRED_BUILTIN_BUNDLES = frozenset(
    {
        "vllm/plugins/builtin_kv_bundles/lmcache.bundle.json",
        "vllm/plugins/builtin_kv_bundles/mooncake.bundle.json",
    }
)


class WheelContentError(ValueError):
    """Raised when a wheel does not carry the required extension contract."""


def verify_wheel(
    wheel_path: Path, *, require_rust_frontend: bool = False
) -> tuple[str, ...]:
    """Return verified contract members or raise ``WheelContentError``."""
    try:
        with ZipFile(wheel_path) as wheel:
            names = frozenset(wheel.namelist())
            required = REQUIRED_CONTRACT_FILES | REQUIRED_BUILTIN_BUNDLES
            missing = sorted(required - names)
            if missing:
                raise WheelContentError(
                    "wheel is missing required extension artifacts: "
                    + ", ".join(missing)
                )

            for member in sorted(required):
                try:
                    json.loads(wheel.read(member))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise WheelContentError(
                        f"wheel member is not valid JSON: {member}"
                    ) from error

            if require_rust_frontend:
                rust_binary = "vllm/vllm-rs"
                rust_extensions = {
                    name
                    for name in names
                    if name.startswith("vllm/_rust_") and name.endswith(".so")
                }
                if rust_binary not in names or not rust_extensions:
                    raise WheelContentError(
                        "wheel is missing the required Rust frontend binary "
                        "or extension"
                    )
                required = required | {rust_binary} | rust_extensions
    except (BadZipFile, FileNotFoundError, IsADirectoryError) as error:
        raise WheelContentError(f"cannot read wheel: {wheel_path}") from error

    return tuple(sorted(required))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--require-rust-frontend",
        action="store_true",
        help="also require vllm-rs and at least one _rust_*.so extension",
    )
    args = parser.parse_args()
    verified = verify_wheel(
        args.wheel, require_rust_frontend=args.require_rust_frontend
    )
    print(f"verified {len(verified)} extension artifacts in {args.wheel}")


if __name__ == "__main__":
    main()
