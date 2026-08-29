# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Early CLI selection for installed extension Bundles."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def apply_cli_extension_selection(bundle_ids: Sequence[str] | None) -> None:
    """Apply explicit CLI selection before engine or child-process startup."""
    if bundle_ids is None:
        return
    selected = tuple(bundle_ids)
    if any(not bundle_id for bundle_id in selected):
        raise ValueError("--extension values must not be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("--extension values must be unique")

    configured = os.environ.get("VLLM_EXTENSION_BUNDLES")
    encoded = ",".join(selected)
    if configured is not None and configured != encoded:
        raise ValueError(
            "--extension conflicts with VLLM_EXTENSION_BUNDLES; configure one "
            "identical ordered selection"
        )
    os.environ["VLLM_EXTENSION_BUNDLES"] = encoded

    # Imports may have constructed an empty snapshot before argparse finished.
    # A child process receives the environment and builds its own snapshot.
    from vllm.plugins.startup import get_configured_extension_startup

    get_configured_extension_startup.cache_clear()


def extension_selection_from_argv(argv: Sequence[str]) -> tuple[str, ...] | None:
    """Read repeated ``--extension`` flags without importing the serve stack."""
    if len(argv) < 2 or argv[1] != "serve":
        return None
    values: list[str] = []
    index = 2
    while index < len(argv):
        argument = argv[index]
        if argument == "--extension":
            if index + 1 >= len(argv):
                return None
            values.append(argv[index + 1])
            index += 2
            continue
        if argument.startswith("--extension="):
            values.append(argument.partition("=")[2])
        index += 1
    return tuple(values) if values else None
