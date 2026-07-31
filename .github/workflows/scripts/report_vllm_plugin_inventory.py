#!/usr/bin/env python3
"""Report vLLM entry points and verify the Ascend CI allowlist."""

from __future__ import annotations

import os
from importlib.metadata import entry_points

PLUGIN_GROUPS = (
    "vllm.general_plugins",
    "vllm.io_processor_plugins",
    "vllm.platform_plugins",
    "vllm.stat_logger_plugins",
    "vllm.victim_selector",
)
EXPECTED_PLUGINS = {"ascend"}


def main() -> None:
    configured = {
        name.strip()
        for name in os.environ.get("VLLM_PLUGINS", "").split(",")
        if name.strip()
    }
    if configured != EXPECTED_PLUGINS:
        raise SystemExit(
            f"Ascend CI must set VLLM_PLUGINS=ascend; configured={sorted(configured)}"
        )

    found_ascend = False
    for group in PLUGIN_GROUPS:
        print(f"[{group}]")
        for plugin in entry_points(group=group):
            distribution = plugin.dist
            dist_name = distribution.name if distribution is not None else "unknown"
            dist_version = (
                distribution.version if distribution is not None else "unknown"
            )
            active = plugin.name in configured
            print(
                f"- {plugin.name} -> {plugin.value} "
                f"({dist_name} {dist_version}, active={active})"
            )
            if active and group != "vllm.platform_plugins":
                raise SystemExit(
                    f"Allowed plugin name {plugin.name!r} also exists in {group}"
                )
            if group == "vllm.platform_plugins" and plugin.name == "ascend":
                found_ascend = True

    if not found_ascend:
        raise SystemExit("Ascend platform plugin entry point was not discovered")


if __name__ == "__main__":
    main()
