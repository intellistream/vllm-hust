from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def apply_ascend_compatibility_overlay(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    server_parameters = dict(payload.get("server_parameters") or {})
    client_parameters = dict(payload.get("client_parameters") or {})

    server_parameters["no_enable_chunked_prefill"] = True
    server_parameters["no_enable_prefix_caching"] = True
    preview_gpu_memory_utilization = 0.85
    requested_gpu_memory_utilization = server_parameters.get("gpu_memory_utilization")
    if requested_gpu_memory_utilization is None:
        server_parameters["gpu_memory_utilization"] = preview_gpu_memory_utilization
    else:
        try:
            requested_gpu_memory_utilization = float(requested_gpu_memory_utilization)
        except (TypeError, ValueError):
            requested_gpu_memory_utilization = None

        if requested_gpu_memory_utilization is None:
            server_parameters["gpu_memory_utilization"] = preview_gpu_memory_utilization
        else:
            server_parameters["gpu_memory_utilization"] = min(
                requested_gpu_memory_utilization,
                preview_gpu_memory_utilization,
            )
    client_parameters.setdefault("temperature", 0)

    resolved["server_parameters"] = server_parameters
    resolved["client_parameters"] = client_parameters
    return resolved


def write_ascend_compatibility_spec(source: Path, target: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: same-spec file must contain a JSON object")
    resolved = apply_ascend_compatibility_overlay(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the shared Ascend perfgate compatibility spec overlay."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        print(write_ascend_compatibility_spec(args.source, args.output))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
