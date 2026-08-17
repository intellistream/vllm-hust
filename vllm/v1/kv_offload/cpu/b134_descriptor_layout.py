# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional normalized descriptor capture for B134 layout probes."""

import json
import os
from pathlib import Path
from typing import Any

_CAPTURE_DIR = os.environ.get("B134_DESCRIPTOR_LAYOUT_DIR")
_EVIDENCE_LABEL = os.environ.get(
    "B134_DESCRIPTOR_EVIDENCE_LABEL",
    "existing-server-probe",
)
CAPTURE_ENABLED = bool(_CAPTURE_DIR)
_ALLOWED_EVIDENCE_LABELS = {
    "existing-server-probe",
    "real-online",
    "replay",
    "simulation/model",
}


def capture_descriptor_layout(
    job_id: int,
    direction: str,
    descriptors: list[dict[str, Any]],
) -> Path | None:
    """Write one region-relative inventory without process addresses."""
    if not CAPTURE_ENABLED:
        return None
    if _EVIDENCE_LABEL not in _ALLOWED_EVIDENCE_LABELS:
        raise ValueError(f"unsupported descriptor evidence label: {_EVIDENCE_LABEL}")
    if direction not in {"d2h", "h2d"}:
        raise ValueError(f"unsupported descriptor direction: {direction}")
    if not descriptors:
        raise ValueError("descriptor inventory is empty")

    for descriptor in descriptors:
        if descriptor["src_offset"] < 0 or descriptor["dst_offset"] < 0:
            raise ValueError("descriptor offsets must be non-negative")
        if descriptor["size"] <= 0:
            raise ValueError("descriptor size must be positive")
        if descriptor["direction"] != direction:
            raise ValueError("descriptor direction changed within one inventory")

    assert _CAPTURE_DIR is not None
    capture_dir = Path(_CAPTURE_DIR)
    if not capture_dir.is_dir():
        raise ValueError(f"descriptor capture directory does not exist: {capture_dir}")
    output = capture_dir / f"{os.getpid()}-{direction}-job{job_id}.json"
    payload = {
        "descriptors": descriptors,
        "direction": direction,
        "evidence_label": _EVIDENCE_LABEL,
        "job_id": job_id,
        "schema": "kv-transfer-descriptor-layout/v1",
    }
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd = os.open(
        output,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise OSError("short descriptor inventory write")
    finally:
        os.close(fd)
    return output
