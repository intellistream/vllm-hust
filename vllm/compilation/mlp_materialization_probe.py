# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off MLP materialization path probe.

This module is intentionally tiny and side-effect-free unless the
VLLM_ASCEND_MLP_MATERIALIZATION_CLASSIFY gate is enabled.  It lets the
optimization repository prove which managed-service compile/capture path is
actually used before carrying a real boundary rewrite.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_ENABLE_ENV = "VLLM_ASCEND_MLP_MATERIALIZATION_CLASSIFY"
_OUTPUT_ENV = "VLLM_ASCEND_MLP_MATERIALIZATION_CLASSIFY_FILE"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _truthy(os.getenv(_ENABLE_ENV))


def emit(event: str, **payload: Any) -> None:
    if not enabled():
        return
    output_file = os.getenv(_OUTPUT_ENV)
    if not output_file:
        return
    record = {
        "timestamp_ns": time.time_ns(),
        "evidence_label": "real-compile-graph-path-probe",
        "probe": "mlp_materialization_path",
        "event": event,
        **payload,
    }
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, sort_keys=True) + "\n")
