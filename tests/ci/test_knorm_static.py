# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

KNORM_MANAGER = Path(__file__).resolve().parents[2] / "vllm/knorm/manager.py"


def test_knorm_free_handles_parent_signature_compatibility() -> None:
    text = KNORM_MANAGER.read_text(encoding="utf-8")

    assert "from inspect import signature" in text
    assert "parent_free = super().free" in text
    assert '"prioritize_uncached_for_reuse" in signature(parent_free).parameters' in text
    assert "parent_free(request_id)" in text
