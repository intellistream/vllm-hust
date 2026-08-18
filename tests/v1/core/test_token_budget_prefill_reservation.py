# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "vllm/v1/core/sched/token_budget_reservation.py"
SCHEDULER_PATH = ROOT / "vllm/v1/core/sched/scheduler.py"
SPEC = importlib.util.spec_from_file_location(
    "token_budget_reservation_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    (
        "configured_tokens",
        "scheduling_policy",
        "request_priority",
        "remaining_prefill_tokens",
        "max_step_tokens",
        "enable_chunked_prefill",
        "expected",
    ),
    [
        (0, "priority", -1, 64, 128, True, 0),
        (32, "fcfs", -1, 64, 128, True, 0),
        (32, "priority", 0, 64, 128, True, 0),
        (32, "priority", 1, 64, 128, True, 0),
        (32, "priority", -1, 0, 128, True, 0),
        (32, "priority", -1, 64, 0, True, 0),
        (32, "priority", -1, 64, 128, True, 32),
        (96, "priority", -1, 64, 128, True, 64),
        (96, "priority", -1, 256, 48, True, 48),
        (64, "priority", -7, 64, 64, False, 64),
        (32, "priority", -1, 64, 128, False, 0),
        (-1, "priority", -1, 64, 128, True, 0),
    ],
)
def test_priority_prefill_reservation_matrix(
    configured_tokens: int,
    scheduling_policy: str,
    request_priority: int,
    remaining_prefill_tokens: int,
    max_step_tokens: int,
    enable_chunked_prefill: bool,
    expected: int,
) -> None:
    actual = MODULE.priority_prefill_reservation_tokens(
        configured_tokens=configured_tokens,
        scheduling_policy=scheduling_policy,
        request_priority=request_priority,
        remaining_prefill_tokens=remaining_prefill_tokens,
        max_step_tokens=max_step_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    assert actual == expected


def test_reservation_environment_is_default_off_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = MODULE.TOKEN_BUDGET_PREFILL_RESERVATION_ENV
    monkeypatch.delenv(variable, raising=False)
    assert MODULE.read_priority_prefill_reservation_tokens() == 0

    monkeypatch.setenv(variable, "48")
    assert MODULE.read_priority_prefill_reservation_tokens() == 48

    for invalid in ("-1", "not-an-integer"):
        monkeypatch.setenv(variable, invalid)
        with pytest.raises(ValueError, match=variable):
            MODULE.read_priority_prefill_reservation_tokens()


def test_actual_scheduler_calls_reservation_between_native_passes() -> None:
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "priority_prefill_reservation_tokens"
    ]

    assert len(calls) == 1
    running_marker = source.index("# First, schedule the RUNNING requests.")
    restore_marker = source.index(
        "# Restore selected-prefill capacity immediately before the waiting pass."
    )
    waiting_marker = source.index("# Next, schedule the WAITING requests.")
    call_offset = source.index("priority_prefill_reservation_tokens(")
    assert call_offset < running_marker < restore_marker < waiting_marker
