# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level contract tests for the B134 six-event chain ordering.

These tests do not import vLLM runtime modules (no torch/tblib needed): they
parse the source with ``ast`` and assert the ordering contract structurally,
so a future edit that regresses the event order, moves an emit past an early
return, or re-gates an emit on ``log_stats`` is caught in a plain pytest
environment.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SCHEDULER = REPO_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
CPU_MANAGER = REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cpu" / "manager.py"
TIERING_MANAGER = REPO_ROOT / "vllm" / "v1" / "kv_offload" / "tiering" / "manager.py"


def _emit_calls(tree: ast.AST) -> list[tuple[str, int]]:
    """Return (event_name, lineno) for every ``emit("event", ...)`` call."""
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "emit"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            calls.append((node.args[0].value, node.lineno))
    return calls


def _is_gated_on_log_stats(tree: ast.AST, emit_lineno: int) -> bool:
    """True if the emit at emit_lineno sits inside an ``if self.log_stats``."""
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "log_stats"
        ):
            continue
        end_lineno = node.end_lineno
        if end_lineno is None:
            # Defensive: ast.If always has end_lineno on Python >= 3.8.
            continue
        if node.lineno < emit_lineno <= end_lineno:
            return True
    return False


def test_restore_chain_order_in_scheduler_source() -> None:
    """Restore requests must emit wakeup -> admission -> scheduled."""
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    events = dict(_emit_calls(tree))
    assert events["wakeup"] < events["admission"] < events["scheduled"]


def test_b134_emits_are_not_gated_on_log_stats() -> None:
    """preempt/scheduled/admission/wakeup must not depend on log_stats."""
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    events = dict(_emit_calls(tree))
    for event in ("preempt", "wakeup", "admission", "scheduled"):
        assert not _is_gated_on_log_stats(tree, events[event]), (
            f"{event} emit at line {events[event]} is gated on log_stats"
        )


def test_cpu_store_emit_precedes_return_in_prepare_store() -> None:
    """cpu_store emit must be reachable, i.e. before any return."""
    tree = ast.parse(CPU_MANAGER.read_text(encoding="utf-8"))
    events = dict(_emit_calls(tree))
    cpu_store_line = events["cpu_store"]

    prepare_store = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "prepare_store"
        ):
            prepare_store = node
            break
    assert prepare_store is not None, "prepare_store not found"

    first_return = min(
        stmt.lineno for stmt in prepare_store.body if isinstance(stmt, ast.Return)
    )
    assert cpu_store_line < first_return, (
        "cpu_store emit is unreachable: it appears after the return at "
        f"line {first_return}"
    )


def test_restore_start_precedes_restore_done_in_tiering_manager() -> None:
    """restore_start must be emitted before restore_done."""
    tree = ast.parse(TIERING_MANAGER.read_text(encoding="utf-8"))
    events = dict(_emit_calls(tree))
    assert events["restore_start"] < events["restore_done"]
