# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static AST checks for issue #163: verify that gpu_model_runner guards
all Knorm code paths with _knorm_active / _knorm_wrapper_installed so no
half-enabled state can occur."""
from __future__ import annotations

import ast
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parents[2] / "vllm/v1/worker/gpu_model_runner.py"


def _source() -> str:
    return RUNNER_PATH.read_text(encoding="utf-8")


def _tree() -> ast.AST:
    return ast.parse(_source())


def _attr_name(node) -> str:
    """Extract attribute name from Assign or AnnAssign target."""
    if isinstance(node, ast.AnnAssign):
        return node.target.attr if isinstance(node.target, ast.Attribute) else ""
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Attribute):
                return t.attr
    return ""


def test_knorm_active_flag_defined_in_init():
    """_knorm_active must be computed via should_activate() from
    vllm.knorm.config, the single source of truth shared with
    register_all_kvcache_specs."""
    tree = _tree()
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    init = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    active_assigns = [
        n for n in ast.walk(init)
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and _attr_name(n) == "_knorm_active"
    ]
    assert len(active_assigns) == 1, (
        "_knorm_active must be assigned exactly once in __init__"
    )

    assert active_assigns[0].value is not None, (
        "_knorm_active assignment must have a value"
    )
    src = ast.unparse(active_assigns[0].value)
    # Must call should_activate (single source of truth), not inline
    # envs checks that could drift from register_all_kvcache_specs.
    assert "should_activate" in src, (
        "_knorm_active must use should_activate() from vllm.knorm.config"
    )


def test_wrapper_install_guarded_by_knorm_active():
    """install_ascend_wrapper call must be guarded by self._knorm_active."""
    src = _source()
    assert "install_ascend_wrapper" in src
    assert "self._knorm_active and not self._knorm_wrapper_installed" in src, (
        "install_ascend_wrapper must be guarded by _knorm_active"
    )


def test_collect_knorm_scores_guarded_by_wrapper_installed():
    """collect_knorm_scores call must be guarded by _knorm_wrapper_installed."""
    src = _source()
    assert "collect_knorm_scores" in src
    assert "self._knorm_wrapper_installed and get_pp_group" in src, (
        "collect_knorm_scores must be guarded by _knorm_wrapper_installed"
    )


def test_attach_knorm_scores_guarded_by_wrapper_installed():
    """attach_knorm_scores call must be guarded by _knorm_wrapper_installed."""
    src = _source()
    assert "attach_knorm_scores" in src
    assert "if self._knorm_wrapper_installed:" in src, (
        "attach_knorm_scores must be guarded by _knorm_wrapper_installed"
    )


def test_no_unguarded_knorm_score_getattr():
    """getattr(self, '_knorm_scores') must only appear inside a
    _knorm_wrapper_installed guard."""
    src = _source()
    lines = src.splitlines()
    in_guard = False
    found_guarded_getattr = False
    for line in lines:
        stripped = line.strip()
        if stripped == "if self._knorm_wrapper_installed:":
            in_guard = True
            continue
        if in_guard and "knorm_scores = getattr(self" in stripped:
            found_guarded_getattr = True
            break
    assert found_guarded_getattr, (
        "_knorm_scores getattr must be inside _knorm_wrapper_installed guard"
    )
