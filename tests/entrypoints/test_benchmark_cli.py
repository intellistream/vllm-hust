# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `vllm bench` CLI subcommand."""

import argparse
import sys
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.benchmark.main import BenchmarkSubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


def _find_subparsers_action(
    parser: FlexibleArgumentParser,
) -> argparse._SubParsersAction:
    return next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )


def test_bench_subparser_imports_modules_when_invoked():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")

    with patch.object(sys, "argv", ["vllm", "bench"]):
        bench_parser = BenchmarkSubcommand().subparser_init(subparsers)

    bench_subparsers = _find_subparsers_action(bench_parser)
    assert "sweep" in bench_subparsers.choices
    assert "serve" in bench_subparsers.choices


def test_bench_subparser_skips_nested_imports_when_not_invoked(monkeypatch):
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")

    imported = False

    def fake_import() -> None:
        nonlocal imported
        imported = True

    monkeypatch.setattr(
        "vllm.entrypoints.cli.benchmark.main._import_bench_subcommand_modules",
        fake_import,
    )

    with patch.object(sys, "argv", ["vllm", "serve"]):
        BenchmarkSubcommand().subparser_init(subparsers)

    assert not imported
