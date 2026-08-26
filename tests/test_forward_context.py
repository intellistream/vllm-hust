# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)


def _context(marker: str) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"marker": marker},
    )


def test_forward_context_nested_exception_restores_previous_value():
    outer = _context("outer")
    inner = _context("inner")

    assert not is_forward_context_available()
    with override_forward_context(outer):
        assert get_forward_context() is outer
        with (
            pytest.raises(RuntimeError, match="expected"),
            override_forward_context(inner),
        ):
            assert get_forward_context() is inner
            raise RuntimeError("expected")
        assert get_forward_context() is outer
    assert not is_forward_context_available()


def test_forward_context_is_isolated_between_threads():
    main_context = _context("main")
    worker_context = _context("worker")

    def run_worker() -> tuple[bool, bool, bool]:
        unavailable_before = not is_forward_context_available()
        with override_forward_context(worker_context):
            owns_worker_context = get_forward_context() is worker_context
        unavailable_after = not is_forward_context_available()
        return unavailable_before, owns_worker_context, unavailable_after

    with override_forward_context(main_context):
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker_result = executor.submit(run_worker).result()
        assert get_forward_context() is main_context

    assert worker_result == (True, True, True)
    assert not is_forward_context_available()