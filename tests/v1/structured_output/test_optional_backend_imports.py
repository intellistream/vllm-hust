# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib
import sys


def test_structured_output_import_does_not_require_xgrammar(monkeypatch):
    real_import = builtins.__import__
    structured_output_module = sys.modules.pop("vllm.v1.structured_output", None)
    backend_xgrammar_module = sys.modules.pop(
        "vllm.v1.structured_output.backend_xgrammar", None
    )

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "xgrammar":
            raise ModuleNotFoundError("No module named 'xgrammar'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    try:
        module = importlib.import_module("vllm.v1.structured_output")
        assert hasattr(module, "StructuredOutputManager")
        assert "vllm.v1.structured_output.backend_xgrammar" not in sys.modules
    finally:
        sys.modules.pop("vllm.v1.structured_output", None)
        sys.modules.pop("vllm.v1.structured_output.backend_xgrammar", None)
        if structured_output_module is not None:
            sys.modules["vllm.v1.structured_output"] = structured_output_module
        if backend_xgrammar_module is not None:
            sys.modules["vllm.v1.structured_output.backend_xgrammar"] = (
                backend_xgrammar_module
            )
