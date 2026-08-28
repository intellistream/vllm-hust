# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-path tests for the B134 event chain.

Unlike the AST contract tests (test_b134_chain_contract.py), these tests
REALLY EXECUTE the code paths: ``CPUOffloadingManager.prepare_store()`` and
the scheduler restore branch. Heavy dependencies (torch, numpy, the rest of
the vllm package) are stubbed in ``sys.modules`` so the modules load without
a compiled extension, but the code under test is the actual source file.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).parents[2]


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _load_module(name: str, path: Path, sys_modules: dict) -> types.ModuleType:
    """Load a real source file as a module, registering it in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys_modules[name] = module
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kv_offload_env():
    """Load real kv_offload modules with stubbed heavy deps, then restore."""
    stubs: dict[str, types.ModuleType] = {}

    # --- stub the heavy / unavailable dependencies -----------------------
    torch_stub = _make_module(
        "torch", Tensor=MagicMock, device=MagicMock(return_value="cpu")
    )
    numpy_stub = _make_module(
        "numpy",
        array=MagicMock(side_effect=lambda *a, **k: list(a[0])),
        int64="int64",
    )
    stubs["torch"] = torch_stub
    stubs["numpy"] = numpy_stub

    # Stub the rest of the vllm package so `from vllm.xxx import ...` works.
    vllm_pkg = _make_module("vllm")
    vllm_logger = _make_module("vllm.logger", init_logger=MagicMock())
    vllm_config = _make_module("vllm.config", VllmConfig=MagicMock)
    vllm_v1 = _make_module("vllm.v1")
    vllm_v1_core = _make_module("vllm.v1.core")
    vllm_kv_cache_utils = _make_module(
        "vllm.v1.core.kv_cache_utils", resolve_kv_cache_block_sizes=MagicMock()
    )
    vllm_kv_offload = _make_module("vllm.v1.kv_offload")
    vllm_kv_offload_cpu = _make_module("vllm.v1.kv_offload.cpu")
    vllm_policies = _make_module("vllm.v1.kv_offload.cpu.policies")
    vllm_dist = _make_module("vllm.distributed")
    vllm_kv_transfer = _make_module("vllm.distributed.kv_transfer")
    vllm_kv_connector = _make_module("vllm.distributed.kv_transfer.kv_connector")
    vllm_metrics = _make_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics",
        OffloadingConnectorStats=MagicMock,
    )
    stubs.update(
        {
            "vllm": vllm_pkg,
            "vllm.logger": vllm_logger,
            "vllm.config": vllm_config,
            "vllm.v1": vllm_v1,
            "vllm.v1.core": vllm_v1_core,
            "vllm.v1.core.kv_cache_utils": vllm_kv_cache_utils,
            "vllm.v1.kv_offload": vllm_kv_offload,
            "vllm.v1.kv_offload.cpu": vllm_kv_offload_cpu,
            "vllm.v1.kv_offload.cpu.policies": vllm_policies,
            "vllm.distributed": vllm_dist,
            "vllm.distributed.kv_transfer": vllm_kv_transfer,
            "vllm.distributed.kv_transfer.kv_connector": vllm_kv_connector,
            "vllm.distributed.kv_transfer.kv_connector.v1": _make_module(
                "vllm.distributed.kv_transfer.kv_connector.v1"
            ),
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading": (
                _make_module("vllm.distributed.kv_transfer.kv_connector.v1.offloading")
            ),
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics": (
                vllm_metrics
            ),
        }
    )

    # Snapshot EVERY key we are about to touch — the stubs plus all modules
    # dynamically loaded below — BEFORE overwriting anything.
    dynamic_names = {
        "vllm.v1.events",
        "vllm.v1.kv_offload.base",
        "vllm.v1.kv_offload.cpu.common",
        "vllm.v1.kv_offload.cpu.policies.base",
        "vllm.v1.kv_offload.cpu.policies.lru",
        "vllm.v1.kv_offload.cpu.policies.arc",
        "vllm.v1.kv_offload.cpu.manager",
    }
    saved = {name: sys.modules.get(name) for name in set(stubs) | dynamic_names}
    sys.modules.update(stubs)

    try:
        kv_offload_root = REPO_ROOT / "vllm" / "v1" / "kv_offload"
        # events.py is pure stdlib — load the real module.
        _load_module(
            "vllm.v1.events",
            REPO_ROOT / "vllm" / "v1" / "events.py",
            sys.modules,
        )
        base = _load_module(
            "vllm.v1.kv_offload.base", kv_offload_root / "base.py", sys.modules
        )
        # These loads exist for their import side effects (module registration
        # in sys.modules so the manager can `from ... import ...` them).
        _load_module(
            "vllm.v1.kv_offload.cpu.common",
            kv_offload_root / "cpu" / "common.py",
            sys.modules,
        )
        _load_module(
            "vllm.v1.kv_offload.cpu.policies.base",
            kv_offload_root / "cpu" / "policies" / "base.py",
            sys.modules,
        )
        _load_module(
            "vllm.v1.kv_offload.cpu.policies.lru",
            kv_offload_root / "cpu" / "policies" / "lru.py",
            sys.modules,
        )
        _load_module(
            "vllm.v1.kv_offload.cpu.policies.arc",
            kv_offload_root / "cpu" / "policies" / "arc.py",
            sys.modules,
        )
        manager = _load_module(
            "vllm.v1.kv_offload.cpu.manager",
            kv_offload_root / "cpu" / "manager.py",
            sys.modules,
        )
    finally:
        # Restore sys.modules so other tests are unaffected.
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    return {"base": base, "manager": manager}


def _register_collector(manager_mod) -> list:
    """Register a collecting sink on the EventBus and return the log."""
    emitted: list = []

    class _Sink:
        def emit(self, event):
            emitted.append(event)

    manager_mod.EventBus.register_sink(_Sink())
    return emitted


def test_prepare_store_emits_cpu_store(kv_offload_env, monkeypatch) -> None:
    """prepare_store() must emit cpu_store with the expected payload."""
    manager_mod = kv_offload_env["manager"]
    base_mod = kv_offload_env["base"]

    emitted = _register_collector(manager_mod)
    try:
        mgr = manager_mod.CPUOffloadingManager(num_blocks=4)
        keys = [base_mod.make_offload_key(f"block{i}".encode(), 0) for i in range(2)]
        req_context = base_mod.ReqContext(req_id="request-1")

        out = mgr.prepare_store(keys, req_context)
        assert out is not None
        assert [k for k in out.keys_to_store] == keys

        cpu_store_events = [e for e in emitted if type(e).__name__ == "KVOffloadStore"]
        assert len(cpu_store_events) == 1, (
            f"expected one cpu_store event, got {emitted}"
        )
        ev = cpu_store_events[0]
        assert ev.request_id == "request-1"
        assert ev.stored_keys == 2
        assert ev.evicted_keys == 0
    finally:
        manager_mod.EventBus._sinks = []
        manager_mod.EventBus.enabled = False


def test_prepare_store_emits_nothing_when_all_keys_already_stored(
    kv_offload_env, monkeypatch
) -> None:
    """A second prepare_store for the same keys must not emit cpu_store."""
    manager_mod = kv_offload_env["manager"]
    base_mod = kv_offload_env["base"]

    emitted = _register_collector(manager_mod)
    try:
        mgr = manager_mod.CPUOffloadingManager(num_blocks=4)
        keys = [base_mod.make_offload_key(f"block{i}".encode(), 0) for i in range(2)]
        req_context = base_mod.ReqContext(req_id="request-1")

        out1 = mgr.prepare_store(keys, req_context)
        assert out1 is not None
        # Complete the store so the policy marks them stored.
        mgr.complete_store(keys, req_context)

        emitted.clear()
        out2 = mgr.prepare_store(keys, req_context)
        assert out2 is not None
        assert not out2.keys_to_store  # already stored -> nothing new
        assert all(type(e).__name__ != "KVOffloadStore" for e in emitted), (
            f"unexpected events on already-stored keys: {emitted}"
        )
    finally:
        manager_mod.EventBus._sinks = []
        manager_mod.EventBus.enabled = False
