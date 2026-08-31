from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.skip_global_cleanup


class _Scheduler:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


_SCHEDULER_MODULE = ModuleType("vllm.v1.core.sched.scheduler")
_SCHEDULER_MODULE.Scheduler = _Scheduler
sys.modules[_SCHEDULER_MODULE.__name__] = _SCHEDULER_MODULE

_SOURCE = (
    Path(__file__).parents[3]
    / "vllm"
    / "v1"
    / "core"
    / "sched"
    / "sarathi_scheduler.py"
)
_SPEC = importlib.util.spec_from_file_location("sarathi_scheduler_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
SarathiSchedulerPort = _MODULE.SarathiSchedulerPort


def _install_scheduler_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunked_prefill: bool,
    token_budget: int,
) -> None:
    def _initialize(instance: _Scheduler, *_args: object, **_kwargs: object) -> None:
        instance.scheduler_config = SimpleNamespace(
            enable_chunked_prefill=chunked_prefill
        )
        instance.max_num_scheduled_tokens = token_budget

    monkeypatch.setattr(_Scheduler, "__init__", _initialize)


def test_sarathi_port_requires_chunked_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_scheduler_stub(
        monkeypatch,
        chunked_prefill=False,
        token_budget=512,
    )

    with pytest.raises(ValueError, match="requires chunked prefill"):
        SarathiSchedulerPort()


def test_sarathi_port_requires_frozen_chunk_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_scheduler_stub(
        monkeypatch,
        chunked_prefill=True,
        token_budget=1024,
    )

    with pytest.raises(ValueError, match="max-num-batched-tokens=512"):
        SarathiSchedulerPort()


def test_sarathi_port_emits_component_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_scheduler_stub(
        monkeypatch,
        chunked_prefill=True,
        token_budget=512,
    )

    scheduler = SarathiSchedulerPort()

    assert scheduler.get_baseline_scheduler_receipt() == {
        "scheduler_type": "sarathi",
        "chunk_size": 512,
        "dynamic_chunking_schedule": False,
        "decode_first": True,
    }
