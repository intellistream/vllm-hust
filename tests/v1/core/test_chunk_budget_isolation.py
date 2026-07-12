import runpy
from pathlib import Path


def _cap_scheduled_tokens():
    module = runpy.run_path(
        Path(__file__).parents[3] / "vllm/v1/core/sched/prefill_budget.py"
    )
    return module["cap_scheduled_tokens"]


def test_chunk_budget_isolates_decode_from_long_prefill():
    cap = _cap_scheduled_tokens()
    assert cap(1, 512, 2048) == 1
    assert cap(4096, 512, 2048) == 512
    assert cap(4096, 0, 2048) == 2048
    assert cap(128, 512, 64) == 64
