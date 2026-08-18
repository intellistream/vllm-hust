# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def _manager(*, captured: bool, candidates):
    manager = CudaGraphManager.__new__(CudaGraphManager)
    manager.decode_query_len = 4
    manager._graphs_captured = captured
    manager._candidates = candidates
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    return manager


def test_v2_dispatch_trace_records_spec_shape_hit(monkeypatch):
    selected = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=2,
        uniform_token_count=4,
    )
    manager = _manager(captured=True, candidates={(8, 0): [selected]})
    events = []
    monkeypatch.setattr(
        cudagraph_utils,
        "emit_compilation_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    result = manager.dispatch(2, 8, 4, 0)

    assert result is selected
    assert events == [
        (
            "cudagraph_v2_dispatch",
            {
                "requested": {
                    "num_reqs": 2,
                    "num_tokens": 8,
                    "uniform_token_count": 4,
                    "num_active_loras": 0,
                    "effective_loras": 0,
                    "decode_query_len": 4,
                },
                "selected": {
                    "mode": "FULL",
                    "num_reqs": 2,
                    "num_tokens": 8,
                    "uniform_token_count": 4,
                    "num_active_loras": 0,
                },
                "hit": True,
                "fallback_reason": None,
                "configured_candidate_count": 1,
            },
        )
    ]


def test_v2_dispatch_trace_explains_shape_miss(monkeypatch):
    manager = _manager(captured=True, candidates={})
    events = []
    monkeypatch.setattr(
        cudagraph_utils,
        "emit_compilation_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    result = manager.dispatch(3, 12, 4, 0)

    assert result.cg_mode == CUDAGraphMode.NONE
    assert events[0][0] == "cudagraph_v2_dispatch"
    assert events[0][1]["hit"] is False
    assert events[0][1]["fallback_reason"] == "no_candidate_for_shape"
    assert events[0][1]["requested"]["decode_query_len"] == 4
