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
    manager.decode_query_len = 2
    manager._graphs_captured = captured
    manager._candidates = candidates
    return manager


def test_v2_dispatch_trace_records_spec_shape_hit(monkeypatch):
    selected = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=2,
        uniform_token_count=2,
    )
    manager = _manager(captured=True, candidates=[[], [], [], [], [selected]])
    events = []
    monkeypatch.setattr(
        cudagraph_utils,
        "emit_compilation_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    result = manager.dispatch(2, 4, 2)

    assert result is selected
    assert events == [
        (
            "cudagraph_v2_dispatch",
            {
                "requested": {
                    "num_reqs": 2,
                    "num_tokens": 4,
                    "uniform_token_count": 2,
                    "decode_query_len": 2,
                },
                "selected": {
                    "mode": "FULL",
                    "num_reqs": 2,
                    "num_tokens": 4,
                    "uniform_token_count": 2,
                },
                "hit": True,
                "fallback_reason": None,
                "configured_candidate_count": 1,
            },
        )
    ]


def test_v2_dispatch_trace_explains_shape_miss(monkeypatch):
    manager = _manager(captured=True, candidates=[[], [], []])
    events = []
    monkeypatch.setattr(
        cudagraph_utils,
        "emit_compilation_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    result = manager.dispatch(3, 6, 2)

    assert result.cg_mode == CUDAGraphMode.NONE
    assert events[0][0] == "cudagraph_v2_dispatch"
    assert events[0][1]["hit"] is False
    assert events[0][1]["fallback_reason"] == "no_candidate_for_shape"
    assert events[0][1]["requested"]["decode_query_len"] == 2
