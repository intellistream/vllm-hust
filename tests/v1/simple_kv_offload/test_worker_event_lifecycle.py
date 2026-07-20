# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


class _QueuedBackend:
    """Record submissions without consuming their wait events."""

    def __init__(self) -> None:
        self.wait_events = []

    def launch_copy(
        self,
        src_blocks,
        dst_blocks,
        is_store,
        event_idx,
        events_list,
        wait_event=None,
    ) -> None:
        assert is_store
        self.wait_events.append(wait_event)


def test_delayed_store_submissions_keep_distinct_compute_events() -> None:
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    backend = _QueuedBackend()
    worker._backend = backend

    compute_stream = object()
    compute_events = [MagicMock(name="step_1"), MagicMock(name="step_2")]
    with (
        patch(
            "vllm.v1.simple_kv_offload.worker.torch.Event",
            side_effect=compute_events,
        ),
        patch(
            "vllm.v1.simple_kv_offload.worker.torch.cuda.current_stream",
            return_value=compute_stream,
        ),
    ):
        for event_idx in (1, 2):
            worker._connector_metadata = SimpleCPUOffloadMetadata(
                store_event=event_idx,
                store_gpu_blocks=[event_idx],
                store_cpu_blocks=[event_idx],
            )
            worker.get_finished(set())

    assert backend.wait_events == compute_events
    assert backend.wait_events[0] is not backend.wait_events[1]
    for event in compute_events:
        event.record.assert_called_once_with(compute_stream)
