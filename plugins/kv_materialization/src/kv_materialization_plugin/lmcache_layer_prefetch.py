"""Experimental bounded device prefetch for LMCache's Ascend layerwise path.

This module is deliberately an instance-local patch: it is enabled only by
the project's Formula connector and never changes the installed LMCache or
lmcache-ascend packages.  It preserves the required dependency ``T_l -> C_l``
by recording an event after each layer's materialization and waiting for that
event immediately before that layer's attention.  Unlike the upstream
implementation, it does *not* wait for the whole load stream at every layer.
"""

from __future__ import annotations

import json
import os
import time
from types import MethodType
from typing import Any, Generator, Optional, Union

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import MemoryFormat
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)


def _audit(connector: Any, event: str, **fields: Any) -> None:
    path = getattr(connector, "_kvmat_audit_path", None)
    if not path:
        return
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\n")


def install_layer_prefetch(engine: Any, window: int) -> bool:
    """Install bounded, event-based layer prefetch on one LMCache engine.

    A window of one is native behaviour and is intentionally left untouched.
    ``window`` is capped by the model's number of layers.  The patch is
    instance-local and idempotent, which makes it safe for worker restarts.
    """
    if window < 2:
        return False
    connector = getattr(engine, "gpu_connector", None)
    if connector is None or not hasattr(connector, "load_stream"):
        logger.info("KVMaterialization device prefetch unavailable: connector=%s", type(connector).__name__)
        return False

    engine._kvmat_prefetch_window = min(int(window), int(engine.num_layers))
    engine._kvmat_prefetch_installed = True
    connector._kvmat_prefetch_window = engine._kvmat_prefetch_window
    connector._kvmat_prefetch_installed = True
    connector._kvmat_audit_path = getattr(engine, "_kvmat_audit_path", None)
    connector._kvmat_layer_events = []
    connector._kvmat_staging_buffer = None
    engine.retrieve_layer = MethodType(_retrieve_layer_prefetch, engine)
    connector.batched_to_gpu = MethodType(_batched_to_gpu_prefetch, connector)
    logger.info("KVMaterialization device prefetch installed: W=%s", engine._kvmat_prefetch_window)
    return True


def wait_for_prefetched_layer(engine: Any, layer: int) -> bool:
    """Add only this layer's readiness edge to the current compute stream."""
    connector = getattr(engine, "gpu_connector", None)
    events = getattr(connector, "_kvmat_layer_events", ()) if connector else ()
    if layer >= len(events):
        return False
    torch.npu.current_stream().wait_event(events[layer])
    _audit(connector, "prefetch_compute_wait", layer=layer, ts_ns=time.perf_counter_ns())
    return True


def release_prefetch_staging(engine: Any) -> None:
    """Keep the request-independent staging buffer alive for safe reuse.

    The last event is only enqueued when this function is reached; Python must
    not return the buffer to LMCache's allocator before that event executes.
    We therefore retain one bounded buffer for the connector lifetime rather
    than performing an unsafe host-side free.  A production implementation
    should return it through an event-aware pool at request completion.
    """
    return None


def _batched_to_gpu_prefetch(self: Any, starts: list[int], ends: list[int], **kwargs):
    """Queue loads on ``load_stream`` and record one readiness event per layer."""
    _audit(self, "prefetch_consumer_enter", window=self._kvmat_prefetch_window)
    self.initialize_kvcaches_ptr(**kwargs)
    assert self.kvcaches is not None
    if "slot_mapping" not in kwargs:
        raise ValueError("'slot_mapping' should be provided in kwargs.")

    slot_mapping: torch.Tensor = kwargs["slot_mapping"]
    self._lazy_initialize_buffer(self.kvcaches)
    slot_mapping_full = torch.cat(
        [slot_mapping[start:end] for start, end in zip(starts, ends, strict=False)],
        dim=0,
    )
    chunk_sizes = [end - start for start, end in zip(starts, ends, strict=False)]
    chunk_offsets: list[int] = []
    offset = 0
    for size in chunk_sizes:
        chunk_offsets.append(offset)
        offset += size

    staging = None
    if self.use_gpu:
        required_elements = self.get_shape(len(slot_mapping_full)).numel()
        staging = getattr(self, "_kvmat_staging_buffer", None)
        if staging is None or staging.tensor is None or staging.tensor.numel() < required_elements:
            staging = self.gpu_buffer_allocator.allocate(
                self.get_shape(len(slot_mapping_full)), self.dtype, MemoryFormat.KV_T2D
            )
            assert staging is not None and staging.tensor is not None
            self._kvmat_staging_buffer = staging

    self._kvmat_layer_events = []
    for layer_id in range(self.num_layers):
        memory_objs_layer = yield
        with torch.npu.stream(self.load_stream):
            if self.use_gpu:
                cpu_tensors = []
                for memory_obj in memory_objs_layer:
                    assert memory_obj.tensor is not None
                    assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                    cpu_tensors.append(memory_obj.tensor)
                lmc_ops.batched_fused_single_layer_kv_transfer(
                    cpu_tensors,
                    staging.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    chunk_offsets,
                    chunk_sizes,
                    False,
                    self.kv_format.value,
                    True,
                    self.vllm_two_major,
                )
            else:
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping[start:end],
                        False,
                        self.kv_format.value,
                        True,
                        self.vllm_two_major,
                    )
            event = torch.npu.Event()
            event.record()
        self._kvmat_layer_events.append(event)
        _audit(self, "prefetch_layer_enqueued", layer=layer_id, ts_ns=time.perf_counter_ns())
    _audit(self, "prefetch_consumer_complete", events=len(self._kvmat_layer_events))

    yield


@torch.inference_mode()
def _retrieve_layer_prefetch(
    self: Any,
    tokens: Union[torch.Tensor, list[int]],
    mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Generator[Optional[torch.Tensor], None, None]:
    """Submit all CPU reads, while keeping at most W device loads queued."""
    _audit(self.gpu_connector, "prefetch_retrieve_enter", window=self._kvmat_prefetch_window)
    if not self.is_healthy():
        yield torch.zeros(len(tokens), dtype=torch.bool)
        return
    assert self.storage_manager is not None and self.gpu_connector is not None

    ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
    starts: list[int] = []
    ends: list[int] = []
    keys: list[list[CacheEngineKey]] = []
    request_configs = kwargs.get("request_configs")
    location = None
    for start, end, key in self.token_database.process_tokens(
        tokens=tokens, mask=mask, request_configs=request_configs
    ):
        assert isinstance(key, CacheEngineKey)
        keys_multi_layer = key.split_layers(self.num_layers)
        current_location = self.storage_manager.contains(
            keys_multi_layer[0], self.retrieve_locations
        )
        if not current_location:
            break
        if location is None:
            location = current_location
        else:
            assert location == current_location
        starts.append(start)
        ends.append(end)
        keys.append(keys_multi_layer)
        ret_mask[start:end] = True

    if not keys:
        for _ in range(self.num_layers + 2):
            yield None
        return

    keys_layer_major = [list(row) for row in zip(*keys, strict=False)]
    get_generator = self.storage_manager.layerwise_batched_get(
        keys_layer_major, location=location
    )
    assert_layerwise_gpu_connector(self.gpu_connector)
    consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
    next(consumer)

    # Iterating now only *submits* storage coroutines; it does not block on
    # their results.  This prevents CPU retrieval from serializing layers.
    tasks = [next(get_generator) for _ in range(self.num_layers)]
    next_to_launch = 0
    retained: list[Any] = []

    def launch_one() -> None:
        nonlocal next_to_launch
        memory_objs = tasks[next_to_launch].result()
        consumer.send(memory_objs)
        retained.extend(memory_objs)
        next_to_launch += 1

    for _ in range(min(self._kvmat_prefetch_window, self.num_layers)):
        launch_one()

    # vLLM primes the layerwise generator twice before attention layer 0.
    yield torch.sum(ret_mask)
    yield None

    # On the hook for layer i, its event has already been added to the compute
    # stream.  Queue one future transfer afterwards, so that wait depends on
    # T_i rather than on all work currently in the load stream.
    for layer in range(self.num_layers):
        if next_to_launch < self.num_layers:
            launch_one()
        if layer == self.num_layers - 1:
            # Do not return CPU memory objects to LMCache's pool here: their
            # H2D commands may still be outstanding on the load stream.  The
            # short-lived experimental server releases them at process exit.
            # Production uses an event-aware host-pinned buffer pool.
            release_prefetch_staging(self)
            yield ret_mask
        else:
            yield None
