# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    OffloadingSpec,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker,
    TransferSpec,
)
from vllm.v1.kv_recovery_profile import (
    MAX_TRANSFER_IDS_PER_WAIT_SET,
    KVRecoveryComputeContext,
    KVRecoveryH2DReceipt,
    KVRecoveryTransferAttempt,
    KVRecoveryTransferContext,
    KVRecoveryWaitAttempt,
    KVRecoveryWaitMembership,
    KVRecoveryWorkerObserver,
)

logger = init_logger(__name__)

_ISSUE19_FAILURE_SENTINEL_ENV = "VLLM_RLP_WORKER_FAILURE_SENTINEL_DIR"
_ISSUE19_FAILURE_EXIT_CODE = 86


class _Issue19WorkerFailureInjector:
    """Default-off watchdog for a deterministic real Worker process exit."""

    def __init__(
        self,
        observer: KVRecoveryWorkerObserver,
        sentinel_dir: Path,
        *,
        exit_function: Callable[[int], None] = os._exit,
        poll_seconds: float = 0.01,
    ) -> None:
        self._observer = observer
        self._sentinel_dir = sentinel_dir
        self._exit_function = exit_function
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self.pid = os.getpid()
        self.trigger_path = sentinel_dir / f"{self.pid}.trigger"
        self.closed_path = sentinel_dir / f"{self.pid}.observer-closed"
        self.pending_arm_path = sentinel_dir / f"{self.pid}.pending-transfer-arm"
        self.pending_witness_path = (
            sentinel_dir / f"{self.pid}.pending-transfer.json"
        )
        self._pending_witness_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._watch,
            name=f"issue19-worker-failure-{self.pid}",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_environment(
        cls, observer: KVRecoveryWorkerObserver | None
    ) -> _Issue19WorkerFailureInjector | None:
        raw = os.environ.get(_ISSUE19_FAILURE_SENTINEL_ENV, "").strip()
        if observer is None or not raw:
            return None
        sentinel_dir = Path(raw)
        if not sentinel_dir.is_absolute() or not sentinel_dir.is_dir():
            logger.error("Invalid Issue #19 worker failure sentinel dir: %s", raw)
            return None
        return cls(observer, sentinel_dir)

    def stop(self) -> None:
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _watch(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            if not self.trigger_path.is_file():
                continue
            try:
                self._observer.prepare_worker_failure()
                self.closed_path.touch(exist_ok=False)
            except Exception:
                logger.exception("Issue #19 worker observer close failed")
            finally:
                self._exit_function(_ISSUE19_FAILURE_EXIT_CODE)
            return

    def pause_if_pending_transfer_is_armed(
        self,
        attempt: KVRecoveryTransferAttempt,
        timestamp_ns: int,
    ) -> None:
        """Expose a submitted transfer, then wait briefly for harness SIGKILL.

        The Issue #19 harness creates the arm file only for its verified Worker
        child. Once a real backend submission is observable, this method
        durably records that exact pending transfer and pauses the calling
        Worker thread. The harness then sends SIGKILL to the same PID. If the
        harness disappears, the bounded wait expires and serving resumes.
        """

        if not self.pending_arm_path.is_file():
            return
        with self._pending_witness_lock:
            if self.pending_witness_path.exists():
                return
            context = attempt.context
            witness = {
                "schema_version": "issue19-pending-transfer-witness/v1",
                "pid": self.pid,
                "timestamp_ns": timestamp_ns,
                "connector_job_id": attempt.connector_job_id,
                "transfer_id": attempt.transfer_id,
                "operation": context.operation,
                "runtime_request_id": context.identity.runtime_request_id,
                "worker_generation": f"VllmWorker-0:{self.pid}",
            }
            try:
                descriptor = os.open(
                    self.pending_witness_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    payload = (json.dumps(witness, sort_keys=True) + "\n").encode()
                    os.write(descriptor, payload)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except Exception:
                logger.exception("Issue #19 pending-transfer witness failed")
                return
        self._stop.wait(30.0)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        kv_recovery_observer: KVRecoveryWorkerObserver | None = None,
    ):
        self.spec = spec
        self.worker = OffloadingWorker()
        self.kv_connector_stats = OffloadingConnectorStats()
        self._kv_recovery_observer = kv_recovery_observer
        self._issue19_failure_injector = _Issue19WorkerFailureInjector.from_environment(
            kv_recovery_observer
        )

        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []
        self._kv_recovery_store_contexts: (
            dict[int, KVRecoveryTransferContext] | None
        ) = {} if kv_recovery_observer is not None else None
        self._kv_recovery_profiled_attempts: (
            dict[int, KVRecoveryTransferAttempt] | None
        ) = {} if kv_recovery_observer is not None else None
        self._kv_recovery_compute_contexts: (
            dict[ReqId, KVRecoveryComputeContext] | None
        ) = {} if kv_recovery_observer is not None else None
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _begin_kv_recovery_transfer(
        self,
        job_id: int,
        context: KVRecoveryTransferContext | None,
    ) -> KVRecoveryTransferAttempt | None:
        observer = self._kv_recovery_observer
        if observer is None or context is None:
            return None
        try:
            attempt = observer.begin_transfer(job_id, context)
            if not isinstance(attempt, KVRecoveryTransferAttempt):
                return None
            if attempt.connector_job_id != job_id or attempt.context != context:
                return None
            return attempt
        except Exception:
            return None

    def _record_kv_recovery_submit(
        self,
        attempt: KVRecoveryTransferAttempt | None,
        timestamp_ns: int,
    ) -> bool:
        observer = self._kv_recovery_observer
        if observer is None or attempt is None:
            return False
        try:
            observer.transfer_submitted(attempt, timestamp_ns)
        except Exception:
            return False
        return True

    def _record_kv_recovery_not_submitted(
        self,
        attempt: KVRecoveryTransferAttempt | None,
    ) -> None:
        observer = self._kv_recovery_observer
        if observer is None or attempt is None:
            return
        try:
            observer.transfer_not_submitted(attempt)
        except Exception:
            return

    def _admit_kv_recovery_submission_evidence(
        self,
        attempt: KVRecoveryTransferAttempt | None,
    ) -> bool:
        if attempt is None:
            return False
        timestamp_ns = self._profile_clock_ns()
        if timestamp_ns is None:
            # The backend submission succeeded, but without its observation
            # timestamp the evidence attempt cannot be accepted. Release the
            # bounded prepared slot while leaving serving unaffected.
            self._record_kv_recovery_not_submitted(attempt)
            return False
        return self._record_kv_recovery_submit(attempt, timestamp_ns)

    def _submit_pending_stores(self) -> None:
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            context = (
                self._kv_recovery_store_contexts.pop(job_id, None)
                if self._kv_recovery_store_contexts is not None
                else None
            )
            attempt = self._begin_kv_recovery_transfer(job_id, context)
            try:
                success = self.worker.transfer_async(job_id, transfer_spec)
            except Exception:
                self._record_kv_recovery_not_submitted(attempt)
                raise
            if not success:
                self._record_kv_recovery_not_submitted(attempt)
            assert success
            if self._admit_kv_recovery_submission_evidence(attempt):
                assert attempt is not None
                assert self._kv_recovery_profiled_attempts is not None
                self._kv_recovery_profiled_attempts[job_id] = attempt
                injector = self._issue19_failure_injector
                if injector is not None:
                    injector.pause_if_pending_transfer_is_armed(
                        attempt,
                        time.monotonic_ns(),
                    )
        self._unsubmitted_store_jobs.clear()

    def _prepare_kv_recovery_wait(
        self, job_ids: set[int]
    ) -> KVRecoveryWaitMembership | None:
        observer = self._kv_recovery_observer
        if observer is None or not job_ids:
            return None
        try:
            observed_job_ids = frozenset(
                job_ids
                if len(job_ids) <= MAX_TRANSFER_IDS_PER_WAIT_SET
                else islice(job_ids, MAX_TRANSFER_IDS_PER_WAIT_SET + 1)
            )
            membership = observer.prepare_wait(observed_job_ids)
            return (
                membership if isinstance(membership, KVRecoveryWaitMembership) else None
            )
        except Exception:
            return None

    def _record_kv_recovery_wait(
        self,
        membership: KVRecoveryWaitMembership,
        entry_timestamp_ns: int,
    ) -> None:
        observer = self._kv_recovery_observer
        if observer is None:
            return
        try:
            observer.wait_completed(
                KVRecoveryWaitAttempt(
                    membership,
                    entry_timestamp_ns,
                )
            )
        except Exception:
            return

    @staticmethod
    def _profile_clock_ns() -> int | None:
        try:
            return time.monotonic_ns()
        except Exception:
            return None

    @staticmethod
    def _device_duration_ns(transfer_time: float | None) -> int | None:
        if (
            transfer_time is None
            or type(transfer_time) not in (int, float)
            or not math.isfinite(transfer_time)
        ):
            return None
        if transfer_time <= 0 or transfer_time > ((2**64 - 1) / 1_000_000_000):
            return None
        try:
            duration_ns = math.floor(transfer_time * 1_000_000_000 + 0.5)
        except OverflowError:
            return None
        return duration_ns if duration_ns <= 2**64 - 1 else None

    def _record_kv_recovery_completion(
        self,
        job_id: int,
        timestamp_ns: int,
        success: bool,
        bytes_moved: int | None,
        transfer_time: float | None,
        attempt: KVRecoveryTransferAttempt,
    ) -> None:
        observer = self._kv_recovery_observer
        if observer is None:
            return
        try:
            receipt = observer.transfer_completed(
                connector_job_id=job_id,
                timestamp_ns=timestamp_ns,
                success=success,
                bytes_moved=bytes_moved,
                device_duration_ns=self._device_duration_ns(transfer_time),
            )
            if receipt is None or not isinstance(receipt, KVRecoveryH2DReceipt):
                return
            context = attempt.context
            if (
                context.operation != "h2d_restore"
                or receipt.connector_job_id != job_id
                or receipt.transfer_id != attempt.transfer_id
                or receipt.identity != context.identity
                or receipt.block_set_id != context.block_set_id
                or receipt.timestamp_ns != timestamp_ns
                or receipt.bytes_moved != bytes_moved
            ):
                return
        except Exception:
            return
        if self._connector_worker_meta.add_kv_recovery_h2d_receipt(receipt):
            return
        try:
            observer.h2d_receipt_capacity_exhausted(
                receipt,
                "serialization_failure",
            )
        except Exception:
            return

    def _register_handlers(self, kv_caches: CanonicalKVCaches):
        for src_cls, dst_cls, handler in self.spec.get_handlers(kv_caches):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(
        self,
        kv_caches: dict[
            str,
            torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        ],
    ):
        num_blocks = self.spec.kv_cache_config.num_blocks

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> per-tensor size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, tuple[int, ...]] = {}
        # layer_name -> per-tensor size of page in bytes
        page_size_bytes: dict[str, tuple[int, ...]] = {}
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    if isinstance(layer_kv_cache, torch.Tensor):
                        assert layer_kv_cache.storage_offset() == 0

                        storage = layer_kv_cache.untyped_storage()
                        page = layer_kv_cache_spec.page_size_bytes
                        tensors_per_block[layer_name] = (
                            torch.tensor(
                                [],
                                dtype=torch.int8,
                                device=layer_kv_cache.device,
                            )
                            .set_(storage)
                            .view(num_blocks, page),
                        )
                        page_size_bytes[layer_name] = (page,)
                        unpadded_page_size_bytes[layer_name] = (
                            layer_kv_cache_spec.real_page_size_bytes,
                        )
                    else:
                        # Ascend exposes K and V as separately allocated,
                        # blocks-outermost tensors. Preserve every component
                        # and ignore alignment padding outside the tensor view.
                        assert isinstance(layer_kv_cache, tuple)
                        assert layer_kv_cache
                        block_tensors = []
                        padded_sizes = []
                        unpadded_sizes = []
                        for tensor in layer_kv_cache:
                            assert isinstance(tensor, torch.Tensor)
                            assert tensor.ndim >= 1
                            assert tensor.shape[0] >= num_blocks
                            assert tensor[0].is_contiguous()

                            element_size = tensor.element_size()
                            padded_size = tensor.stride(0) * element_size
                            unpadded_size = tensor[0].numel() * element_size
                            assert unpadded_size <= padded_size
                            storage_offset = tensor.storage_offset() * element_size
                            block_tensor = torch.empty(
                                0,
                                dtype=torch.int8,
                                device=tensor.device,
                            ).set_(
                                tensor.untyped_storage(),
                                storage_offset,
                                (num_blocks, padded_size),
                                (padded_size, 1),
                            )
                            block_tensors.append(block_tensor)
                            padded_sizes.append(padded_size)
                            unpadded_sizes.append(unpadded_size)

                        assert sum(padded_sizes) == (
                            layer_kv_cache_spec.page_size_bytes
                        )
                        assert sum(unpadded_sizes) == (
                            layer_kv_cache_spec.real_page_size_bytes
                        )
                        tensors_per_block[layer_name] = tuple(block_tensors)
                        page_size_bytes[layer_name] = tuple(padded_sizes)
                        unpadded_page_size_bytes[layer_name] = tuple(unpadded_sizes)

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    state_tensors = kv_caches[layer_name]
                    assert isinstance(state_tensors, list)

                    # re-construct the raw (num_blocks, page_size) tensor
                    # from the first state tensor
                    assert len(state_tensors) > 0
                    first_state_tensor = state_tensors[0]
                    assert first_state_tensor.storage_offset() == 0
                    tensor = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=first_state_tensor.device,
                        )
                        .set_(first_state_tensor.untyped_storage())
                        .view((num_blocks, layer_kv_cache_spec.page_size_bytes))
                    )
                    tensors_per_block[layer_name] = (tensor,)

                    page_size_bytes[layer_name] = (layer_kv_cache_spec.page_size_bytes,)
                    unpadded_page_size_bytes[layer_name] = (
                        replace(
                            layer_kv_cache_spec,
                            page_size_padded=None,
                        ).page_size_bytes,
                    )

                else:
                    raise NotImplementedError

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in self.spec.kv_cache_config.kv_cache_tensors:
            # Filter to layers that were actually processed above.
            # _get_kv_cache_config_deepseek_v4 emits KVCacheTensor entries for
            # every (tuple_idx, page_size) slot; slots where no group has a
            # layer at that index produce an empty shared_by (reserved memory
            # with no corresponding model layer).
            tensor_layer_names = [
                n for n in kv_cache_tensor.shared_by if n in tensors_per_block
            ]
            if not tensor_layer_names:
                continue

            # verify all layers in the group reference the exact same tensors
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
            tensor_count = len(tensors_per_block[tensor_layer_names[0]])
            for tensor_idx in range(tensor_count):
                assert (
                    len(
                        {
                            tensors_per_block[n][tensor_idx].data_ptr()
                            for n in tensor_layer_names
                        }
                    )
                    == 1
                )
                assert (
                    len(
                        {
                            tensors_per_block[n][tensor_idx].stride()
                            for n in tensor_layer_names
                        }
                    )
                    == 1
                )

            # pick the first layer to represent the group
            first_layer_name = tensor_layer_names[0]
            tensor_entries = zip(
                tensors_per_block[first_layer_name],
                page_size_bytes[first_layer_name],
            )
            for data_ref_idx, (tensor, tensor_page_size) in enumerate(tensor_entries):
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=tensor_page_size,
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(
                                unpadded_page_size_bytes[layer_name][data_ref_idx]
                            ),
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._register_handlers(canonical_kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # verify that num_blocks is at physical position 0 in the cross-layers
        # tensor layout.
        test_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
        )
        num_blocks_logical_dim = test_shape.index(1234) + 1
        physical_to_logical = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        num_blocks_physical_dim = physical_to_logical.index(num_blocks_logical_dim)
        assert num_blocks_physical_dim == 0

        kv_cache_groups = self.spec.kv_cache_config.kv_cache_groups
        assert len(kv_cache_groups) == 1
        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        num_layers = len(kv_cache_groups[0].layer_names)
        page_size_bytes = kv_cache_spec.page_size_bytes * num_layers

        assert kv_cache.storage_offset() == 0
        storage = kv_cache.untyped_storage()
        assert len(storage) % page_size_bytes == 0
        num_blocks = len(storage) // page_size_bytes
        tensor = (
            torch.tensor(
                [],
                dtype=torch.int8,
                device=kv_cache.device,
            )
            .set_(storage)
            .view(num_blocks, page_size_bytes)
        )
        kv_cache_tensor = CanonicalKVCacheTensor(
            tensor=tensor, page_size_bytes=page_size_bytes
        )
        # in cross layers layout, there's currently only a single group
        kv_cache_data_ref = CanonicalKVCacheRef(
            tensor_idx=0, page_size_bytes=page_size_bytes
        )
        canonical_kv_caches = CanonicalKVCaches(
            tensors=[kv_cache_tensor], group_data_refs=[[kv_cache_data_ref]]
        )

        self._register_handlers(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        self._submit_pending_stores()

        if kv_connector_metadata.jobs_to_flush:
            wait_membership = self._prepare_kv_recovery_wait(
                kv_connector_metadata.jobs_to_flush
            )
            entry_timestamp_ns = (
                self._profile_clock_ns() if wait_membership is not None else None
            )
            self.worker.wait(kv_connector_metadata.jobs_to_flush)
            if entry_timestamp_ns is not None and wait_membership is not None:
                self._record_kv_recovery_wait(
                    wait_membership,
                    entry_timestamp_ns,
                )

        # A flush is normally a completion fence, not an abandonment. Only
        # consume contexts named by the scheduler's explicit discard handoff,
        # and do so after any required backend wait has completed.
        if kv_connector_metadata.kv_recovery_jobs_to_invalidate:
            observer = self._kv_recovery_observer
            if observer is not None:
                with suppress(Exception):
                    observer.invalidate_transfers(
                        kv_connector_metadata.kv_recovery_jobs_to_invalidate
                    )

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        self._submit_pending_stores()

        if self._kv_recovery_compute_contexts is not None:
            self._fail_unobserved_first_compute()
            self._kv_recovery_compute_contexts = dict(
                metadata.kv_recovery_compute_contexts or {}
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "KV-recovery start_kv_transfers compute_contexts=%s",
                    sorted(self._kv_recovery_compute_contexts),
                )

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            context = None
            if metadata.kv_recovery_contexts is not None:
                context = metadata.kv_recovery_contexts.get(job_id)
            attempt = self._begin_kv_recovery_transfer(job_id, context)
            try:
                success = self.worker.transfer_async(job_id, entry.transfer_spec)
            except Exception:
                self._record_kv_recovery_not_submitted(attempt)
                raise
            if not success:
                self._record_kv_recovery_not_submitted(attempt)
            assert success
            if self._admit_kv_recovery_submission_evidence(attempt):
                assert attempt is not None
                assert self._kv_recovery_profiled_attempts is not None
                self._kv_recovery_profiled_attempts[job_id] = attempt
                injector = self._issue19_failure_injector
                if injector is not None:
                    injector.pause_if_pending_transfer_is_armed(
                        attempt,
                        time.monotonic_ns(),
                    )

    def observe_kv_recovery_first_compute(
        self,
        scheduled_request_ids: Iterable[str],
    ) -> None:
        """Consume the exact admitted sidecars at the worker forward entry."""
        observer = self._kv_recovery_observer
        contexts = self._kv_recovery_compute_contexts
        if observer is None or not contexts:
            return
        try:
            scheduled_request_ids = frozenset(scheduled_request_ids)
        except Exception:
            self._fail_unobserved_first_compute()
            return
        timestamp_ns = self._profile_clock_ns()
        if timestamp_ns is None:
            self._fail_unobserved_first_compute()
            return
        pending = tuple(contexts.items())
        contexts.clear()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KV-recovery first_compute observe: contexts=%s scheduled=%s",
                sorted(k for k, _ in pending),
                sorted(scheduled_request_ids),
            )
        for runtime_request_id, context in pending:
            if (
                runtime_request_id not in scheduled_request_ids
                or context.identity.runtime_request_id != runtime_request_id
            ):
                logger.debug(
                    "KV-recovery first_compute NOT observed for %s (in_scheduled=%s)",
                    runtime_request_id,
                    runtime_request_id in scheduled_request_ids,
                )
                with suppress(Exception):
                    observer.first_compute_not_observed(context)
                continue
            logger.debug(
                "KV-recovery first_compute observed for %s", runtime_request_id
            )
            with suppress(Exception):
                observer.first_compute(context, timestamp_ns)

    def _fail_unobserved_first_compute(self) -> None:
        observer = self._kv_recovery_observer
        contexts = self._kv_recovery_compute_contexts
        if observer is None or not contexts:
            return
        pending = tuple(contexts.values())
        contexts.clear()
        for context in pending:
            with suppress(Exception):
                observer.first_compute_not_observed(context)

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            self._unsubmitted_store_jobs.append((job_id, entry.transfer_spec))
            if (
                self._kv_recovery_store_contexts is not None
                and metadata.kv_recovery_contexts is not None
            ):
                context = metadata.kv_recovery_contexts.get(job_id)
                if context is not None:
                    self._kv_recovery_store_contexts[job_id] = context

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Returns:
            tuple of (finished_sending, finished_recving). Stores never
            emit finished_sending — the scheduler tracks store completion
            via kv_connector_worker_meta.completed_jobs and fences any
            block reuse via jobs_to_flush. Loads still emit
            finished_recving so the base scheduler can resume requests
            blocked on remote KV (and free aborted-during-load reqs).
        """
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            timestamp_ns = (
                self._profile_clock_ns()
                if self._kv_recovery_profiled_attempts is not None
                else None
            )
            attempt = (
                self._kv_recovery_profiled_attempts.pop(job_id, None)
                if self._kv_recovery_profiled_attempts is not None
                else None
            )
            if attempt is not None and timestamp_ns is not None:
                self._record_kv_recovery_completion(
                    job_id=job_id,
                    timestamp_ns=timestamp_ns,
                    success=transfer_result.success,
                    bytes_moved=transfer_result.transfer_size,
                    transfer_time=transfer_result.transfer_time,
                    attempt=attempt,
                )
            if (
                transfer_result.transfer_time
                and transfer_result.transfer_size is not None
                and transfer_result.transfer_type is not None
            ):
                self.kv_connector_stats.record_transfer(
                    num_bytes=transfer_result.transfer_size,
                    time=transfer_result.transfer_time,
                    transfer_type=transfer_result.transfer_type,
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """

        if self.kv_connector_stats.is_empty():
            return None
        # Clear stats for next iteration
        kv_connector_stats = self.kv_connector_stats
        self.kv_connector_stats = OffloadingConnectorStats()
        return kv_connector_stats

    def shutdown(self) -> None:
        if self._issue19_failure_injector is not None:
            self._issue19_failure_injector.stop()
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        if self._kv_recovery_store_contexts is not None:
            self._kv_recovery_store_contexts.clear()
        if self._kv_recovery_profiled_attempts is not None:
            self._kv_recovery_profiled_attempts.clear()
        if self._kv_recovery_compute_contexts is not None:
            self._fail_unobserved_first_compute()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        try:
            self.worker.shutdown()
        finally:
            if self._kv_recovery_observer is not None:
                with suppress(Exception):
                    self._kv_recovery_observer.close()
