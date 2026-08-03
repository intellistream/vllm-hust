# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import replace
from itertools import islice

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
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
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_recovery_profile import (
    MAX_TRANSFER_IDS_PER_WAIT_SET,
    KVRecoveryComputeContext,
    KVRecoveryComputeKind,
    KVRecoveryH2DReceipt,
    KVRecoveryTransferAttempt,
    KVRecoveryTransferContext,
    KVRecoveryWaitAttempt,
    KVRecoveryWaitMembership,
    KVRecoveryWorkerObserver,
)

logger = init_logger(__name__)


def _byte_storage_offset(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.data_ptr() - tensor.untyped_storage().data_ptr())
    except AttributeError:
        return int(tensor.storage_offset() * tensor.element_size())


def _byte_storage_view(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    assert tensor.is_contiguous(), (
        "KV offload expects contiguous KV cache tensors when rebuilding byte views"
    )
    return torch.tensor(
        [],
        dtype=torch.int8,
        device=tensor.device,
    ).set_(tensor.untyped_storage(), _byte_storage_offset(tensor), shape)


def _page_size_from_tensor(tensor: torch.Tensor, num_blocks: int) -> int:
    total_bytes = tensor.numel() * tensor.element_size()
    assert total_bytes % num_blocks == 0
    return total_bytes // num_blocks


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        kv_recovery_observer: KVRecoveryWorkerObserver | None = None,
    ):
        self.spec = spec
        self.worker: OffloadingWorker | None = None
        self._kv_recovery_observer = kv_recovery_observer

        # Preserve the original profiler-off state shapes. Optional identity
        # uses separate tables allocated only when an observer is active.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
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

    def _submit_pending_stores(self) -> None:
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            context = (
                self._kv_recovery_store_contexts.pop(job_id, None)
                if self._kv_recovery_store_contexts is not None
                else None
            )
            attempt = self._begin_kv_recovery_transfer(job_id, context)
            try:
                success = self.worker.submit_store(job_id, src_spec, dst_spec)
            except Exception:
                self._record_kv_recovery_not_submitted(attempt)
                raise
            if not success:
                self._record_kv_recovery_not_submitted(attempt)
            assert success
            if attempt is not None:
                timestamp_ns = self._profile_clock_ns()
                if timestamp_ns is not None and self._record_kv_recovery_submit(
                    attempt, timestamp_ns
                ):
                    assert self._kv_recovery_profiled_attempts is not None
                    self._kv_recovery_profiled_attempts[job_id] = attempt
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
                or receipt.binding != context.binding
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

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        kv_cache_config = self.spec.kv_cache_config
        num_blocks = kv_cache_config.num_blocks

        # Packed layouts (e.g. DSv4) set block_stride > 0; their tensors use
        # stride(0) as the manager-block stride (equals total_num_bytes_per_block).
        # General (non-packed) layouts size the tensor at page_size_bytes per
        # manager block, so page_size_bytes is the correct offloading stride.
        layer_is_packed: dict[str, bool] = {
            ln: bool(kv_tensor.block_stride)
            for kv_tensor in kv_cache_config.kv_cache_tensors
            for ln in kv_tensor.shared_by
        }

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
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
                    if isinstance(layer_kv_cache, (tuple, list)):
                        page_sizes = tuple(
                            _page_size_from_tensor(tensor, num_blocks)
                            for tensor in layer_kv_cache
                        )
                        assert len(set(page_sizes)) == 1, (
                            "Split KV cache tensors must have the same page size "
                            "for canonical offloading refs"
                        )
                        page_size = page_sizes[0]
                        tensors_per_block[layer_name] = tuple(
                            _byte_storage_view(tensor, (num_blocks, page_size))
                            for tensor in layer_kv_cache
                        )
                        page_size_bytes[layer_name] = page_size
                        unpadded_page_size_bytes[layer_name] = page_size
                        continue

                    assert isinstance(layer_kv_cache, torch.Tensor)

                    page = layer_kv_cache_spec.page_size_bytes
                    elem_size = layer_kv_cache.element_size()
                    byte_offset = layer_kv_cache.storage_offset() * elem_size
                    block_stride_bytes = (
                        layer_kv_cache.stride(0) * elem_size
                        if layer_is_packed[layer_name]
                        else page
                    )
                    tensors_per_block[layer_name] = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=layer_kv_cache.device,
                        ).set_(
                            layer_kv_cache.untyped_storage(),
                            byte_offset,
                            (num_blocks, page),
                            (block_stride_bytes, 1),
                        ),
                    )
                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.real_page_size_bytes
                    )

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    state_tensors = kv_caches[layer_name]
                    assert isinstance(state_tensors, list)

                    # re-construct the raw (num_blocks, page_size) tensor
                    # from the first state tensor
                    assert len(state_tensors) > 0
                    first_state_tensor = state_tensors[0]
                    tensor = _byte_storage_view(
                        first_state_tensor,
                        (num_blocks, layer_kv_cache_spec.page_size_bytes),
                    )
                    tensors_per_block[layer_name] = (tensor,)

                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        packed_kv_cache_tensor = next(
            (
                t
                for t in kv_cache_config.kv_cache_tensors
                if t.block_stride and t.shared_by
            ),
            None,
        )
        if packed_kv_cache_tensor is not None:
            (tensor,) = tensors_per_block[packed_kv_cache_tensor.shared_by[0]]
            block_stride = tensor.stride(0)
            packed_tensor = tensor.as_strided(
                (num_blocks, block_stride),
                (block_stride, 1),
                storage_offset=0,
            )
            self._init_worker(
                CanonicalKVCaches(
                    [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                    [
                        [CanonicalKVCacheRef(0, block_stride)]
                        for _ in kv_cache_config.kv_cache_groups
                    ],
                )
            )
            return

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # Filter to layers that were actually processed above.
            # Packed KV allocation emits KVCacheTensor entries for
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
            assert (
                len({tensors_per_block[n][0].data_ptr() for n in tensor_layer_names})
                == 1
            )
            assert (
                len({tensors_per_block[n][0].stride() for n in tensor_layer_names}) == 1
            )

            # pick the first layer to represent the group
            first_layer_name = tensor_layer_names[0]
            for tensor in tensors_per_block[first_layer_name]:
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes[first_layer_name],
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)

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

        self._init_worker(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        self._submit_pending_stores()

        if kv_connector_metadata.jobs_to_flush:
            observer = self._kv_recovery_observer
            if observer is not None:
                with suppress(Exception):
                    observer.invalidate_transfers(
                        frozenset(kv_connector_metadata.jobs_to_flush)
                    )
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

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        self._submit_pending_stores()

        if self._kv_recovery_compute_contexts is not None:
            self._fail_unobserved_first_compute()
            self._kv_recovery_compute_contexts = dict(
                metadata.kv_recovery_compute_contexts or {}
            )

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            context = None
            if metadata.kv_recovery_contexts is not None:
                context = metadata.kv_recovery_contexts.get(job_id)
            attempt = self._begin_kv_recovery_transfer(job_id, context)
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            try:
                success = self.worker.submit_load(
                    job_id, entry.src_spec, entry.dst_spec
                )
            except Exception:
                self._record_kv_recovery_not_submitted(attempt)
                raise
            if not success:
                self._record_kv_recovery_not_submitted(attempt)
            assert success
            if attempt is not None:
                timestamp_ns = self._profile_clock_ns()
                if timestamp_ns is not None and self._record_kv_recovery_submit(
                    attempt, timestamp_ns
                ):
                    assert self._kv_recovery_profiled_attempts is not None
                    self._kv_recovery_profiled_attempts[job_id] = attempt

    def observe_kv_recovery_first_compute(
        self,
        runtime_request_id: str,
        recovery_epoch: int,
        timestamp_ns: int,
        compute_kind: KVRecoveryComputeKind,
        base_event_id: str,
    ) -> None:
        """Consume one exact admitted-roster sidecar at worker forward entry."""
        observer = self._kv_recovery_observer
        contexts = self._kv_recovery_compute_contexts
        if observer is None or contexts is None:
            return
        context = contexts.get(runtime_request_id)
        if context is None or context.identity.recovery_epoch != recovery_epoch:
            return
        contexts.pop(runtime_request_id, None)
        with suppress(Exception):
            observer.first_compute(
                context,
                timestamp_ns,
                compute_kind,
                base_event_id,
            )

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
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )
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
        assert self.worker is not None
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
            is_load = job_id in self._load_jobs
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
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
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

    def shutdown(self) -> None:
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
            if self.worker is not None:
                self.worker.shutdown()
        finally:
            if self._kv_recovery_observer is not None:
                with suppress(Exception):
                    self._kv_recovery_observer.close()
