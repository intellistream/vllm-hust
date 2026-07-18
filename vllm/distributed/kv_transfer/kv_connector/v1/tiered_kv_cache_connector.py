# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiered KV restore connector built on the native CPU offload path."""

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

RestoreCandidateProvider = Callable[
    [Any, int, int, int, str | None], Mapping[str, Any]
]
_restore_candidate_provider: RestoreCandidateProvider | None = None
RestoreEvidenceSink = Callable[[Mapping[str, Any]], None]
_restore_evidence_sink: RestoreEvidenceSink | None = None


def register_restore_candidate_provider(
    provider: RestoreCandidateProvider | None,
) -> None:
    """Register a scheduler-process provider owned by the parent plugin."""
    global _restore_candidate_provider
    _restore_candidate_provider = provider


def register_restore_evidence_sink(sink: RestoreEvidenceSink | None) -> None:
    """Register a scheduler-process evidence sink owned by the parent plugin."""
    global _restore_evidence_sink
    _restore_evidence_sink = sink


class TieredKVCacheConnector(SimpleCPUOffloadConnector):
    """Fail-closed restore registration over the native CPU offload path.

    A request opts into the tiered restore gate with a ``tiered_restore``
    mapping in ``kv_transfer_params``. The mapping describes the contiguous
    suffix expected in host memory. The inherited connector still owns block
    matching, destination allocation metadata, worker copies, and completion.
    """

    _PARAM_KEY = "tiered_restore"

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        expected_epoch = extra_config.get("restore_epoch")
        self._expected_restore_epoch = (
            None if expected_epoch is None else str(expected_epoch)
        )
        self._restore_fallback_reasons: dict[str, str] = {}
        self._restore_started_at: dict[str, float] = {}
        self._restore_transfer_evidence: dict[str, dict[str, Any]] = {}

    def get_restore_fallback_reason(self, request: "Request") -> str | None:
        return self._restore_fallback_reasons.get(request.request_id)

    def has_restore_candidate(self, request: "Request") -> bool:
        return (
            self._get_restore_candidate(request) is not None
            or _restore_candidate_provider is not None
        )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        candidate = self._get_restore_candidate(request)
        if candidate is None:
            if _restore_candidate_provider is None:
                self._restore_fallback_reasons.pop(request.request_id, None)
                return super().get_num_new_matched_tokens(
                    request, num_computed_tokens
                )
            matched_tokens, async_load = super().get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            if matched_tokens is None:
                self._restore_fallback_reasons[request.request_id] = (
                    "connector_match_unavailable"
                )
                return None, async_load
            manager = self.scheduler_manager
            block_size = manager.hash_block_size if manager is not None else 0
            candidate = _restore_candidate_provider(
                request,
                num_computed_tokens,
                matched_tokens,
                block_size,
                self._expected_restore_epoch,
            )
        else:
            matched_tokens = None
            async_load = False

        fallback_reason = self._validate_restore_candidate(
            candidate, request, num_computed_tokens
        )
        if fallback_reason is not None:
            self._restore_fallback_reasons[request.request_id] = fallback_reason
            return 0, False

        if matched_tokens is None:
            matched_tokens, async_load = super().get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        if matched_tokens is None:
            self._restore_fallback_reasons[request.request_id] = (
                "connector_match_unavailable"
            )
            return None, async_load

        expected_tokens = int(candidate["block_count"]) * int(
            candidate["block_size_tokens"]
        )
        if matched_tokens < expected_tokens:
            self._restore_fallback_reasons[request.request_id] = (
                "restore_span_partial"
                if matched_tokens > 0
                else "restore_payload_unavailable"
            )
            return 0, False

        self._restore_fallback_reasons.pop(request.request_id, None)
        self._restore_started_at[request.request_id] = time.perf_counter()
        return expected_tokens, async_load

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        request_id = request.request_id
        if num_external_tokens <= 0:
            self._restore_started_at.pop(request_id, None)
            self._restore_transfer_evidence.pop(request_id, None)
            return
        if request_id not in self._restore_started_at:
            return
        transfer_blocks = 0
        manager = self.scheduler_manager
        if manager is not None:
            load_state = manager._reqs_to_load.get(request_id)
            transfer_meta = getattr(load_state, "transfer_meta", None)
            transfer_blocks = len(getattr(transfer_meta, "gpu_block_ids", ()))
        evidence = {
            "event": "kv_restore_connector_scheduled",
            "request_id": request_id,
            "restored_tokens": int(num_external_tokens),
            "transfer_block_pairs": transfer_blocks,
            "estimated_hbm_host_bytes": self._estimate_transfer_bytes(
                int(num_external_tokens)
            ),
            "evidence_quality": "scheduler-derived",
        }
        self._restore_transfer_evidence[request_id] = evidence
        self._emit_restore_evidence(evidence)

    def update_connector_output(self, connector_output: Any) -> None:
        now = time.perf_counter()
        self._emit_completed_store_evidence(connector_output)
        for request_id in connector_output.finished_recving or ():
            started_at = self._restore_started_at.pop(request_id, None)
            evidence = self._restore_transfer_evidence.pop(request_id, None)
            if started_at is None or evidence is None:
                continue
            self._emit_restore_evidence(
                {
                    **evidence,
                    "event": "kv_restore_connector_complete",
                    "restore_latency_ms": (now - started_at) * 1000.0,
                    "completion_source": "worker_finished_recving",
                    "executed_hbm_host_bytes": evidence[
                        "estimated_hbm_host_bytes"
                    ],
                    "executed_transfer_count": evidence[
                        "transfer_block_pairs"
                    ],
                    "traffic_direction": "host_to_hbm",
                    "traffic_counter_source": (
                        "worker_completion_x_kv_page_size"
                    ),
                }
            )
        super().update_connector_output(connector_output)

    def _emit_completed_store_evidence(self, connector_output: Any) -> None:
        manager = self.scheduler_manager
        worker_meta = getattr(
            connector_output, "kv_connector_worker_meta", None
        )
        completed = getattr(worker_meta, "completed_store_events", None)
        if manager is None or not isinstance(completed, Mapping):
            return
        expected_workers = int(getattr(manager, "_expected_worker_count", 1))
        pending_counts = getattr(manager, "_store_event_pending_counts", {})
        transfers = getattr(manager, "_store_event_to_blocks", {})
        request_map = getattr(manager, "_store_event_to_reqs", {})
        for event_idx, count in completed.items():
            if int(pending_counts.get(event_idx, 0)) + int(count) < expected_workers:
                continue
            transfer = transfers.get(event_idx)
            if transfer is None:
                continue
            transfer_blocks = len(getattr(transfer, "gpu_block_ids", ()))
            self._emit_restore_evidence(
                {
                    "event": "kv_store_connector_complete",
                    "store_event": int(event_idx),
                    "request_ids": list(request_map.get(event_idx, ())),
                    "executed_transfer_count": transfer_blocks,
                    "executed_hbm_host_bytes": (
                        self._estimate_uniform_block_transfer_bytes(
                            transfer_blocks
                        )
                    ),
                    "traffic_direction": "hbm_to_host",
                    "traffic_counter_source": (
                        "worker_completion_x_kv_page_size"
                    ),
                    "completion_source": "worker_store_event_count",
                }
            )

    def _estimate_transfer_bytes(self, num_external_tokens: int) -> int | None:
        manager = self.scheduler_manager
        config = getattr(manager, "cpu_kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", None)
        if manager is None or not groups:
            return None
        cp_world_size = int(getattr(manager, "cp_world_size", 1))
        total = 0
        for group in groups:
            spec = group.kv_cache_spec
            effective_block_size = int(spec.block_size) * cp_world_size
            if effective_block_size <= 0 or num_external_tokens % effective_block_size:
                return None
            total += (
                num_external_tokens // effective_block_size
            ) * int(spec.page_size_bytes)
        return total

    def _estimate_uniform_block_transfer_bytes(
        self, transfer_blocks: int
    ) -> int | None:
        manager = self.scheduler_manager
        config = getattr(manager, "cpu_kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", None)
        if not groups:
            return None
        page_sizes = {
            int(group.kv_cache_spec.page_size_bytes) for group in groups
        }
        if len(page_sizes) != 1:
            return None
        return transfer_blocks * page_sizes.pop()

    @staticmethod
    def _emit_restore_evidence(event: Mapping[str, Any]) -> None:
        if _restore_evidence_sink is not None:
            _restore_evidence_sink(event)

    @classmethod
    def _get_restore_candidate(cls, request: "Request") -> Mapping[str, Any] | None:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, Mapping) or cls._PARAM_KEY not in params:
            return None
        candidate = params[cls._PARAM_KEY]
        return candidate if isinstance(candidate, Mapping) else {}

    def _validate_restore_candidate(
        self,
        candidate: Mapping[str, Any],
        request: "Request",
        num_computed_tokens: int,
    ) -> str | None:
        fallback_reason = candidate.get("fallback_reason")
        if fallback_reason:
            return str(fallback_reason)
        if candidate.get("ready") is not True:
            return "restore_event_not_ready"
        if candidate.get("complete") is not True:
            return "restore_span_partial"
        if not candidate.get("chain_id"):
            return "hash_chain_identity_missing"

        request_id = str(candidate.get("request_id", ""))
        if request_id != request.request_id:
            return "restore_candidate_request_mismatch"

        try:
            start_offset = int(candidate["start_offset"])
            block_count = int(candidate["block_count"])
            block_size = int(candidate["block_size_tokens"])
        except (KeyError, TypeError, ValueError):
            return "restore_candidate_invalid"

        if block_count <= 0 or block_size <= 0:
            return "restore_span_empty"
        if start_offset < 0:
            return "restore_span_invalid_offset"
        if (
            self._expected_restore_epoch is not None
            and str(candidate.get("epoch", "")) != self._expected_restore_epoch
        ):
            return "hash_chain_epoch_mismatch"

        manager = self.scheduler_manager
        if manager is None:
            return "restore_scheduler_manager_unavailable"
        if block_size != manager.hash_block_size:
            return "restore_block_size_mismatch"
        if num_computed_tokens % block_size != 0:
            return "restore_prefix_unaligned"
        if start_offset != num_computed_tokens // block_size:
            return "restore_span_not_prefill_aligned"

        candidate_hashes = candidate.get("block_hashes")
        if not isinstance(candidate_hashes, (list, tuple)):
            return "hash_chain_identity_missing"
        if len(candidate_hashes) != block_count:
            return "hash_chain_identity_missing"
        request_hashes = request.block_hashes[
            start_offset : start_offset + block_count
        ]
        if len(request_hashes) != block_count:
            return "restore_span_exceeds_request_hash_chain"
        actual_hashes = tuple(
            self._block_hash_fingerprint(item) for item in request_hashes
        )
        if actual_hashes != tuple(str(item) for item in candidate_hashes):
            return "hash_chain_identity_mismatch"

        restored_tokens = block_count * block_size
        max_restorable = max(request.num_tokens - 1 - num_computed_tokens, 0)
        if restored_tokens > max_restorable:
            return "restore_span_exceeds_request"
        return None

    @staticmethod
    def _block_hash_fingerprint(block_hash: Any) -> str:
        if isinstance(block_hash, (bytes, bytearray)):
            return bytes(block_hash).hex()
        return str(block_hash)
