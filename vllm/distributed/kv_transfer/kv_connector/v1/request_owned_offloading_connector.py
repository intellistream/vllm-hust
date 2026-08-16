# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exclusive transfer substrate for request-owned bulk KV correctness.

This connector deliberately keeps the generic scheduler offload lifecycle
inert.  Its worker role only canonicalizes/registers the real KV caches and
then hands the existing manager/worker primitives to the owner-local strict
adapter.  Sharing those primitives with generic connector jobs would collide
job-id and lifecycle namespaces, so the config gate makes this connector an
exclusive mode.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.request_owned_offload import RequestOwnedBulkOffloadAdapter

if TYPE_CHECKING:
    from vllm.v1.request import Request


class RequestOwnedOffloadingConnector(OffloadingConnector):
    """Default-inert connector whose worker is owned by one strict adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_owned_adapter: RequestOwnedBulkOffloadAdapter | None = None

    def build_request_owned_adapter(
        self, owner_rank: int
    ) -> RequestOwnedBulkOffloadAdapter:
        """Transfer exclusive manager/worker ownership to the local owner."""

        if self.connector_worker is None or self.connector_worker.worker is None:
            raise RuntimeError(
                "request-owned offload adapter requires registered worker KV caches"
            )
        if self._request_owned_adapter is None:
            self._request_owned_adapter = RequestOwnedBulkOffloadAdapter(
                owner_rank=owner_rank,
                manager=self.spec.get_manager(),
                worker=self.connector_worker.worker,
            )
        elif self._request_owned_adapter.ledger.owner_rank != owner_rank:
            raise RuntimeError(
                "request-owned offload adapter is already bound to owner "
                f"{self._request_owned_adapter.ledger.owner_rank}, not {owner_rank}"
            )
        return self._request_owned_adapter

    # Generic scheduler/worker connector hooks stay inert. Request-owned
    # command handling invokes the adapter explicitly at its physical fences.
    def on_new_request(self, request: "Request") -> None:
        return

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return OffloadingConnectorMetadata(load_jobs={}, store_jobs={})

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        return

    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        return set(), set()

    def build_connector_worker_meta(self):
        return None

    def update_connector_output(self, connector_output) -> None:
        return

    def has_pending_push_work(self) -> bool:
        return False

    def take_reclaimable_block_ids(self) -> set[int]:
        return set()

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        return ()

    def reset_cache(self) -> bool:
        # A scheduler-side reset cannot synchronously prove that every
        # worker-private owner adapter dropped the same host image set.
        # Refuse the generic reset rather than acknowledge a false clear.
        return False

    def shutdown(self) -> None:
        if self._request_owned_adapter is None:
            super().shutdown()
            return
        self._request_owned_adapter.shutdown()
        self._request_owned_adapter = None
        assert self.connector_worker is not None
        self.connector_worker.worker = None
