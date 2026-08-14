# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    PrefixCacheEventUploaderFactory,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
)
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.owner_layout import GlobalRowId
from vllm.v1.core.sched.owner_layout_probe import OwnerLayoutProbe
from vllm.v1.core.sched.owner_window_policy import (
    OwnerPrefillCandidate,
    OwnerWindowPolicy,
    OwnerWindowPolicyConfig,
    OwnerWindowPolicyPhase,
    OwnerWindowReadiness,
)
from vllm.v1.core.sched.ownership import (
    EpochFenceError,
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerAssignmentObservation,
    OwnerCachePoolSnapshot,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseCoordinator,
    OwnerLeaseKey,
    OwnerReadinessReceipt,
    OwnerReceipt,
    OwnerReceiptBatch,
    OwnerResidencyState,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.core.sched.victim_selector import (
    get_victim_selector,
    infer_kv_utilization_from_scheduler,
)
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import (
    DraftTokenIds,
    KVConnectorOutput,
    ModelRunnerOutput,
    OwnerSamplingBatch,
)
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class _OwnerWindowPhase(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass(frozen=True)
class _OwnerWindowStep:
    """One immutable positive-token window step awaiting model output."""

    step_seq: int
    phase: _OwnerWindowPhase
    members: tuple[OwnerLeaseKey, ...]
    num_scheduled_tokens: tuple[tuple[str, int], ...]


@dataclass
class _OwnerWindowState:
    """Scheduler-owned phase state for the experimental window policy."""

    phase: _OwnerWindowPhase | None = None
    # Owner-ordered and stable, but not necessarily complete: an exact
    # world-size cohort is FULL-graph eligible, while a partial cohort must
    # remain runnable through the ordinary non-FULL fallback.
    decode_slots: tuple[OwnerLeaseKey, ...] = ()
    yielded_decode_slots: tuple[OwnerLeaseKey, ...] = ()
    suspended_decode_slots: tuple[OwnerLeaseKey, ...] = ()
    prefill_wave: tuple[OwnerLeaseKey, ...] = ()
    decode_steps: int = 0
    inflight: _OwnerWindowStep | None = None


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = self.kv_events_config is not None and (
            self.kv_events_config.enable_kv_cache_events
            or self.kv_events_config.prefix_cache_upload_endpoint is not None
        )
        self.available_kv_cache_memory_bytes: int | None = None
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # Victim selector (pluggable via vllm.victim_selector entry point).
        self.victim_selector = get_victim_selector(self.vllm_config)

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.use_dflash():
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                self.num_lookahead_tokens = self.num_spec_tokens + 1
            if speculative_config.use_dspark():
                # DSpark drafts a block of num_spec_tokens query tokens in which the
                # anchor itself is the first prediction position (no separate bonus
                # query), so it needs exactly num_spec_tokens lookahead slots.
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
        )
        prefix_cache_snapshot = (
            self.kv_cache_manager.get_prefix_cache_snapshot(
                node_id="",
                data_parallel_rank=self.parallel_config.data_parallel_index,
            )
            if self.kv_events_config is not None
            and self.kv_events_config.prefix_cache_upload_endpoint is not None
            else None
        )
        self.prefix_cache_event_uploader = PrefixCacheEventUploaderFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
            initial_snapshot=prefix_cache_snapshot,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

        # G2 request-owned attention: scheduler-side receipt-gated admission
        # and control plane.  Inert while the experimental flag is off; when
        # on, the scheduler issues owner commands, applies worker receipts,
        # and publishes leases without allocating scheduler KV blocks or
        # dispatching token execution (executor/worker terminal gates reject
        # token-bearing steps until owner-local KV routing lands).
        self.owner_coordinator: OwnerLeaseCoordinator | None = None
        # request_id -> lease key (request id + reuse epoch) tracked here.
        self._owner_key: dict[str, OwnerLeaseKey] = {}
        # request_id -> next lease epoch (request-id reuse fences forward).
        self._owner_epoch: dict[str, int] = {}
        # request_id -> (command_seq, kind) of the in-flight owner command.
        self._owner_pending_command: dict[str, tuple[int, OwnerCommandKind]] = {}
        # owner rank -> highest command_seq already placed on the wire.
        self._owner_emitted_command_seq: dict[int, int] = {}
        # Commands issued since the last schedule() call, drained per step.
        self._owner_outbox: list[OwnerCommand] = []
        # Internal per-step RUNNING token plans (never dispatched for
        # execution; see _schedule_request_owned).
        self._owner_token_plans: dict[str, int] = {}
        # request_id -> status before promotion whose corresponding worker
        # dispatch has not happened yet (WAITING = first dispatch,
        # PREEMPTED = resumed).  This is deliberately persistent across
        # schedule() calls: admission can promote more requests than the
        # global token budget can dispatch in that step.
        self._owner_pending_dispatch: dict[str, RequestStatus] = {}
        # Experimental scheduler-owned batch phase.  Request/coordinator
        # objects remain authoritative for tokens, leases, and capacity; this
        # record only selects a stable execution cohort and advances at the
        # completed-output boundary.
        self._owner_window_state = _OwnerWindowState()
        self._owner_window_policy: OwnerWindowPolicy | None = None
        self._owner_readiness: dict[OwnerLeaseKey, OwnerReadinessReceipt] = {}
        self._owner_readiness_seen = False
        self._owner_wait_started_step: dict[OwnerLeaseKey, int] = {}
        # Global rank -> latest worker-confirmed physical pool snapshot
        # (block-ID-free), empty until the first non-None receipt envelope.
        self._owner_pool_snapshots: dict[int, OwnerCachePoolSnapshot] = {}
        # True once any rank published a physical snapshot; afterwards every
        # step must re-publish one per rank (no silent stale physical facts).
        self._owner_pool_snapshot_seen = False
        self._owner_layout_probe: OwnerLayoutProbe | None = None
        if (
            OwnerLayoutProbe.requested()
            and not self.scheduler_config.enable_request_owned_attention
        ):
            raise RuntimeError(
                "request-owner layout probe requires "
                "enable_request_owned_attention=true"
            )
        if self.scheduler_config.enable_request_owned_attention:
            self._init_request_owned_control_plane()
            self._owner_layout_probe = OwnerLayoutProbe.from_env(
                world_size=self.parallel_config.world_size
            )

    def _should_get_num_common_prefix_blocks(self) -> bool:
        device_config = getattr(self.vllm_config, "device_config", None)
        device_type = getattr(device_config, "device_type", None)

        # Only the v1 CUDA runner consumes this field for cascade attention.
        return (
            device_type == "cuda"
            and not self.use_v2_model_runner
            and not getattr(
                self.vllm_config.model_config,
                "disable_cascade_attn",
                False,
            )
            and not self.parallel_config.use_ubatching
        )

    def _get_num_common_prefix_blocks(self) -> list[int]:
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if not self.running or not self._should_get_num_common_prefix_blocks():
            return num_common_prefix_blocks

        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            any_request_id = self.running[0].request_id
            return self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_uncached_common_prefix_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass

            # Marconi cache admission optimization:
            # cache common prefixes by scheduling num_new_tokens = common prefix length
            if (
                num_uncached_common_prefix_tokens >= block_size
                and num_new_tokens > num_uncached_common_prefix_tokens
            ):
                num_new_tokens = num_uncached_common_prefix_tokens
                # keep alignment to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
        return num_new_tokens

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        if self.scheduler_config.enable_request_owned_attention:
            return self._schedule_request_owned()
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt a victim via pluggable selector.
                    preempted_req = self.victim_selector.pick_victim(
                        self.running,
                        self.policy,
                        kv_utilization=infer_kv_utilization_from_scheduler(self),
                        now_s=scheduled_timestamp,
                    )
                    self.running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens.pop(preempted_req_id)
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            # Restore encoder compute budget if the preempted
                            # request had encoder inputs scheduled in this step.
                            num_embeds_to_restore = sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
                            )
                            encoder_compute_budget += num_embeds_to_restore
                        req_index -= 1

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                num_uncached_common_prefix_tokens = 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    if (
                        self.connector is not None
                        and self.has_mamba_layers
                        and isinstance(
                            self.kv_cache_manager.coordinator,
                            HybridKVCacheCoordinator,
                        )
                    ):
                        computed, per_group_hits = (
                            self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
                                request.block_hashes,
                                request.num_tokens - 1,
                            )
                        )
                        new_computed_blocks = (
                            self.kv_cache_manager.create_kv_cache_blocks(computed)
                        )
                        # NOTE(ZhanqiuHu): For Mamba hybrid models,
                        # num_new_local_computed_tokens should be the FA hit
                        # length. This value is passed to the connector's
                        # get_num_new_matched_tokens which computes:
                        # external = total - local_computed.
                        # Using the FA hit skips re-transferring FA blocks
                        # already cached on D-side. The Mamba state (always
                        # the last block) is transferred unconditionally by
                        # _apply_prefix_caching in nixl/worker.py.
                        num_new_local_computed_tokens = max(per_group_hits)
                        if self.kv_cache_manager.log_stats:
                            assert self.kv_cache_manager.prefix_cache_stats is not None
                            self.kv_cache_manager.prefix_cache_stats.record(
                                num_tokens=request.num_tokens,
                                num_hits=num_new_local_computed_tokens,
                                preempted=request.num_preemptions > 0,
                            )
                    else:
                        new_computed_blocks, num_new_local_computed_tokens = (
                            self.kv_cache_manager.get_computed_blocks(request)
                        )

                    # In case of hybrid models, obtain hint for Marconi-style APC logic
                    if self.has_mamba_layers:
                        num_uncached_common_prefix_tokens = getattr(
                            self.kv_cache_manager.coordinator,
                            "num_uncached_common_prefix_tokens",
                            0,
                        )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # Skip request with pending mm encoding prefetches
                    if (
                        self.ec_connector is not None
                        and request.mm_features
                        and not self.ec_connector.ensure_cache_available(
                            request, num_computed_tokens
                        )
                    ):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats is not None:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and num_new_tokens == 1
                        and (scheduled_running_reqs and not prefill_scheduled)
                    ):
                        num_new_tokens = 1 + self.num_spec_tokens
                        if (
                            num_new_tokens > token_budget
                            or num_computed_tokens + num_new_tokens > self.max_model_len
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            break
                        pad_spec_decode = True

                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                        num_uncached_common_prefix_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                limit_lookahead_tokens = load_kv_async and self.use_eagle
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if self.running:
                        # Running requests will free blocks when they
                        # complete; stop here to preserve queue-order
                        # admission.
                        break
                    # Nothing is running, so no future event frees blocks and
                    # stopping at this request would freeze this state
                    # permanently. Requests behind this one may hold blocks
                    # while parked (async KV loads in WAITING_FOR_REMOTE_KVS)
                    # and are only promoted when this traversal reaches them.
                    # Keep scanning so they can be promoted, scheduled, and
                    # eventually free the blocks this request needs.
                    # See https://github.com/vllm-project/vllm/issues/45388
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if pad_spec_decode:
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = self._get_num_common_prefix_blocks()

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            kv_cache_usage=self.kv_cache_manager.usage,
            # Request-owned attention uses the scheduler call ordinal as its
            # global execution/receipt fence.  This is intentionally distinct
            # from sched_step_seq, which only advances for deferred KV frees.
            step_seq=self.current_step,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)

        if preempted_reqs:
            self.victim_selector.emit_observability_log(logger, self.__class__.__name__)

        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        if self.scheduler_config.enable_request_owned_attention:
            self._preempt_request_owned(request, timestamp)
            return
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            if item_identifier in mm_hashes_to_schedule:
                # The same encoder input has already been scheduled in the
                # current step.
                continue

            if self.encoder_cache_manager.check_and_update_cache(request, i):
                # The encoder input is already computed and cached from a
                # previous step.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # G3: fail-closed semantic validation of the aggregated
        # owner-sampling envelope before ANY request/output mutation below
        # (receipt application, sampled-token application, computed-token
        # adjustments).  See _validate_request_owned_sampling_envelope for
        # the exact contract and the runner-unlock call seam.
        self._validate_request_owned_sampling_envelope(
            scheduler_output, model_runner_output
        )

        scheduler_config = getattr(self, "scheduler_config", None)
        request_owned_attention = getattr(
            scheduler_config, "enable_request_owned_attention", False
        )
        if not isinstance(request_owned_attention, bool):
            raise RuntimeError(
                "enable_request_owned_attention must remain a bool at the "
                f"scheduler output boundary, got {request_owned_attention!r}."
            )
        if request_owned_attention:
            # G2: apply structurally valid receipts before any ordinary
            # output or request mutation below.
            self._apply_request_owned_receipts(scheduler_output, model_runner_output)
            # Evidence must bind the executed layout to the worker-confirmed
            # post-step pool state.  Recording during schedule() would lag
            # physical capacity by one step and could not directly prove the
            # final RELEASE returned every owner-local block.
            owner_layout_probe = getattr(self, "_owner_layout_probe", None)
            if owner_layout_probe is not None:
                owner_layout_probe.record_step(
                    step_seq=scheduler_output.step_seq,
                    leases=scheduler_output.scheduled_owner_leases,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    commands=scheduler_output.owner_commands,
                    receipt_batches=model_runner_output.owner_receipt_batches or (),
                    cache_pool_snapshots=(
                        self._owner_pool_snapshots
                        if self._owner_pool_snapshot_seen
                        else None
                    ),
                )
        else:
            self._validate_request_owned_receipt_ingress(
                scheduler_output, model_runner_output
            )

        # Knorm: route block scores from model runner to KV cache manager.
        knorm_scores = getattr(model_runner_output, "knorm_block_scores", None)
        if knorm_scores:
            from vllm.knorm.manager import submit_block_scores

            submit_block_scores(knorm_scores)

        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
            self.prefix_cache_event_uploader.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        if getattr(
            self.scheduler_config,
            "enable_request_owned_windows",
            False,
        ):
            # This is deliberately the final state mutation: sampled tokens,
            # EOS/abort handling, queue removal, and output construction have
            # all succeeded.  A failure above leaves the step in flight and
            # prevents scheduling from optimistic num_computed_tokens.
            self._owner_window_ack_step(scheduler_output)

        return engine_core_outputs

    def _validate_request_owned_receipt_ingress(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> None:
        """Fail closed at the G1 scheduler receipt boundary.

        G1 proves exact per-step all-worker transport but deliberately does
        not apply resource events: applying them would require the G2
        owner-local allocator/coordinator authority.  Structural violations
        and nonempty events raise before ordinary output or request mutation.
        """
        if not getattr(
            getattr(self, "scheduler_config", None),
            "enable_request_owned_attention",
            False,
        ):
            return

        step_seq = scheduler_output.step_seq
        if isinstance(step_seq, bool) or not isinstance(step_seq, int) or step_seq <= 0:
            raise RuntimeError(
                "request-owned receipt ingress requires a positive non-bool "
                f"step_seq, got {step_seq!r}."
            )

        batches = model_runner_output.owner_receipt_batches
        if batches is None:
            raise RuntimeError(
                "request-owned receipt ingress is missing all-worker receipt batches."
            )
        expected_ranks = list(range(self.parallel_config.world_size))
        ranks = [batch.owner_rank for batch in batches]
        if (
            any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
            or len(ranks) != len(expected_ranks)
            or sorted(ranks) != expected_ranks
        ):
            raise RuntimeError(
                "request-owned receipt ingress requires exactly one batch per "
                f"process-global owner rank {expected_ranks}, got {ranks}."
            )
        for batch in batches:
            if (
                isinstance(batch.emitted_step_seq, bool)
                or not isinstance(batch.emitted_step_seq, int)
                or batch.emitted_step_seq != step_seq
            ):
                raise RuntimeError(
                    "request-owned receipt ingress got emitted_step_seq "
                    f"{batch.emitted_step_seq} from owner {batch.owner_rank}, "
                    f"expected {step_seq}."
                )
            for event in batch.events:
                if (
                    isinstance(event.owner_id, bool)
                    or not isinstance(event.owner_id, int)
                    or event.owner_id != batch.owner_rank
                ):
                    raise RuntimeError(
                        "request-owned receipt ingress got an event for owner "
                        f"{event.owner_id} in owner {batch.owner_rank}'s batch."
                    )
        if any(batch.events for batch in batches):
            raise RuntimeError(
                "request-owned attention G1 cannot apply resource receipt "
                "events without the G2 owner-local allocator/coordinator; "
                "refusing to ignore them."
            )

    def _validate_request_owned_sampling_envelope(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> None:
        """G3: fail-closed scheduler-side validation of the aggregated
        owner-sampling envelope, before any request state mutation.

        Runs from :meth:`update_from_output` ahead of receipt application,
        sampled-token application, and computed-token adjustments.  The
        envelope aggregates every owner's :class:`OwnerSamplingBatch`
        (all-worker envelope cardinality/slot checks remain the executor
        aggregator's authority); this boundary validates the
        scheduler-facing semantics:

        * the envelope request set is exactly the set of positive-count
          scheduled requests, each appearing exactly once (no unknown,
          finished, or duplicate requests),
        * every batch's ``owner_rank`` and ``emitted_step_seq`` match the
          scheduler step and the request's authoritative current
          owner/lease epoch,
        * every row is the terminal logits-producing
          :class:`GlobalRowId` for that request (``pre-step
          num_computed_tokens + scheduled_count - 1``, lane 0), not one
          row per scheduled token,
        * the merged output has a bijective req map aligned to the same
          request set, with at most one sampled token per request (the G3
          no-spec envelope), and
        * zero-token heartbeat steps reject nonempty sampling identities
          and never mutate sampling state (this boundary keeps no sampling
          state at all; it only validates).

        ``owner_sampling_batches is None`` remains the default-off output and
        is accepted only while ``enable_request_owned_sampling`` is false.
        Once that experimental transport gate is enabled, absence is
        authoritative failure even for a zero-work heartbeat: every worker
        slot must emit an explicit (possibly empty) batch.  Conversely, an
        envelope arriving while the sampling gate is disabled is rejected.
        """
        batches = model_runner_output.owner_sampling_batches
        # Some narrow scheduler utility/test paths intentionally construct a
        # bare Scheduler without running __init__.  They remain the default-off
        # protocol path, so an absent config is equivalent to both request-owned
        # gates being false.  A real owner envelope still fails closed below.
        scheduler_config = getattr(self, "scheduler_config", None)
        sampling_enabled = getattr(
            scheduler_config, "enable_request_owned_sampling", False
        )
        if not isinstance(sampling_enabled, bool):
            raise RuntimeError(
                "enable_request_owned_sampling must remain a bool at the "
                f"scheduler boundary, got {sampling_enabled!r}."
            )
        if batches is None:
            if sampling_enabled:
                raise RuntimeError(
                    "request-owned sampling is enabled but the terminal "
                    "model output carries no owner_sampling_batches; every "
                    "worker slot must emit an explicit batch, including an "
                    "empty batch on zero-work steps."
                )
            return

        if not sampling_enabled:
            raise RuntimeError(
                "model output carries owner_sampling_batches while "
                "enable_request_owned_sampling is disabled; refusing an "
                "unexpected sampling authority path."
            )

        if not getattr(scheduler_config, "enable_request_owned_attention", False):
            raise RuntimeError(
                "owner-sampling envelope ingress requires "
                "enable_request_owned_attention; the scheduler is not "
                "participating in the request-owned protocol."
            )
        coordinator = getattr(self, "owner_coordinator", None)
        owner_keys = getattr(self, "_owner_key", None)
        if coordinator is None or owner_keys is None:
            raise RuntimeError(
                "owner-sampling envelope ingress requires the G2 owner "
                "coordinator; the scheduler was not initialized for it."
            )

        step_seq = scheduler_output.step_seq
        if isinstance(step_seq, bool) or not isinstance(step_seq, int) or step_seq <= 0:
            raise RuntimeError(
                "owner-sampling envelope ingress requires a positive "
                f"non-bool step_seq, got {step_seq!r}."
            )

        # Authoritative per-request owner/lease epoch for every
        # positive-count scheduled request.  A scheduled request that is
        # unknown/finished or lacks a live lease cannot be validated and
        # fails closed here.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        authoritative_owner: dict[str, int] = {}
        authoritative_epoch: dict[str, int] = {}
        for req_id, count in num_scheduled_tokens.items():
            if count <= 0:
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                raise RuntimeError(
                    "owner-sampling envelope: scheduled request "
                    f"{req_id!r} is unknown or finished; refusing to "
                    "validate a sampling identity for it."
                )
            key = owner_keys.get(req_id)
            if key is None:
                raise RuntimeError(
                    "owner-sampling envelope: scheduled request "
                    f"{req_id!r} has no live owner lease key."
                )
            owner_rank = coordinator.owner_of(key)
            if owner_rank is None:
                raise RuntimeError(
                    "owner-sampling envelope: scheduled request "
                    f"{req_id!r} has no assigned owner rank."
                )
            authoritative_owner[req_id] = owner_rank
            authoritative_epoch[req_id] = key.owner_epoch

        # Pre-step computed-token counts: request.num_computed_tokens is
        # already advanced by _update_after_schedule at this seam, so the
        # authoritative pre-step snapshot rides the scheduler_output
        # payload; scheduled_new_reqs + scheduled_cached_reqs cover exactly
        # the positive-count scheduled requests on the request-owned path,
        # so any coverage gap here is evidence the flow changed and fails
        # closed instead of inferring post-mutation request state.
        pre_step_num_computed_tokens = self._owner_sampling_pre_step_counts(
            scheduler_output
        )

        self._validate_owner_sampling_envelope(
            scheduler_step_seq=step_seq,
            num_scheduled_tokens=num_scheduled_tokens,
            pre_step_num_computed_tokens=pre_step_num_computed_tokens,
            authoritative_owner_by_request_id=authoritative_owner,
            authoritative_epoch_by_request_id=authoritative_epoch,
            owner_sampling_batches=batches,
            model_runner_output=model_runner_output,
        )

    @staticmethod
    def _owner_sampling_pre_step_counts(
        scheduler_output: SchedulerOutput,
    ) -> dict[str, int]:
        """Snapshot per-request pre-step ``num_computed_tokens`` from the
        scheduler output payload (never from mutated request state).

        Conflicting snapshots for one request id fail closed: the same
        incarnation cannot carry two different pre-step counts in one step.
        """
        pre_step: dict[str, int] = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            prev = pre_step.get(new_req.req_id)
            if prev is not None and prev != new_req.num_computed_tokens:
                raise RuntimeError(
                    "owner-sampling envelope: conflicting pre-step "
                    "num_computed_tokens snapshots for "
                    f"{new_req.req_id!r}."
                )
            pre_step[new_req.req_id] = new_req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        if len(cached.req_ids) != len(cached.num_computed_tokens):
            raise RuntimeError(
                "owner-sampling envelope: scheduled_cached_reqs "
                "num_computed_tokens must be aligned 1:1 with req_ids "
                f"({len(cached.num_computed_tokens)} counts vs "
                f"{len(cached.req_ids)} ids)."
            )
        for req_id, num_computed in zip(cached.req_ids, cached.num_computed_tokens):
            prev = pre_step.get(req_id)
            if prev is not None and prev != num_computed:
                raise RuntimeError(
                    "owner-sampling envelope: conflicting pre-step "
                    f"num_computed_tokens snapshots for {req_id!r}."
                )
            pre_step[req_id] = num_computed
        return pre_step

    @staticmethod
    def _validate_owner_sampling_envelope(
        *,
        scheduler_step_seq: int,
        num_scheduled_tokens: Mapping[str, int],
        pre_step_num_computed_tokens: Mapping[str, int],
        authoritative_owner_by_request_id: Mapping[str, int],
        authoritative_epoch_by_request_id: Mapping[str, int],
        owner_sampling_batches: Sequence[OwnerSamplingBatch],
        model_runner_output: ModelRunnerOutput,
    ) -> None:
        """Pure G3 owner-sampling envelope validation (see the seam method).

        Raises :class:`RuntimeError` on the first contract violation;
        performs no state mutation and keeps no state, so a rejection can
        never leave partial sampling or request state behind.  Rows carry
        ``GlobalRowId`` identities by :class:`OwnerSamplingBatch`
        construction; the executor aggregator is the authority for
        all-worker envelope cardinality and transport-slot checks.
        """
        scheduled_req_ids = {
            req_id for req_id, count in num_scheduled_tokens.items() if count > 0
        }

        rows: list[tuple[OwnerSamplingBatch, GlobalRowId]] = []
        seen_req_ids: set[str] = set()
        for batch in owner_sampling_batches:
            if (
                isinstance(batch.owner_rank, bool)
                or not isinstance(batch.owner_rank, int)
                or batch.owner_rank < 0
            ):
                raise RuntimeError(
                    "owner-sampling envelope: owner_rank must be a "
                    f"nonnegative non-bool int, got {batch.owner_rank!r}."
                )
            if batch.emitted_step_seq != scheduler_step_seq:
                raise RuntimeError(
                    "owner-sampling envelope: batch emitted_step_seq "
                    f"{batch.emitted_step_seq} from owner "
                    f"{batch.owner_rank} does not match the scheduler "
                    f"step {scheduler_step_seq}."
                )
            for row in batch.row_ids:
                req_id = row.request_uid.request_id
                if req_id in seen_req_ids:
                    raise RuntimeError(
                        "owner-sampling envelope: duplicate request "
                        f"{req_id!r} across owner batches."
                    )
                seen_req_ids.add(req_id)
                rows.append((batch, row))

        if not scheduled_req_ids and rows:
            # Zero-token heartbeat/control steps must not carry sampling
            # identities and must not mutate sampling state.
            raise RuntimeError(
                "owner-sampling envelope: zero-token heartbeat step "
                f"carries nonempty sampling identities "
                f"{[row.request_uid.request_id for _, row in rows]!r}; "
                "refusing to mutate sampling state."
            )

        envelope_req_ids = [row.request_uid.request_id for _, row in rows]
        envelope_set = set(envelope_req_ids)
        missing = sorted(scheduled_req_ids - envelope_set)
        if missing:
            raise RuntimeError(
                "owner-sampling envelope: missing sampling identity for "
                f"scheduled request(s) {missing}."
            )
        extra = sorted(envelope_set - scheduled_req_ids)
        if extra:
            raise RuntimeError(
                "owner-sampling envelope: sampling identity for "
                f"unscheduled/unknown request(s) {extra}."
            )

        for batch, row in rows:
            req_id = row.request_uid.request_id
            if (
                req_id not in authoritative_owner_by_request_id
                or req_id not in authoritative_epoch_by_request_id
            ):
                raise RuntimeError(
                    "owner-sampling envelope: request "
                    f"{req_id!r} is unknown or finished."
                )
            expected_owner = authoritative_owner_by_request_id[req_id]
            if batch.owner_rank != expected_owner:
                raise RuntimeError(
                    "owner-sampling envelope: request "
                    f"{req_id!r} is scheduled on owner {expected_owner}, "
                    f"but appears in owner {batch.owner_rank}'s batch."
                )
            expected_epoch = authoritative_epoch_by_request_id[req_id]
            if row.request_uid.owner_epoch != expected_epoch:
                raise RuntimeError(
                    "owner-sampling envelope: request "
                    f"{req_id!r} sampling row carries lease epoch "
                    f"{row.request_uid.owner_epoch}, expected "
                    f"{expected_epoch}."
                )
            count = num_scheduled_tokens[req_id]
            pre_step = pre_step_num_computed_tokens.get(req_id)
            if pre_step is None:
                raise RuntimeError(
                    "owner-sampling envelope: no pre-step "
                    "num_computed_tokens snapshot for scheduled request "
                    f"{req_id!r}; cannot establish its terminal row "
                    "(flow seam gap)."
                )
            expected_position = pre_step + count - 1
            if row.logical_token_position != expected_position:
                raise RuntimeError(
                    "owner-sampling envelope: request "
                    f"{req_id!r} terminal row position "
                    f"{row.logical_token_position} does not match pre-step "
                    f"computed {pre_step} + scheduled {count} - 1 = "
                    f"{expected_position} (one terminal row per request, "
                    "not one per scheduled token)."
                )
            if row.logical_lane != 0:
                raise RuntimeError(
                    "owner-sampling envelope: request "
                    f"{req_id!r} sampling row lane {row.logical_lane} "
                    "must be 0."
                )

        # Merged output req map: bijective and aligned to the same request
        # set as the envelope, with at most one sampled token per request.
        req_ids = model_runner_output.req_ids
        req_id_to_index = model_runner_output.req_id_to_index
        if not isinstance(req_id_to_index, dict):
            raise RuntimeError(
                "owner-sampling envelope: merged output req_id_to_index "
                f"must be a dict, got {type(req_id_to_index).__name__}."
            )
        if len(req_id_to_index) != len(req_ids):
            raise RuntimeError(
                "owner-sampling envelope: merged output req map is not "
                f"bijective ({len(req_ids)} req_ids vs "
                f"{len(req_id_to_index)} index entries)."
            )
        for index, req_id in enumerate(req_ids):
            if req_id_to_index.get(req_id) != index:
                raise RuntimeError(
                    "owner-sampling envelope: merged output req map is "
                    f"not bijective (req_ids[{index}]={req_id!r})."
                )
        merged_set = set(req_ids)
        if merged_set != envelope_set:
            raise RuntimeError(
                "owner-sampling envelope: merged output request set "
                f"{sorted(merged_set)} does not match the envelope "
                f"request set {sorted(envelope_set)}."
            )
        sampled_token_ids = model_runner_output.sampled_token_ids
        if len(sampled_token_ids) != len(req_ids):
            raise RuntimeError(
                "owner-sampling envelope: merged output sampled_token_ids "
                f"({len(sampled_token_ids)} entries) must be aligned 1:1 "
                f"with req_ids ({len(req_ids)})."
            )
        for index, tokens in enumerate(sampled_token_ids):
            if not isinstance(tokens, list):
                raise RuntimeError(
                    "owner-sampling envelope: merged output "
                    f"sampled_token_ids[{index}] must be a list, got "
                    f"{type(tokens).__name__}."
                )
            if len(tokens) > 1:
                raise RuntimeError(
                    "owner-sampling envelope: spec-shaped multi-token "
                    f"sampling is unsupported (sampled_token_ids[{index}]"
                    f"={tokens!r}); the G3 no-spec envelope carries at "
                    "most one sampled token per request."
                )

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    # ------------------------------------------------------------------
    # G2 request-owned attention: receipt-gated admission/control plane
    # ------------------------------------------------------------------

    def _init_request_owned_control_plane(self) -> None:
        """Initialize the owner coordinator and its deterministic per-rank
        observations for every process-global rank."""
        self.owner_coordinator = OwnerLeaseCoordinator()
        self._owner_pool_snapshots = {}
        self._owner_pool_snapshot_seen = False
        self._owner_readiness = {}
        self._owner_readiness_seen = False
        self._owner_wait_started_step = {}
        if getattr(self.scheduler_config, "enable_request_owned_windows", False):
            self._owner_window_policy = OwnerWindowPolicy(
                OwnerWindowPolicyConfig(
                    world_size=self.parallel_config.world_size,
                    decode_observation_steps=(
                        self.scheduler_config.request_owned_decode_window_steps
                    ),
                    hot_low_watermark=getattr(
                        self.scheduler_config,
                        "request_owned_hot_low_watermark",
                        1,
                    ),
                    hot_high_watermark=getattr(
                        self.scheduler_config,
                        "request_owned_hot_high_watermark",
                        2,
                    ),
                    prefill_invocation_budget=getattr(
                        self.scheduler_config,
                        "request_owned_prefill_wave_steps",
                        1,
                    ),
                    prefill_max_wait_steps=getattr(
                        self.scheduler_config,
                        "request_owned_prefill_max_wait_steps",
                        32,
                    ),
                )
            )
        else:
            self._owner_window_policy = None
        for owner_id in range(self.parallel_config.world_size):
            self.owner_coordinator.observe(self._owner_observation(owner_id))

    def _owner_observation(self, owner_id: int) -> OwnerAssignmentObservation:
        """Deterministic per-rank observation for one owner rank.

        Once worker-confirmed physical pool snapshots are stored for every
        rank, the observation reports pool facts as
        ``work = total_blocks - free_blocks`` with zero residency (the
        prefix cluster cache is off, so aggregate residency must not be
        misused as a hit signal) and zero pending DMA.  Before the first
        snapshot every rank starts from the same zero-work base, so the
        coordinator's default least-work assignment is deterministic
        (stable numeric rank breaks ties).  The emitted observation
        explicitly excludes coordinator-local projected charges.
        """
        snapshot = self._owner_pool_snapshots.get(owner_id)
        work = (
            snapshot.total_blocks - snapshot.free_blocks if snapshot is not None else 0
        )
        return OwnerAssignmentObservation(
            owner_id=owner_id,
            observation_seq=self.current_step,
            work=work,
            residency=0,
            pending_dma=0,
        )

    def _schedule_request_owned(self) -> SchedulerOutput:
        """Run the G3 scheduler-side request-owned control plane.

        This replaces the ordinary schedule path when
        ``enable_request_owned_attention`` is on: the step freezes a global
        token plan within ``max_num_scheduled_tokens`` in RUNNING order,
        emits the block-ID-free logical token payload (``NewRequestData`` /
        ``CachedRequestData`` with empty block ids, no scheduler KV
        allocation/free/block-ID calls anywhere), and publishes exactly one
        authorization lease per scheduled key on every token-bearing step
        (even when the horizon is unchanged).  A key with an in-flight
        command is stalled and therefore never carries both a command and an
        execution token in the same step.  The executor and worker terminal
        gates still reject token-bearing steps until owner-local KV routing
        lands; this slice only constructs the payload.  Empty command-only
        control steps remain valid zero-token heartbeats.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None

        if (
            getattr(
                self.scheduler_config,
                "enable_request_owned_windows",
                False,
            )
            and self._owner_window_state.inflight is not None
        ):
            raise RuntimeError(
                "request-owned window requires completed model output before "
                "the next schedule call."
            )

        # Deterministic observations for every process-global owner rank.
        observations = [
            self._owner_observation(owner_id)
            for owner_id in range(self.parallel_config.world_size)
        ]
        for observation in observations:
            coordinator.observe(observation)

        scheduled_timestamp = time.monotonic()
        self._owner_token_plans = {}
        # Pool snapshots describe the worker-confirmed state after the prior
        # step.  Several fresh requests may be admitted before another worker
        # snapshot can arrive, so account for the block demand selected in
        # this pass instead of repeatedly spending the same free-block fact.
        self._owner_admission_projected_blocks = {
            owner_id: 0 for owner_id in range(self.parallel_config.world_size)
        }
        if self._pause_state != PauseState.PAUSED_ALL:
            self._owner_admission_pass(scheduled_timestamp)
            if getattr(
                self.scheduler_config,
                "enable_request_owned_windows",
                False,
            ):
                self._owner_window_running_pass()
            else:
                self._owner_running_pass()

        # Drain newly issued commands, per-owner ordered, each exactly once.
        owner_commands = self._drain_owner_outbox()

        # Freeze the global token plan: only positive per-request plans are
        # scheduled this step (the running pass already applied the global
        # budget in RUNNING order).
        num_scheduled_tokens = {
            req_id: plan for req_id, plan in self._owner_token_plans.items() if plan > 0
        }
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        # Classify first dispatch vs cached/resumed exactly.  A promotion is
        # not a dispatch: when the global token budget is exhausted, the
        # pending marker survives until a later positive-token step.
        first_dispatch: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        dispatched_pending_ids: set[str] = set()
        for request_id, prior_status in self._owner_pending_dispatch.items():
            if self._owner_token_plans.get(request_id, 0) <= 0:
                continue
            request = self.requests.get(request_id)
            if request is None or request.status is not RequestStatus.RUNNING:
                raise RuntimeError(
                    "request-owned pending dispatch must name a live RUNNING "
                    f"request, got {request_id!r}."
                )
            if prior_status is RequestStatus.WAITING:
                first_dispatch.append(request)
            elif prior_status is RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(
                    "request-owned promotion requires WAITING or PREEMPTED "
                    f"prior status, got {prior_status}"
                )
            dispatched_pending_ids.add(request_id)
        scheduled_running_reqs = [
            request
            for request in self.running
            if request.request_id not in self._owner_pending_dispatch
            and num_scheduled_tokens.get(request.request_id, 0) > 0
        ]
        if getattr(
            self.scheduler_config,
            "enable_request_owned_windows",
            False,
        ):
            plan_order = {
                request_id: index
                for index, request_id in enumerate(num_scheduled_tokens)
            }
            first_dispatch.sort(key=lambda request: plan_order[request.request_id])
            scheduled_resumed_reqs.sort(
                key=lambda request: plan_order[request.request_id]
            )
            scheduled_running_reqs.sort(
                key=lambda request: plan_order[request.request_id]
            )

        # Block-ID-free logical token payload.
        if self.use_v2_model_runner:
            first_dispatch.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(request, (), request._all_token_ids)
                for request in first_dispatch
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(request, ()) for request in first_dispatch
            ]
        cached_reqs_data = self._make_request_owned_cached_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
        )
        for request_id in dispatched_pending_ids:
            del self._owner_pending_dispatch[request_id]

        # Record the request ids scheduled in this step (MRV1-only), so the
        # worker-side persistent batch can skip re-propagating their full
        # token ids next step.
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # Authorization leases for exactly the scheduled keys, every
        # token-bearing step even when the horizon is unchanged.  Missing or
        # un-receipted grants for a scheduled key fail closed here.
        scheduled_keys = {self._owner_key[req_id] for req_id in num_scheduled_tokens}
        scheduled_owner_leases = coordinator.publish(self.current_step, scheduled_keys)
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            preempted_req_ids=self.reset_preempted_req_ids,
            new_block_ids_to_zero=None,
            num_spec_tokens_to_schedule=self.num_spec_tokens,
            kv_cache_usage=0.0,
            step_seq=self.current_step,
            owner_commands=owner_commands,
            owner_assignment_observations=observations,
            scheduled_owner_leases=scheduled_owner_leases,
        )

        # The request-owned scheduler bypasses the ordinary KV block planner,
        # but O-line bulk offload still installs an exclusive connector whose
        # worker-side lifecycle expects a non-None metadata envelope on every
        # step.  The connector is deliberately inert for generic jobs; invoke
        # its standard builder here so it can supply that empty typed envelope
        # without leaking generic scheduler ownership into the O-line path.
        if self.connector is not None:
            scheduler_output.kv_connector_metadata = self._build_kv_connector_meta(
                self.connector, scheduler_output
            )

        if getattr(
            self.scheduler_config,
            "enable_request_owned_windows",
            False,
        ):
            self._owner_window_record_step(scheduler_output)

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)

        return scheduler_output

    def _make_request_owned_cached_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
    ) -> CachedRequestData:
        """Block-ID-free MRV1 cached payload for the request-owned path.

        Mirrors ``_make_cached_request_data`` but never touches the
        scheduler KV pool: every ``new_block_ids`` entry is ``None`` (the
        request-owned mode allocates no scheduler KV blocks), and only the
        logical MRV1 token state is propagated (``all_token_ids`` for
        requests not scheduled in the prior step, ``num_computed_tokens``,
        ``num_output_tokens``).
        """
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, token ids are consumed via a
            # direct GPU broadcast path, so the payload may omit them.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                num_tokens = num_scheduled_tokens[req_id]
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
                # MRV1 GPUModelRunner._update_states asserts a non-None
                # ``new_block_ids`` entry for every resumed request and
                # replaces the cached block ids with it; emit the ID-free
                # empty reset ``()`` so the worker resets to zero scheduler
                # blocks.
                new_block_ids.append(())
            else:
                # Ordinary continuation: no new scheduler KV blocks exist in
                # request-owned mode, so there is nothing to append.
                new_block_ids.append(None)
            # MRV1-only: propagate full token ids for requests not scheduled
            # in the prior step.
            if (
                not self.use_v2_model_runner
                and req_id not in self.prev_step_scheduled_req_ids
            ):
                all_token_ids[req_id] = req.all_token_ids.copy()
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _owner_admission_pass(self, scheduled_timestamp: float) -> None:
        """Admission pass over the waiting queues.

        First admission assigns the least-committed-work owner and issues a
        small chunk-scaled RESERVE with a WAITING allocation descriptor,
        leaving the request provisional and unscheduled until its receipt is
        applied.  PREEMPTED requests resume with a PREEMPTED descriptor on
        their sticky owner.  Accepted leases promote the request to RUNNING.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None
        step_skipped_waiting = create_request_queue(self.policy)
        while (self.waiting or self.skipped_waiting) and (
            len(self.running) + self.num_waiting_for_streaming_input
            < self.max_num_running_reqs
        ):
            request_queue = self._select_waiting_queue_for_scheduling()
            assert request_queue is not None
            request = request_queue.peek_request()
            request_id = request.request_id

            if self._is_blocked_waiting_status(
                request.status
            ) and not self._try_promote_blocked_waiting_request(request):
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            key = self._owner_key_for(request)
            if coordinator.owner_of(key) is None:
                # First admission: least-work/stable-rank assignment plus a
                # small chunk-scaled RESERVE(WAITING); the request stays
                # provisional until the worker receipt is applied.
                self._assign_owner_and_reserve(request, key)
            elif self._owner_pending_command.get(request_id) is not None:
                # A command is in flight: stall for its receipt.
                pass
            elif (
                coordinator.is_release_pending(key)
                or coordinator.is_released(key)
                or coordinator.is_superseded(key)
            ):
                # Stale incarnation (finished/reused request id): fence
                # forward to a fresh epoch and admit again.
                key = self._roll_owner_epoch(request)
                self._assign_owner_and_reserve(request, key)
            elif coordinator.runnable_num_tokens_of(key) is None:
                # Defensive: RESERVE in flight without a tracked pending.
                pass
            elif coordinator.is_preempted(key):
                if getattr(
                    self.scheduler_config,
                    "enable_request_owned_kv_offload",
                    False,
                ) and not coordinator.is_restored(key):
                    # A cold lease first reserves/restores its final device
                    # destination. The accepted RESTORE receipt is emitted
                    # only after full H2D completion; RESERVE then reacquires
                    # runnable capacity on the following control step.
                    self._issue_owner_restore(request, key)
                else:
                    # PREEMPT receipt applied (and, in offload mode, the full
                    # restore completed): reacquire on the sticky owner.
                    self._issue_owner_reserve(request, key)
            else:
                # Accepted lease: promote to RUNNING.
                self._promote_owner_request(request, key, scheduled_timestamp)

            request_queue.pop_request()
            if request.status == RequestStatus.RUNNING:
                # Promoted: do not requeue.
                continue
            step_skipped_waiting.prepend_request(request)
        if step_skipped_waiting:
            self.waiting.prepend_requests(step_skipped_waiting)

    def _owner_running_pass(self) -> None:
        """Plan RUNNING token counts capped by the horizon and global budget.

        Per-request plans are strictly capped by the granted horizon; the
        global step budget (``max_num_scheduled_tokens``) normally freezes
        the plan in RUNNING order.  The isolated owner-FULL experiment has
        one additional, deliberately narrow rule: an exact one-request-per-
        owner prefill cohort at the same logical position advances in equal
        chunks.  This prevents an undersized global token budget from
        splitting one balanced wave into permanently phase-shifted owner
        subsets before decode.  Ragged, mixed, incomplete, and non-graph
        batches retain ordinary RUNNING-order semantics.  At the horizon
        with work remaining, EXTEND is issued and the request stalls until
        its receipt is applied.  Plans are logical payload only: nothing is
        dispatched for execution by this slice.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None
        balanced_prefill = self._owner_balanced_prefill_plans()
        if balanced_prefill is not None:
            self._owner_token_plans.update(balanced_prefill)
            return

        budget = self.max_num_scheduled_tokens
        for request in self.running:
            if request.is_finished():
                continue
            request_id = request.request_id
            key = self._owner_key.get(request_id)
            if key is None:
                continue
            if self._owner_pending_command.get(request_id) is not None:
                continue
            if (
                coordinator.is_release_pending(key)
                or coordinator.is_released(key)
                or coordinator.is_preempted(key)
            ):
                continue
            plan = self._owner_plan_num_new_tokens(request)
            if plan == 0:
                self._owner_token_plans[request_id] = 0
                if request.num_computed_tokens < request.num_tokens:
                    # At the granted horizon with work remaining: EXTEND and
                    # stall.
                    self._issue_owner_extend(request, key)
                continue
            if budget <= 0:
                # Global step budget exhausted in RUNNING order: this and
                # later requests wait for a future step.
                self._owner_token_plans[request_id] = 0
                continue
            plan = min(plan, budget)
            self._owner_token_plans[request_id] = plan
            budget -= plan

    def _owner_window_running_pass(self) -> None:
        """Plan one phase-isolated scheduler window.

        Decode windows keep at most one request per owner in stable owner
        order across acknowledged steps.  Exact world-size cohorts are FULL
        graph eligible; partial cohorts use the existing non-FULL fallback so
        low-load and tail traffic cannot starve.  Prefill work is frozen into
        bounded waves and never mixed with decode in one model invocation.
        This method only selects and plans; :meth:`update_from_output` is the
        sole authority that commits phase transitions.
        """
        state = self._owner_window_state
        if state.inflight is not None:
            raise RuntimeError(
                "request-owned window cannot schedule another positive-token "
                f"step before step {state.inflight.step_seq} is acknowledged."
            )

        self._owner_window_reconcile()
        if state.phase is None:
            decode_slots = self._owner_window_form_decode_slots()
            if len(decode_slots) == self.parallel_config.world_size:
                state.phase = _OwnerWindowPhase.DECODE
                state.decode_slots = decode_slots
                state.yielded_decode_slots = ()
                state.suspended_decode_slots = ()
                state.decode_steps = 0
            elif state.phase is None:
                # Before committing a partial fallback, give runnable prefill
                # a separate wave: it may fill the missing owner slots and
                # produce the exact FULL cohort on the following boundary.
                prefill_wave = self._owner_window_form_prefill_wave()
                if prefill_wave:
                    state.phase = _OwnerWindowPhase.PREFILL
                    state.prefill_wave = prefill_wave
                    policy = self._owner_window_policy
                    assert policy is not None
                    policy.start_prefill(prefill_wave, reason="initial-prefill")
                elif decode_slots:
                    state.phase = _OwnerWindowPhase.DECODE
                    state.decode_slots = decode_slots
                    state.yielded_decode_slots = ()
                    state.suspended_decode_slots = ()
                    state.decode_steps = 0

        if state.phase is _OwnerWindowPhase.DECODE:
            self._owner_window_plan_decode(state.decode_slots)
        elif state.phase is _OwnerWindowPhase.PREFILL:
            self._owner_window_plan_prefill(state.prefill_wave)

    def _owner_window_live_request(
        self,
        key: OwnerLeaseKey,
    ) -> Request | None:
        """Return the live RUNNING request for an exact lease incarnation."""
        request = self.requests.get(key.request_id)
        if (
            request is None
            or request.is_finished()
            or request.status is not RequestStatus.RUNNING
            or self._owner_key.get(key.request_id) != key
        ):
            return None
        return request

    def _owner_window_request_available(self, request: Request) -> bool:
        """Whether a RUNNING lease may remain in a scheduler window."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        request_id = request.request_id
        key = self._owner_key.get(request_id)
        available = bool(
            key is not None
            and self._owner_pending_command.get(request_id) is None
            and not coordinator.is_release_pending(key)
            and not coordinator.is_released(key)
            and not coordinator.is_preempted(key)
        )
        if not available or not self._owner_readiness_seen:
            return available
        assert key is not None
        fact = self._owner_readiness.get(key)
        if fact is None or fact.state is not OwnerResidencyState.HOT:
            return False
        if request.num_computed_tokens < request.num_prompt_tokens:
            return fact.hot_for_prefill
        return fact.hot_for_decode

    def _owner_window_reconcile(self) -> None:
        """Discard phase membership only at an acknowledged boundary."""
        state = self._owner_window_state
        assert state.inflight is None
        if state.phase is _OwnerWindowPhase.DECODE:
            requests = [
                self._owner_window_live_request(key) for key in state.decode_slots
            ]
            if any(request is None for request in requests):
                state.phase = None
                state.decode_slots = ()
                state.decode_steps = 0
                state.suspended_decode_slots = ()
                policy = self._owner_window_policy
                assert policy is not None
                policy.reset_decode_observation()
            elif any(
                request.num_computed_tokens < request.num_prompt_tokens
                or not self._owner_window_request_available(request)
                for request in requests
                if request is not None
            ):
                # Resume/preempt reset or a malformed transition: never emit a
                # prefill token under a decode window identity.
                state.phase = None
                state.decode_slots = ()
                state.decode_steps = 0
                state.suspended_decode_slots = ()
                policy = self._owner_window_policy
                assert policy is not None
                policy.reset_decode_observation()
        elif state.phase is _OwnerWindowPhase.PREFILL:
            live = tuple(
                key
                for key in state.prefill_wave
                if (request := self._owner_window_live_request(key)) is not None
                and request.num_computed_tokens < request.num_prompt_tokens
                and self._owner_window_request_available(request)
            )
            if live:
                state.prefill_wave = live
            else:
                state.phase = None
                state.prefill_wave = ()
                policy = self._owner_window_policy
                assert policy is not None
                policy.cancel_prefill()

    def _owner_window_form_decode_slots(self) -> tuple[OwnerLeaseKey, ...]:
        """Return one owner-indexed decode cohort, exact when available."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        world_size = self.parallel_config.world_size
        suspended = self._owner_window_state.suspended_decode_slots
        if len(suspended) == world_size:
            suspended_requests = [
                self._owner_window_live_request(key) for key in suspended
            ]
            if all(
                request is not None
                and request.num_computed_tokens >= request.num_prompt_tokens
                and self._owner_window_request_available(request)
                for request in suspended_requests
            ):
                return suspended
        slots: list[OwnerLeaseKey | None] = [None] * world_size
        alternates: list[OwnerLeaseKey | None] = [None] * world_size
        yielded = set(self._owner_window_state.yielded_decode_slots)
        for request in self.running:
            if (
                request.is_finished()
                or request.num_computed_tokens < request.num_prompt_tokens
                or not self._owner_window_request_available(request)
            ):
                continue
            key = self._owner_key.get(request.request_id)
            if key is None:
                continue
            owner = coordinator.owner_of(key)
            if owner is None:
                continue
            if slots[owner] is None:
                slots[owner] = key
            if key not in yielded and alternates[owner] is None:
                alternates[owner] = key
        return tuple(
            alternate if alternate is not None else slot
            for slot, alternate in zip(slots, alternates)
            if slot is not None
        )

    def _owner_window_form_prefill_wave(self) -> tuple[OwnerLeaseKey, ...]:
        """Freeze at most one runnable prefill per owner in owner order."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        slots: list[OwnerLeaseKey | None] = [None] * self.parallel_config.world_size
        for request in self.running:
            if (
                request.is_finished()
                or request.num_computed_tokens >= request.num_prompt_tokens
                or not self._owner_window_request_available(request)
            ):
                continue
            key = self._owner_key.get(request.request_id)
            if key is None:
                continue
            owner = coordinator.owner_of(key)
            if owner is None or slots[owner] is not None:
                continue
            slots[owner] = key
        return tuple(key for key in slots if key is not None)

    def _owner_window_plan_decode(
        self,
        slots: tuple[OwnerLeaseKey, ...],
    ) -> None:
        """Plan exactly one decode token per fixed owner slot."""
        if not slots or self.max_num_scheduled_tokens < len(slots):
            return
        requests = [self._owner_window_live_request(key) for key in slots]
        if any(request is None for request in requests):
            return
        for request in requests:
            assert request is not None
            if (
                request.num_computed_tokens < request.num_prompt_tokens
                or not self._owner_window_request_available(request)
            ):
                return
        plans = [self._owner_plan_num_new_tokens(request) for request in requests]
        if any(plan <= 0 for plan in plans):
            for request, plan in zip(requests, plans):
                if plan <= 0 and request.num_computed_tokens < request.num_tokens:
                    self._issue_owner_extend(
                        request,
                        self._owner_key[request.request_id],
                    )
            return
        self._owner_token_plans.update(
            {request.request_id: 1 for request in requests if request is not None}
        )

    def _owner_window_plan_prefill(
        self,
        wave: tuple[OwnerLeaseKey, ...],
    ) -> None:
        """Plan one bounded prefill wave without any decode request."""
        requests = [self._owner_window_live_request(key) for key in wave]
        requests = [
            request
            for request in requests
            if request is not None
            and request.num_computed_tokens < request.num_prompt_tokens
            and self._owner_window_request_available(request)
        ]
        if not requests:
            return
        plans = [self._owner_plan_num_new_tokens(request) for request in requests]
        if any(plan <= 0 for plan in plans):
            for request, plan in zip(requests, plans):
                if plan <= 0 and request.num_computed_tokens < request.num_tokens:
                    self._issue_owner_extend(
                        request,
                        self._owner_key[request.request_id],
                    )
            return
        per_request = self.max_num_scheduled_tokens // len(requests)
        if per_request <= 0:
            return
        for request, runnable_plan in zip(requests, plans):
            plan = min(
                runnable_plan,
                request.num_prompt_tokens - request.num_computed_tokens,
                per_request,
            )
            if plan > 0:
                self._owner_token_plans[request.request_id] = plan

    def _owner_window_record_step(self, output: SchedulerOutput) -> None:
        """Freeze one positive-token phase step before optimistic mutation."""
        if not output.num_scheduled_tokens:
            return
        state = self._owner_window_state
        if state.inflight is not None or state.phase is None:
            raise RuntimeError("request-owned window has invalid in-flight state.")
        expected = (
            state.decode_slots
            if state.phase is _OwnerWindowPhase.DECODE
            else state.prefill_wave
        )
        scheduled_ids = set(output.num_scheduled_tokens)
        members = tuple(key for key in expected if key.request_id in scheduled_ids)
        if {key.request_id for key in members} != scheduled_ids:
            raise RuntimeError(
                "request-owned window scheduled requests outside its frozen "
                f"{state.phase.name.lower()} cohort."
            )
        state.inflight = _OwnerWindowStep(
            step_seq=output.step_seq,
            phase=state.phase,
            members=members,
            num_scheduled_tokens=tuple(output.num_scheduled_tokens.items()),
        )

    def _owner_window_ack_step(self, output: SchedulerOutput) -> None:
        """Commit a completed window step after output/lifecycle mutation."""
        state = self._owner_window_state
        inflight = state.inflight
        if not output.num_scheduled_tokens:
            if inflight is not None:
                raise RuntimeError(
                    "request-owned command-only output cannot acknowledge a "
                    "positive-token window step."
                )
            return
        if (
            inflight is None
            or inflight.step_seq != output.step_seq
            or dict(inflight.num_scheduled_tokens) != output.num_scheduled_tokens
        ):
            raise RuntimeError(
                "request-owned window output does not match the in-flight step."
            )
        state.inflight = None
        if inflight.phase is _OwnerWindowPhase.DECODE:
            state.decode_steps += 1
            if len(state.decode_slots) < self.parallel_config.world_size:
                # Partial decode is a liveness fallback, not a captured lane.
                # Re-form it every acknowledged step so newly runnable work
                # can fill missing owners rather than waiting for this tail
                # cohort to finish.
                state.yielded_decode_slots = state.decode_slots
                state.phase = None
                state.decode_slots = ()
                state.decode_steps = 0
                policy = self._owner_window_policy
                assert policy is not None
                policy.reset_decode_observation()
                return
            policy = self._owner_window_policy
            assert policy is not None
            decision = policy.ack_step(
                OwnerWindowPolicyPhase.DECODE,
                self._owner_window_readiness(),
                positive_tokens=True,
            )
            decode_waits = any(
                not request.is_finished()
                and request.num_computed_tokens >= request.num_prompt_tokens
                and (key := self._owner_key.get(request.request_id)) is not None
                and key not in state.decode_slots
                and self._owner_pending_command.get(request.request_id) is None
                and self.owner_coordinator is not None
                and self.owner_coordinator.runnable_num_tokens_of(key) is not None
                for request in self.requests.values()
            )
            if decision.phase is OwnerWindowPolicyPhase.PREFILL:
                state.suspended_decode_slots = state.decode_slots
                state.decode_slots = ()
                state.decode_steps = 0
                state.prefill_wave = decision.prefill_wave
                state.phase = _OwnerWindowPhase.PREFILL
            elif policy.decode_steps == 0 and decode_waits:
                state.yielded_decode_slots = state.decode_slots
                state.decode_slots = ()
                state.decode_steps = 0
                state.phase = None
        else:
            state.decode_steps = 0
            policy = self._owner_window_policy
            assert policy is not None
            decision = policy.ack_step(
                OwnerWindowPolicyPhase.PREFILL,
                self._owner_window_readiness(),
                positive_tokens=True,
            )
            state.prefill_wave = decision.prefill_wave
            if decision.phase is OwnerWindowPolicyPhase.DECODE:
                state.phase = None
                state.prefill_wave = ()

    def _owner_window_readiness(self) -> OwnerWindowReadiness:
        """Build the controller view from fenced worker readiness facts."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        world_size = self.parallel_config.world_size
        hot_decode = [0] * world_size
        restoring = [0] * world_size
        host_restorable = [0] * world_size
        candidates: list[OwnerPrefillCandidate] = []
        for request in self.requests.values():
            if request.is_finished():
                continue
            key = self._owner_key.get(request.request_id)
            if key is None:
                continue
            owner_id = coordinator.owner_of(key)
            if owner_id is None:
                continue
            fact = self._owner_readiness.get(key)
            if fact is not None:
                if fact.state is OwnerResidencyState.RESTORING:
                    restoring[owner_id] += 1
                elif fact.state is OwnerResidencyState.COLD:
                    host_restorable[owner_id] += 1
                elif (
                    fact.hot_for_decode
                    and request.num_computed_tokens >= request.num_prompt_tokens
                ):
                    hot_decode[owner_id] += 1
            if (
                request.status is RequestStatus.RUNNING
                and request.num_computed_tokens < request.num_prompt_tokens
                and self._owner_pending_command.get(request.request_id) is None
                and coordinator.runnable_num_tokens_of(key) is not None
                and (
                    not self._owner_readiness_seen
                    or (
                        fact is not None
                        and fact.state is OwnerResidencyState.HOT
                        and fact.hot_for_prefill
                    )
                )
            ):
                candidates.append(
                    OwnerPrefillCandidate(
                        key=key,
                        owner_id=owner_id,
                        wait_steps=max(
                            0,
                            self.current_step
                            - self._owner_wait_started_step.get(key, self.current_step),
                        ),
                    )
                )
        return OwnerWindowReadiness(
            hot_decode_by_owner=tuple(hot_decode),
            restoring_by_owner=tuple(restoring),
            host_restorable_by_owner=tuple(host_restorable),
            prefill_candidates=tuple(candidates),
        )

    def _owner_balanced_prefill_plans(self) -> dict[str, int] | None:
        """Return one lockstep prefill chunk for the exact owner FULL cohort.

        The first FULL decode key is finite: exactly one row per process-
        global owner.  With equal-length prompts, greedily consuming the
        global token budget in request order can turn that future decode lane
        into two phase-shifted subsets even though every NPU has useful
        prefill work.  When *all* active RUNNING requests form that exact
        cohort and have equal positive prefill work at the same position,
        divide the budget evenly and intentionally leave any indivisible
        remainder unused.  This is scheduler-level wave formation, not
        owner-side token-budget coordination.

        Returning ``None`` preserves the ordinary policy for every other
        shape.  In particular, a budget smaller than the owner count falls
        back rather than stalling the engine with an all-zero plan.
        """
        if not getattr(
            self.scheduler_config,
            "enable_request_owned_graph",
            False,
        ):
            return None

        coordinator = self.owner_coordinator
        assert coordinator is not None
        cohort = [request for request in self.running if not request.is_finished()]
        world_size = self.parallel_config.world_size
        if len(cohort) != world_size or self.max_num_scheduled_tokens < world_size:
            return None

        owners: list[int] = []
        positions: list[int] = []
        plans: list[int] = []
        for request in cohort:
            request_id = request.request_id
            key = self._owner_key.get(request_id)
            if (
                key is None
                or self._owner_pending_command.get(request_id) is not None
                or coordinator.is_release_pending(key)
                or coordinator.is_released(key)
                or coordinator.is_preempted(key)
                or request.num_computed_tokens >= request.num_prompt_tokens
            ):
                return None
            owner = coordinator.owner_of(key)
            if owner is None:
                return None
            plan = self._owner_plan_num_new_tokens(request)
            if plan <= 0:
                return None
            owners.append(owner)
            positions.append(request.num_computed_tokens)
            plans.append(plan)

        if (
            sorted(owners) != list(range(world_size))
            or len(set(positions)) != 1
            or len(set(plans)) != 1
        ):
            return None

        per_request = min(
            plans[0],
            self.max_num_scheduled_tokens // world_size,
        )
        if per_request <= 0:
            return None
        return {request.request_id: per_request for request in cohort}

    def _owner_key_for(self, request: Request) -> OwnerLeaseKey:
        """Return (and remember) the lease key for a request incarnation."""
        request_id = request.request_id
        key = self._owner_key.get(request_id)
        if key is None:
            epoch = self._owner_epoch.get(request_id, 0)
            key = OwnerLeaseKey(request_id=request_id, owner_epoch=epoch)
            self._owner_key[request_id] = key
            self._owner_epoch[request_id] = epoch
            self._owner_wait_started_step[key] = self.current_step
        return key

    def _roll_owner_epoch(self, request: Request) -> OwnerLeaseKey:
        """Fence a reused request id forward to a fresh lease epoch."""
        request_id = request.request_id
        old_key = self._owner_key.get(request_id)
        epoch = self._owner_epoch.get(request_id, 0) + 1
        key = OwnerLeaseKey(request_id=request_id, owner_epoch=epoch)
        self._owner_key[request_id] = key
        self._owner_epoch[request_id] = epoch
        if old_key is not None:
            self._owner_readiness.pop(old_key, None)
            self._owner_wait_started_step.pop(old_key, None)
        self._owner_wait_started_step[key] = self.current_step
        return key

    def _owner_reserve_required(self, request: Request) -> int:
        """Exact required count for a fresh RESERVE.

        The isolated owner-graph path reserves the request's already-known
        bounded lifetime (prompt plus maximum generated tokens).  Decode can
        then advance under the original authorization instead of alternating
        every token-bearing graph replay with an EXTEND control heartbeat.
        The non-graph control-plane path retains its earlier chunk-scaled
        semantics.
        """
        decode_chunk = getattr(
            self.scheduler_config,
            "request_owned_decode_reservation_tokens",
            None,
        )
        if decode_chunk is not None:
            return min(
                request.num_prompt_tokens + request.max_tokens,
                request.num_prompt_tokens + decode_chunk,
            )
        if getattr(
            self.scheduler_config,
            "enable_request_owned_graph",
            False,
        ):
            return request.num_prompt_tokens + request.max_tokens
        remaining = request.num_tokens - request.num_computed_tokens
        return min(self.max_num_scheduled_tokens, remaining)

    def _owner_resume_required(self, request: Request) -> int:
        """Chunk-scaled RESERVE required count on resume.

        Never below the already-published count: the worker refuses a resume
        that would regress its honored (published) fence.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None
        published = coordinator.published_num_tokens(
            self._owner_key[request.request_id]
        )
        decode_chunk = getattr(
            self.scheduler_config,
            "request_owned_decode_reservation_tokens",
            None,
        )
        if decode_chunk is not None:
            return min(
                request.num_prompt_tokens + request.max_tokens,
                max(published, request.num_prompt_tokens + decode_chunk),
            )
        if getattr(
            self.scheduler_config,
            "enable_request_owned_graph",
            False,
        ):
            return max(
                published,
                request.num_prompt_tokens + request.max_tokens,
            )
        return min(
            request.num_tokens,
            max(published, self.max_num_scheduled_tokens),
        )

    def _owner_extend_required(self, request: Request) -> int:
        """Chunk-scaled cumulative EXTEND requirement past the horizon.

        ``required_num_tokens`` is the new exclusive upper bound the lease
        asks the worker to honor, so the next chunk starts at the current
        computed position (which equals the granted horizon when extending).
        """
        decode_chunk = getattr(
            self.scheduler_config,
            "request_owned_decode_reservation_tokens",
            None,
        )
        increment = (
            decode_chunk if decode_chunk is not None else self.max_num_scheduled_tokens
        )
        upper_bound = (
            request.num_prompt_tokens + request.max_tokens
            if decode_chunk is not None
            else request.num_tokens
        )
        return min(
            upper_bound,
            request.num_computed_tokens + increment,
        )

    def _assign_owner_and_reserve(self, request: Request, key: OwnerLeaseKey) -> None:
        """Assign the scheduler-computed owner and issue the RESERVE.

        The scheduler picks the first-admission owner itself from
        worker-confirmed physical pool facts (or lease counts before any
        snapshot) and forces it through the coordinator with
        ``projected_work=0`` so token units never mix with block facts.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None
        required = self._owner_reserve_required(request)
        owner, projected_blocks = self._select_owner_for_admission(required)
        try:
            coordinator.assign(
                key,
                required_num_tokens=required,
                explicit_owner=owner,
                projected_work=0,
            )
        except EpochFenceError:
            # Request-id reuse raced a higher fence: roll forward once.
            key = self._roll_owner_epoch(request)
            coordinator.assign(
                key,
                required_num_tokens=required,
                explicit_owner=owner,
                projected_work=0,
            )
        self._owner_admission_projected_blocks[owner] += projected_blocks
        self._issue_owner_reserve(request, key)

    @staticmethod
    def _owner_projected_block_demand(
        snapshot: OwnerCachePoolSnapshot,
        required_num_tokens: int,
    ) -> int:
        """Conservatively project unified-pool blocks for a fresh horizon.

        Every request owns one table in every KV group, while the groups use
        heterogeneous effective token capacities.  Their block allocations
        all consume the same physical pool, so the projected pool demand is
        the sum of each group's ceiling.  Empty group metadata is retained
        for compatibility with early/host-only protocol snapshots and still
        charges one block for a nonempty reservation.
        """
        if required_num_tokens <= 0:
            return 0
        if not snapshot.groups:
            return 1
        return sum(
            (required_num_tokens + group.effective_tokens_per_block - 1)
            // group.effective_tokens_per_block
            for group in snapshot.groups
        )

    def _select_owner_for_admission(
        self,
        required_num_tokens: int,
    ) -> tuple[int, int]:
        """Pick the owner for a fresh G2 first admission.

        With physical pool snapshots present, the owner with the greatest
        post-admission projected free capacity wins.  The projection charges
        every provisional choice already made in this scheduler pass, so a
        small stale free-block advantage cannot funnel an entire admission
        wave onto one rank.  Ties break by live/sticky lease count and stable
        global rank.  Without snapshots the choice falls back to lease-count
        balance then rank.  No block or pool IDs are consulted.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None
        ranks = range(self.parallel_config.world_size)
        if self._owner_pool_snapshots:
            snapshots = self._owner_pool_snapshots
            projected = self._owner_admission_projected_blocks
            demands = {
                owner_id: self._owner_projected_block_demand(
                    snapshots[owner_id], required_num_tokens
                )
                for owner_id in ranks
            }

            def score(owner_id: int) -> tuple[int, int, int]:
                return (
                    -(
                        snapshots[owner_id].free_blocks
                        - projected[owner_id]
                        - demands[owner_id]
                    ),
                    coordinator.live_lease_count(owner_id),
                    owner_id,
                )
        else:
            demands = {owner_id: 0 for owner_id in ranks}

            def score(owner_id: int) -> tuple[int, int]:
                return (coordinator.live_lease_count(owner_id), owner_id)

        owner = min(ranks, key=score)
        return owner, demands[owner]

    def _issue_owner_reserve(self, request: Request, key: OwnerLeaseKey) -> None:
        """Issue RESERVE (first admission) or RESUME (PREEMPTED) with the
        matching allocation descriptor and record the in-flight command."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        if request.status == RequestStatus.PREEMPTED:
            command = coordinator.resume(key, self._owner_resume_required(request))
            status = OwnerAdmissionStatus.PREEMPTED
            num_computed_tokens = (
                request.num_computed_tokens
                if getattr(
                    self.scheduler_config,
                    "enable_request_owned_kv_offload",
                    False,
                )
                else 0
            )
        else:
            command = coordinator.reserve(key, self._owner_reserve_required(request))
            status = OwnerAdmissionStatus.WAITING
            num_computed_tokens = request.num_computed_tokens
        allocation = OwnerAllocationDescriptor(
            key=key,
            num_prompt_tokens=request.num_prompt_tokens,
            num_computed_tokens=num_computed_tokens,
            num_tokens=command.required_num_tokens,
            status=status,
        )
        self._owner_outbox.append(replace(command, allocation=allocation))
        self._owner_pending_command[request.request_id] = (
            command.command_seq,
            command.kind,
        )

    def _issue_owner_extend(self, request: Request, key: OwnerLeaseKey) -> None:
        """Issue EXTEND at the horizon and record the in-flight command."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        command = coordinator.extend(key, self._owner_extend_required(request))
        self._owner_outbox.append(command)
        self._owner_pending_command[request.request_id] = (
            command.command_seq,
            command.kind,
        )

    def _issue_owner_restore(self, request: Request, key: OwnerLeaseKey) -> None:
        """Issue one synchronous bulk RESTORE before resume admission."""

        coordinator = self.owner_coordinator
        assert coordinator is not None
        command = coordinator.restore(key, self._owner_resume_required(request))
        self._owner_outbox.append(command)
        self._owner_pending_command[request.request_id] = (
            command.command_seq,
            command.kind,
        )

    def _owner_plan_num_new_tokens(self, request: Request) -> int:
        """Internal RUNNING token plan, strictly capped by the horizon.

        The granted ``runnable_num_tokens`` is an exclusive upper bound
        (positions ``0 <= p < runnable`` are runnable), so the plan never
        pushes ``num_computed_tokens`` to or past the horizon.  The plan is
        constructed for tests/future slices only and is never dispatched.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None
        key = self._owner_key[request.request_id]
        horizon = coordinator.runnable_num_tokens_of(key)
        if horizon is None:
            return 0
        remaining = request.num_tokens - request.num_computed_tokens
        desired = min(self.max_num_scheduled_tokens, remaining)
        return max(0, min(desired, horizon - request.num_computed_tokens))

    def _promote_owner_request(
        self, request: Request, key: OwnerLeaseKey, scheduled_timestamp: float
    ) -> None:
        """Promote an accepted lease to RUNNING with its sticky owner."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        request.attention_owner = coordinator.owner_of(key)
        request.attention_owner_epoch = key.owner_epoch
        # G3: remember whether this promotion still owes its first dispatch
        # or resume payload.  Preserve a pending WAITING marker if an
        # undispatched request was preempted before receiving token budget:
        # no worker has state for it yet, so it still needs NewRequestData.
        prior = self._owner_pending_dispatch.get(request.request_id)
        if prior is not RequestStatus.WAITING:
            self._owner_pending_dispatch[request.request_id] = request.status
        request.status = RequestStatus.RUNNING
        self.running.append(request)
        self._inflight_prefills.discard(request)
        if self.log_stats:
            request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)

    def _drain_owner_outbox(self) -> list[OwnerCommand]:
        """Emit outbox commands in per-owner order, each exactly once."""
        commands: list[OwnerCommand] = []
        for command in self._owner_outbox:
            owner = command.owner_id
            if command.command_seq <= self._owner_emitted_command_seq.get(owner, 0):
                continue
            commands.append(command)
            self._owner_emitted_command_seq[owner] = command.command_seq
        self._owner_outbox = []
        commands.sort(key=lambda command: (command.owner_id, command.command_seq))
        return commands

    def _preempt_request_owned(self, request: Request, timestamp: float) -> None:
        """Preempt with sticky ownership and optional durable host KV."""
        assert request.status == RequestStatus.RUNNING
        coordinator = self.owner_coordinator
        assert coordinator is not None
        key = self._owner_key[request.request_id]
        command = coordinator.preempt(key)
        self._owner_outbox.append(command)
        self._owner_pending_command[request.request_id] = (
            command.command_seq,
            command.kind,
        )
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        if not getattr(
            self.scheduler_config,
            "enable_request_owned_kv_offload",
            False,
        ):
            # Without durable host KV the resumed request must recompute.
            request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _finish_owner_lease(self, request: Request) -> None:
        """Finish/abort the owner lease.

        Before admission the provisional assignment is abandoned (refunded,
        retryable); after admission an idempotent RELEASE keeps command-only
        liveness until its accepted receipt.  Sticky owner+epoch facts are
        preserved so request-id reuse fences forward via the coordinator.
        """
        coordinator = self.owner_coordinator
        if coordinator is None:
            return
        request_id = request.request_id
        self._owner_pending_dispatch.pop(request_id, None)
        key = self._owner_key.get(request_id)
        if key is None or coordinator.owner_of(key) is None:
            return
        if coordinator.is_released(key):
            return
        if coordinator.runnable_num_tokens_of(key) is None:
            # Before admission: refund the provisional assignment.
            coordinator.abandon(key)
            self._owner_key.pop(request_id, None)
            self._owner_pending_command.pop(request_id, None)
            self._owner_readiness.pop(key, None)
            self._owner_wait_started_step.pop(key, None)
            return
        # After admission: idempotent RELEASE while release_pending.
        if not coordinator.is_release_pending(key):
            command = coordinator.finish(key)
            self._owner_outbox.append(command)
            self._owner_pending_command[request_id] = (
                command.command_seq,
                command.kind,
            )

    def _has_pending_owner_control(self) -> bool:
        """True while an owner command is in flight or a RELEASE is pending.

        Keeps the engine stepping (command-only liveness) until the accepted
        release receipt lands, even after the request left the queues.
        """
        coordinator = self.owner_coordinator
        if coordinator is None:
            return False
        if self._owner_pending_command:
            return True
        return any(
            coordinator.owner_of(key) is not None
            and coordinator.is_release_pending(key)
            for key in self._owner_key.values()
        )

    def _apply_request_owned_receipts(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> None:
        """Validate the all-worker receipt envelope and apply every event.

        Runs before ordinary output/request mutation.  Structural violations
        fail closed exactly like the G1 ingress boundary; structurally valid
        nonempty receipts are applied to the coordinator.
        """
        # Envelope violations fail closed exactly like the G1 ingress
        # boundary, before any authority/coordinator access.
        step_seq = scheduler_output.step_seq
        if isinstance(step_seq, bool) or not isinstance(step_seq, int) or step_seq <= 0:
            raise RuntimeError(
                "request-owned receipt ingress requires a positive non-bool "
                f"step_seq, got {step_seq!r}."
            )
        batches = model_runner_output.owner_receipt_batches
        if batches is None:
            raise RuntimeError(
                "request-owned receipt ingress is missing all-worker receipt batches."
            )
        expected_ranks = list(range(self.parallel_config.world_size))
        ranks = [batch.owner_rank for batch in batches]
        if (
            any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
            or len(ranks) != len(expected_ranks)
            or sorted(ranks) != expected_ranks
        ):
            raise RuntimeError(
                "request-owned receipt ingress requires exactly one batch per "
                f"process-global owner rank {expected_ranks}, got {ranks}."
            )
        for batch in batches:
            if (
                isinstance(batch.emitted_step_seq, bool)
                or not isinstance(batch.emitted_step_seq, int)
                or batch.emitted_step_seq != step_seq
            ):
                raise RuntimeError(
                    "request-owned receipt ingress got emitted_step_seq "
                    f"{batch.emitted_step_seq} from owner {batch.owner_rank}, "
                    f"expected {step_seq}."
                )
            for event in batch.events:
                if (
                    isinstance(event.owner_id, bool)
                    or not isinstance(event.owner_id, int)
                    or event.owner_id != batch.owner_rank
                ):
                    raise RuntimeError(
                        "request-owned receipt ingress got an event for owner "
                        f"{event.owner_id} in owner {batch.owner_rank}'s batch."
                    )
        coordinator = getattr(self, "owner_coordinator", None)
        if coordinator is None:
            raise RuntimeError(
                "request-owned receipt ingress requires the G2 owner "
                "coordinator; the scheduler was not initialized for it."
            )
        for batch in batches:
            for fact in getattr(batch, "readiness", ()):
                if (
                    self._owner_key.get(fact.key.request_id) == fact.key
                    and coordinator.owner_of(fact.key) != fact.owner_id
                ):
                    raise RuntimeError(
                        "request-owned readiness owner does not match the "
                        f"coordinator for {fact.key}."
                    )
        self._validate_owner_readiness_coverage(batches)
        self._store_owner_pool_snapshots(batches)
        for batch in batches:
            for event in batch.events:
                self._apply_owner_receipt(event)
        self._store_owner_readiness(batches)

    def _validate_owner_readiness_coverage(
        self, batches: list[OwnerReceiptBatch]
    ) -> None:
        """Validate the post-event live set before mutating receipt state."""
        coordinator = self.owner_coordinator
        assert coordinator is not None
        authoritative: list[tuple[OwnerReceipt, OwnerCommandKind]] = []
        for batch in batches:
            for event in batch.events:
                if self._owner_key.get(event.key.request_id) != event.key:
                    continue
                pending = self._owner_pending_command.get(event.key.request_id)
                if (
                    pending is None
                    or pending[0] != event.command_seq
                    or coordinator.owner_of(event.key) != event.owner_id
                ):
                    continue
                command_kind = pending[1]
                if event.released and command_kind is not OwnerCommandKind.RELEASE:
                    raise RuntimeError(
                        "request-owned non-RELEASE receipt cannot claim release "
                        f"for {event.key}."
                    )
                if event.accepted and command_kind is OwnerCommandKind.RELEASE:
                    if not event.released:
                        raise RuntimeError(
                            "accepted request-owned RELEASE receipt must confirm "
                            f"release for {event.key}."
                        )
                elif event.accepted and event.runnable_num_tokens is None:
                    raise RuntimeError(
                        "accepted request-owned non-RELEASE receipt must carry "
                        f"a runnable grant for {event.key}."
                    )
                authoritative.append((event, command_kind))
        if not getattr(self.scheduler_config, "enable_request_owned_windows", False):
            return
        expected = {
            key
            for key in self._owner_key.values()
            if coordinator.owner_of(key) is not None
            and coordinator.runnable_num_tokens_of(key) is not None
            and not coordinator.is_release_pending(key)
            and not coordinator.is_released(key)
        }
        for event, command_kind in authoritative:
            if not event.accepted:
                continue
            if command_kind is OwnerCommandKind.RELEASE:
                expected.discard(event.key)
            else:
                expected.add(event.key)
        facts = {
            fact.key
            for batch in batches
            for fact in getattr(batch, "readiness", ())
            if self._owner_key.get(fact.key.request_id) == fact.key
            and coordinator.owner_of(fact.key) == fact.owner_id
        }
        missing = sorted(
            expected - facts,
            key=lambda key: (key.request_id, key.owner_epoch),
        )
        if missing:
            raise RuntimeError(
                f"request-owned readiness snapshot is missing live lease(s): {missing}."
            )

    def _store_owner_readiness(self, batches: list[OwnerReceiptBatch]) -> None:
        """Replace each owner's snapshot after full envelope validation."""
        readiness_enabled = bool(
            getattr(self.scheduler_config, "enable_request_owned_windows", False)
        )
        next_readiness: dict[OwnerLeaseKey, OwnerReadinessReceipt] = {}
        for batch in batches:
            for fact in getattr(batch, "readiness", ()):
                if (
                    self._owner_key.get(fact.key.request_id) == fact.key
                    and self.owner_coordinator is not None
                    and self.owner_coordinator.owner_of(fact.key) == fact.owner_id
                ):
                    next_readiness[fact.key] = fact
                elif self._owner_key.get(fact.key.request_id) == fact.key:
                    raise RuntimeError(
                        "request-owned readiness owner does not match the "
                        f"coordinator for {fact.key}."
                    )
        if not readiness_enabled:
            return
        self._owner_readiness = next_readiness
        self._owner_readiness_seen = True

    def _store_owner_pool_snapshots(self, batches: list[OwnerReceiptBatch]) -> None:
        """Record the latest physical pool snapshot per global rank.

        Runs only after the full receipt envelope validated, so the stored
        snapshots are worker-confirmed facts (and never expose block/pool
        IDs).  All-None envelopes are tolerated before the first physical
        snapshot (the initial/legacy control path); partial envelopes within
        a step and any missing snapshot after the first one was observed
        fail closed, so the scheduler never silently reuses stale pool
        facts.
        """
        non_null = [batch for batch in batches if batch.cache_pool is not None]
        if non_null and len(non_null) != len(batches):
            raise RuntimeError(
                "request-owned receipt ingress requires every owner batch "
                f"to carry a cache_pool snapshot, got {len(non_null)} of "
                f"{len(batches)}."
            )
        if non_null:
            self._owner_pool_snapshot_seen = True
            for batch in batches:
                self._owner_pool_snapshots[batch.owner_rank] = batch.cache_pool
        elif self._owner_pool_snapshot_seen:
            raise RuntimeError(
                "request-owned receipt ingress requires a cache_pool "
                "snapshot for every owner rank once physical snapshots are "
                "in use; got an all-None envelope."
            )

    def _apply_owner_receipt(self, event: OwnerReceipt) -> None:
        """Apply one worker receipt to the coordinator and scheduler state.

        Accepted RESERVE promotes the request (sticky owner + epoch); a
        refused provisional RESERVE abandons and retries without any token
        mutation; an accepted RELEASE completes the incarnation so request-id
        reuse fences forward.
        """
        coordinator = self.owner_coordinator
        assert coordinator is not None
        request_id = event.key.request_id
        key = self._owner_key.get(request_id)
        if key is None or key != event.key:
            # Unknown or fenced-out lease: the coordinator ignores it and the
            # current incarnation's pending command is never touched.
            coordinator.apply_receipt(event)
            return
        if event.owner_id != coordinator.owner_of(key):
            # Wrong-owner receipt for a live lease: the coordinator ignores
            # it (the lease's assigned owner differs), and the in-flight
            # command must stay pending so the genuine worker receipt can
            # still land.  Clearing it here would stall the incarnation.
            coordinator.apply_receipt(event)
            return
        pending = self._owner_pending_command.get(request_id)
        if (
            getattr(
                self.scheduler_config,
                "enable_request_owned_kv_offload",
                False,
            )
            and pending is not None
            and pending == (event.command_seq, OwnerCommandKind.RESTORE)
            and event.accepted
            and event.pending_dma != 0
        ):
            raise RuntimeError(
                "request-owned bulk RESTORE requires a terminal receipt with "
                f"pending_dma=0, got {event.pending_dma!r} for {event.key}."
            )
        if pending is not None and pending[0] == event.command_seq:
            self._owner_pending_command.pop(request_id, None)
        applied = coordinator.apply_receipt(event)
        if not applied:
            if not event.accepted:
                # A matching-current rejected PREEMPT/RELEASE is not
                # recoverable by waiting: the worker fence rejects a
                # duplicate command_seq, and the scheduler never re-issues
                # either command, so the protocol/authority stream has
                # diverged. Fail closed instead of stalling forever.
                if (
                    pending is not None
                    and pending[0] == event.command_seq
                    and pending[1]
                    in (OwnerCommandKind.PREEMPT, OwnerCommandKind.RELEASE)
                ):
                    raise RuntimeError(
                        "worker refused fenced "
                        f"{pending[1].value} for {event.key}: unrecoverable "
                        "protocol/authority divergence"
                    )
                # Refused RESERVE/EXTEND is ordinary backpressure: a
                # provisional RESERVE abandons and retries.
                if (
                    coordinator.owner_of(key) is not None
                    and coordinator.runnable_num_tokens_of(key) is None
                    and not coordinator.is_release_pending(key)
                ):
                    coordinator.abandon(key)
                    self._owner_key.pop(request_id, None)
                    self._owner_readiness.pop(key, None)
                    self._owner_wait_started_step.pop(key, None)
            return
        if not event.accepted:
            return
        if coordinator.is_released(key) and self._owner_key.get(request_id) == key:
            # Accepted release: the incarnation is done.  Fence request-id
            # reuse forward so the next incarnation gets a fresh epoch.
            if key.owner_epoch >= self._owner_epoch.get(request_id, 0):
                self._owner_epoch[request_id] = key.owner_epoch + 1
            self._owner_key.pop(request_id, None)
            self._owner_readiness.pop(key, None)
            self._owner_wait_started_step.pop(key, None)
            return
        # Accepted RESERVE promotes the request (sticky owner + epoch).
        request = self.requests.get(request_id)
        if (
            request is not None
            and request.attention_owner is None
            and coordinator.runnable_num_tokens_of(key) is not None
        ):
            request.attention_owner = coordinator.owner_of(key)
            request.attention_owner_epoch = key.owner_epoch

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's +1 read has also passed it, mirroring
        # the shift the encoder scheduling path applies.
        spec_lookahead = 1 if self.use_eagle else 0

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens + spec_lookahead
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        if self.scheduler_config.enable_request_owned_attention:
            # G2: finish/abort the owner lease (abandon before admission,
            # idempotent RELEASE after) before the request leaves the queues.
            self._finish_owner_lease(request)

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if self.scheduler_config.enable_request_owned_attention:
            # G2: physical KV is owned by the worker-local allocator.  The
            # scheduler never allocated blocks in this mode and must not free
            # them (PREEMPT/RELEASE flow through the owner protocol instead).
            return
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Entries are appended with monotonically non-decreasing fences, so
        stop at the first one that is still pending.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or self._has_pending_owner_control()
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # For async scheduling, any output frames already in flight at
                # preemption time are now stale and must be discarded when they
                # return. num_output_placeholders is exactly that count: 0 if
                # the engine has drained (e.g. pause_generation(keep) waited
                # for idle), 1 for vanilla async mid-step, or 1 + spec/PP frames
                # otherwise.
                request.async_tokens_to_discard = request.num_output_placeholders
                request.num_output_placeholders = 0

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        structured_output_cache_stats = (
            self.structured_output_manager.make_cache_stats()
        )
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            available_kv_cache_memory_bytes=self.available_kv_cache_memory_bytes,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            structured_output_cache_stats=structured_output_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.prefix_cache_event_uploader:
            self.prefix_cache_event_uploader.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_computed_tokens,
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""

        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)
            reclaimable_block_ids = self.connector.take_reclaimable_block_ids()
            if reclaimable_block_ids:
                self.kv_cache_manager.evict_blocks(reclaimable_block_ids)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
