# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import torch
import torch.nn as nn

import vllm.ir
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.request_owned_kv import (
    AllocationResult,
    DeferredFreeResult,
    RequestOwnedKVStore,
    RequestOwnedStepBuildCheckpoint,
    RequestOwnedStepMetadata,
    request_owned_allocation_binding_spec,
)
from vllm.v1.worker.request_owned_offload import (
    OwnerOffloadIdentity,
    OwnerOffloadPlan,
    RequestOwnedBulkOffloadAdapter,
    RequestOwnedBulkRestoreWork,
    RequestOwnedOffloadError,
    make_request_owned_offload_keys,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


class CompilationTimes(NamedTuple):
    language_model: float
    encoder: float


class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

        # IR op priority and torch-wrap state are constant for the worker's
        # lifetime.
        vllm_config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Prepare model for execution through compilation/warmup.

        Returns:
            Compilation times (language_model, encoder) in seconds.
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    def set_request_owned_step_metadata(
        self, metadata: RequestOwnedStepMetadata | None
    ) -> None:
        """Worker-private handoff of the G3 step metadata.

        Called by the wrapper with ``None`` at the start of every
        request-owned call to actively clear stale runner state, and with
        the immutable batch immediately after a successful
        ``build_step_metadata`` for this rank's step.  The metadata is
        fully detached and is delivered as a plain method call: it is never
        attached to a SchedulerOutput or any other wire object, and no wire
        object is mutated.  Unsupported workers fail closed: the default
        raises instead of silently dropping the handoff.  No computed
        progress is marked from this hook; completion is declared later,
        after token sampling."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support the request-owned G3 "
            "step metadata handoff; request-owned attention requires a "
            "worker that implements set_request_owned_step_metadata."
        )

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


@dataclass(frozen=True, slots=True)
class _RequestOwnedDeferredStep:
    """One deferred request-owned sampling step awaiting ``sample_tokens``.

    Captured when the underlying ``execute_model`` returns ``None`` under
    ``enable_request_owned_sampling``: the exact step fence, the trial
    logical manager that must be committed only on success, and the exact
    immutable step metadata that must be marked exactly once.  Nothing is
    marked, flushed, emitted, or committed until ``sample_tokens``
    completes the step; a failed completion never clears the record.
    """

    step_seq: int
    trial_manager: AttentionLeaseManager
    metadata: RequestOwnedStepMetadata

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_seq, bool)
            or not isinstance(self.step_seq, int)
            or self.step_seq <= 0
        ):
            raise TypeError(
                f"step_seq must be a positive non-bool int, got {self.step_seq!r}."
            )
        if not isinstance(self.trial_manager, AttentionLeaseManager):
            raise TypeError(
                "trial_manager must be an AttentionLeaseManager, got "
                f"{type(self.trial_manager).__name__}."
            )
        if not isinstance(self.metadata, RequestOwnedStepMetadata):
            raise TypeError(
                "metadata must be a RequestOwnedStepMetadata, got "
                f"{type(self.metadata).__name__}."
            )


def _normalize_request_owned_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> KVCacheConfig:
    """Rank-local manager config for the request-owned KV store (G4).

    The store's ``KVCacheManager`` needs the same concrete per-group specs
    the scheduler uses (``generate_scheduler_kv_cache_config`` semantics): a
    ``UniformTypeKVCacheSpecs`` wrapper has no registered manager and no
    ``compress_ratio``, so it cannot size the block pool or report
    ``effective_tokens_per_block``.  This helper deep-copies the worker's raw
    config only when uniform groups exist (otherwise the original object is
    returned unchanged so plain configs keep identity) and binds each uniform
    group to its allocation-binding inner spec: the one with the smallest
    positive integer ``compress_ratio`` (absent field defaults to 1), i.e.
    the largest storage footprint.  The raw config handed to the underlying
    worker is never mutated.
    """
    if not any(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_config.kv_cache_groups
    ):
        return kv_cache_config

    normalized = deepcopy(kv_cache_config)
    for group in normalized.kv_cache_groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, UniformTypeKVCacheSpecs):
            continue
        try:
            group.kv_cache_spec = request_owned_allocation_binding_spec(spec)
        except ValueError as exc:
            raise ValueError(
                "Cannot build the request-owned KV store for uniform KV cache "
                f"group {group.layer_names!r}: {exc}"
            ) from exc
    return normalized


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank: int = rpc_rank
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig
        self._request_owned_control_manager: AttentionLeaseManager | None = None
        self._request_owned_kv_store: RequestOwnedKVStore | None = None
        self._request_owned_offload_adapter: RequestOwnedBulkOffloadAdapter | None = (
            None
        )
        #: Immutable worker-local G3 execution metadata of the last step
        #: whose command+publication validation succeeded.  Cleared at the
        #: start of every request-owned call (the concrete worker is
        #: actively cleared with a ``None`` handoff) and handed to the
        #: worker through its private hook; never attached to a scheduler
        #: wire.
        self._request_owned_step_metadata: RequestOwnedStepMetadata | None = None

        #: Pending deferred request-owned sampling step (G3).  Set exactly
        #: when the underlying ``execute_model`` returns ``None`` under
        #: ``enable_request_owned_sampling``; consumed (cleared) only by a
        #: fully successful ``sample_tokens`` completion.  A pending record
        #: rejects the next ``execute_model`` call and any replay without a
        #: pending record, and is never cleared by a failed completion.
        self._request_owned_deferred: _RequestOwnedDeferredStep | None = None

        #: Irreversible request-owned fail-stop latch (G3).  Set only when
        #: the computed-batch mark succeeded but the terminal completion
        #: (flush/emit/pool snapshot) failed afterwards: the step is already
        #: marked in the store and can never be retried, so every further
        #: request-owned call fails closed instead of risking a duplicate
        #: mark.
        self._request_owned_fail_stop: str | None = None

        #: Exact destinations of a bulk RESTORE whose H2D has completed but
        #: whose control step has not committed yet.  The public wrapper
        #: rolls these destinations back on *any* later step failure, leaving
        #: the durable cold image intact so the same RESTORE can be retried.
        self._request_owned_restore_guard: (
            tuple[
                tuple[RequestOwnedBulkRestoreWork, ...],
                RequestOwnedKVStore,
                RequestOwnedStepBuildCheckpoint,
                int,
            ]
            | None
        ) = None

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        assigned_physical_gpu_ids = kwargs.pop("assigned_physical_gpu_ids", None)
        if assigned_physical_gpu_ids is not None:
            vllm_config.parallel_config.assigned_physical_gpu_ids = (
                assigned_physical_gpu_ids
            )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

            # G2: after the underlying worker initializes, bind this rank's
            # physical store when request-owned attention is enabled.  The
            # store reuses one real KVCacheManager over this rank's
            # KVCacheConfig; prefix caching, Eagle, events, and stats are all
            # disabled because the scheduler-side manager stays the only
            # prefix/Eagle authority and this store never publishes block IDs.
            if self.vllm_config.scheduler_config.enable_request_owned_attention:
                self._request_owned_kv_store = self._create_request_owned_kv_store(
                    kv_cache_config
                )
            if getattr(
                self.vllm_config.scheduler_config,
                "enable_request_owned_kv_offload",
                False,
            ):
                self._request_owned_offload_adapter = (
                    self._create_request_owned_offload_adapter()
                )

    def _create_request_owned_offload_adapter(
        self,
    ) -> RequestOwnedBulkOffloadAdapter:
        """Take exclusive ownership of the registered offload substrate."""

        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            request_owned_offloading_connector,
        )

        RequestOwnedOffloadingConnector = (
            request_owned_offloading_connector.RequestOwnedOffloadingConnector
        )

        connector = get_kv_transfer_group()
        if not isinstance(connector, RequestOwnedOffloadingConnector):
            raise RuntimeError(
                "enable_request_owned_kv_offload requires the exclusive "
                "RequestOwnedOffloadingConnector worker, got "
                f"{type(connector).__name__}."
            )
        return connector.build_request_owned_adapter(self.global_rank)

    def _create_request_owned_kv_store(
        self, kv_cache_config: KVCacheConfig
    ) -> RequestOwnedKVStore:
        """Build the rank-local physical KV store (G2).

        Scheduler/hash block sizes, DCP/PCP world sizes, and max batched
        tokens come from the same vllm_config facts the scheduler uses, so
        the store's block pool accounting matches the coordinator's.
        The rank-local manager runs on the normalized config (uniform
        wrapper groups bound to their allocation-binding inner spec), never
        on the raw config the underlying worker initialized with.
        """
        kv_cache_config = _normalize_request_owned_kv_cache_config(kv_cache_config)
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, self.vllm_config
        )
        kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.vllm_config.model_config.max_model_len,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            max_num_batched_tokens=(
                self.vllm_config.scheduler_config.max_num_batched_tokens
            ),
            enable_caching=False,
            use_eagle=False,
            log_stats=False,
            enable_kv_cache_events=False,
            dcp_world_size=(
                self.vllm_config.parallel_config.decode_context_parallel_size
            ),
            pcp_world_size=(
                self.vllm_config.parallel_config.prefill_context_parallel_size
            ),
        )
        return RequestOwnedKVStore(kv_cache_manager, owner_rank=self.global_rank)

    def _request_owned_logical_capacity(self) -> int:
        """Nonphysical logical token budget for the reference lease manager.

        G2 keeps the logical manager only as the protocol fence/outbox
        engine; the rank-local physical store is the actual capacity
        authority.  This documented upper bound (max_model_len *
        max_num_seqs) is deliberately not a physical capacity claim: it is
        large enough that every command a physically-capable store could
        accept is also granted logically, so no capacity decision is made on
        logical grounds.
        """
        assert self.vllm_config is not None
        return (
            self.vllm_config.model_config.max_model_len
            * self.vllm_config.scheduler_config.max_num_seqs
        )

    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        if self.vllm_config.scheduler_config.enable_request_owned_attention:
            return self._execute_request_owned_control_step(scheduler_output)

        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    def _execute_request_owned_control_step(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Run one strict control step with whole-step RESTORE rollback."""

        if self._request_owned_restore_guard is not None:
            raise RuntimeError(
                "request-owned attention found an unclosed RESTORE guard "
                "before starting the next control step"
            )
        try:
            result = self._execute_request_owned_control_step_impl(scheduler_output)
        except BaseException:
            self._rollback_request_owned_restore_guard()
            raise
        if self._request_owned_restore_guard is not None:
            # A RESTORE may not escape into deferred sampling or any other
            # path that has not durably committed the control manager.
            self._rollback_request_owned_restore_guard()
            raise RuntimeError(
                "request-owned RESTORE control step returned before its "
                "terminal receipt committed"
            )
        return result

    def _execute_request_owned_control_step_impl(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute the G2 worker boundary over the rank-local physical store.

        Commands are composed failure-atomically: a deep-copied logical
        candidate is the preflight (fence/outbox), the physical store is
        consulted only when the candidate accepts, and a physical refusal
        advances the authoritative fences through the external-reject seam
        without any logical transition.

        G3 sampling (``enable_request_owned_sampling``, default off):
        the flag admits structurally valid token-bearing schedules through
        the same store/lease/metadata checks.  A step whose underlying
        ``execute_model`` returns ``None`` stores exactly one pending
        deferred record (trial manager + exact metadata) keyed by
        ``step_seq`` and returns ``None`` without marking, flushing,
        emitting, or committing anything; the explicit :meth:`sample_tokens`
        then runs the shared terminal path.  A synchronous
        :class:`ModelRunnerOutput` takes the same terminal path
        immediately.  The flag-off path preserves the control-only token
        and split-return rejections byte-for-byte.  A failure after the
        computed-batch mark is irreversible (the step is already marked)
        and latches a fail-stop state that rejects all further
        request-owned calls.
        """
        # G3 lifecycle: fail-closed latch for an irreversible post-mark
        # failure (the step is already marked and cannot be retried) must
        # reject before any state is touched.
        self._request_owned_fail_stop_guard()

        # G3 lifecycle: a prior deferred step must complete through
        # sample_tokens before any next execute call.  Fail closed before
        # touching any state (including the start-of-call None handoff) so
        # the pending record and its exact metadata stay intact.
        if self._request_owned_deferred is not None:
            raise RuntimeError(
                "request-owned attention has a pending deferred sampling "
                f"step (step_seq={self._request_owned_deferred.step_seq}): "
                "sample_tokens must complete before the next execute_model "
                "call."
            )

        sampling_enabled = self._request_owned_sampling_enabled()

        # G3 lifecycle: actively clear stale worker-private metadata at the
        # start of every request-owned call, before any validation.  The
        # ``None`` handoff clears the concrete worker's runner state too,
        # so a failure before the next successful build can never expose
        # the previous step's metadata.
        self._request_owned_step_metadata = None
        self._deliver_request_owned_step_metadata(None)

        step_seq = scheduler_output.step_seq
        if isinstance(step_seq, bool) or not isinstance(step_seq, int) or step_seq <= 0:
            raise RuntimeError(
                "request-owned control step requires a positive non-bool "
                f"step_seq, got {step_seq!r}."
            )

        total_tokens = scheduler_output.total_num_scheduled_tokens
        per_request_tokens = scheduler_output.num_scheduled_tokens
        if not sampling_enabled:
            # Flag-off control-only gate (byte-for-byte unchanged): every
            # token-bearing schedule is refused before the underlying worker
            # runs.
            if (
                isinstance(total_tokens, bool)
                or not isinstance(total_tokens, int)
                or total_tokens != 0
                or any(
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                    for count in per_request_tokens.values()
                )
                or sum(per_request_tokens.values()) != 0
            ):
                raise RuntimeError(
                    "request-owned attention G1 is control-only: refusing to "
                    "execute a nonempty or inconsistent token schedule through "
                    "replicated KV before the G2 owner-local allocator/routing "
                    "prerequisite exists."
                )
        else:
            # G3 sampling: admit structurally valid token-bearing schedules
            # (non-bool int types, nonnegativity, total == sum of per-request
            # counts, the scheduler invariant) through the same store/lease/
            # metadata checks; inconsistent envelopes still fail before the
            # underlying worker runs.
            if (
                isinstance(total_tokens, bool)
                or not isinstance(total_tokens, int)
                or total_tokens < 0
                or any(
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                    for count in per_request_tokens.values()
                )
                or sum(per_request_tokens.values()) != total_tokens
            ):
                raise RuntimeError(
                    "request-owned attention sampling admits only a "
                    "consistent non-bool token schedule, got "
                    f"total={total_tokens!r} per_request="
                    f"{dict(per_request_tokens)!r}."
                )

        # Apply the step to a trial copy.  Manager fences/outbox become durable
        # only after a concrete synchronous output is available, so an
        # underlying exception/None/async result cannot poison the next step.
        store = self._request_owned_kv_store
        if store is None:
            raise RuntimeError(
                "request-owned attention worker store is not initialized: "
                "initialize_from_config must construct the rank-local "
                "RequestOwnedKVStore before execution."
            )

        manager = self._request_owned_control_manager
        if manager is None:
            manager = AttentionLeaseManager(
                owner_rank=self.global_rank,
                capacity=self._request_owned_logical_capacity(),
            )
        trial_manager = deepcopy(manager)
        restore_work: list[RequestOwnedBulkRestoreWork] = []
        restore_commands: list[OwnerCommand] = []
        restore_build_checkpoint: RequestOwnedStepBuildCheckpoint | None = None

        restore_control_step = any(
            command.kind is OwnerCommandKind.RESTORE
            for command in scheduler_output.owner_commands
        )
        if restore_control_step and (
            any(
                command.kind is not OwnerCommandKind.RESTORE
                for command in scheduler_output.owner_commands
            )
            or total_tokens != 0
            or any(count != 0 for count in per_request_tokens.values())
            or scheduler_output.scheduled_owner_leases
        ):
            raise RuntimeError(
                "request-owned bulk RESTORE requires an exclusive zero-token "
                "global owner control step"
            )

        # Commands form one reliable in-order stream per owner.  Every worker
        # receives the global envelope but consumes only its own commands.
        # Per own-rank command, failure-atomic composition: the candidate is
        # a deep copy of the current trial logical manager; its apply() is the
        # logical preflight.  A logical refusal adopts the candidate (its
        # fences/outbox are durable) and never touches the physical store.  A
        # logical accept invokes the corresponding physical operation; a
        # physical accept adopts the candidate, while a physical refusal
        # discards the candidate and advances the authoritative fences via
        # external_reject_error, without any logical transition.
        for command in scheduler_output.owner_commands:
            if command.owner_id != self.global_rank:
                continue
            candidate = deepcopy(trial_manager)
            preflight = candidate.apply(command)
            if not preflight.accepted:
                trial_manager = candidate
                continue
            physical, pending_restore = self._apply_request_owned_physical(
                command, store, step_seq
            )
            if physical.accepted:
                if pending_restore is not None:
                    restore_work.append(pending_restore)
                    restore_commands.append(command)
                    if restore_build_checkpoint is None:
                        restore_build_checkpoint = store.checkpoint_step_build()
                    # Arm immediately: a later command in the same owner
                    # batch can still fail before the H2D hook runs.
                    self._request_owned_restore_guard = (
                        tuple(restore_work),
                        store,
                        restore_build_checkpoint,
                        step_seq,
                    )
                trial_manager = candidate
                continue
            reject_error = physical.error or (
                "physical request-owned KV store rejected the command without an error"
            )
            trial_manager.apply(command, external_reject_error=reject_error)

        if restore_work:
            self._execute_request_owned_bulk_restore(tuple(restore_work), store)
            # H2D and destination readiness are now real, but the logical
            # RESTORE receipt is not durable until this whole control step
            # commits.  Arm the rollback guard *before* completing the trial
            # manager so metadata delivery, model execution, output checks,
            # flush, receipt emission, and commit are all covered.
            for command in restore_commands:
                trial_manager.complete_restore(command.key, command.command_seq)

        for token in scheduler_output.scheduled_owner_leases:
            if token.owner_id == self.global_rank:
                trial_manager.record_published(token)

        # G3 seam: after command+publication validation, freeze the
        # immutable worker-local execution metadata for this step.  No
        # computed progress is marked here; completion is declared by the
        # shared terminal path (mark -> flush -> emit -> commit) once the
        # executing GPU step finished, synchronously or through
        # sample_tokens.
        metadata = self._build_request_owned_step_metadata(
            store, step_seq, scheduler_output
        )
        self._request_owned_step_metadata = metadata
        # G3 handoff: deliver the immutable metadata to the concrete worker
        # through its private hook, without attaching it to or mutating any
        # scheduler wire object; unsupported workers fail closed.
        self._deliver_request_owned_step_metadata(metadata)

        self._apply_mm_cache(scheduler_output)
        output = self.worker.execute_model(scheduler_output)

        from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

        if sampling_enabled and output is None:
            if restore_control_step:
                # Ascend deliberately represents an owner-sampling zero-token
                # heartbeat as execute_model() -> None followed by an empty
                # sample_tokens() envelope. RESTORE must still commit in this
                # synchronous collective RPC so the whole-step rollback guard
                # never escapes across the executor's deferred boundary. Ask
                # the concrete worker for that real heartbeat rather than
                # fabricating an empty output (the sampling aggregator requires
                # one owner-qualified empty batch from every rank).
                output = self.worker.sample_tokens(None)
            else:
                # Deferred sampling: keep the trial manager and the exact
                # metadata in one pending record keyed by step_seq.  Nothing is
                # marked, flushed, emitted, or committed until sample_tokens
                # completes the step.
                self._request_owned_deferred = _RequestOwnedDeferredStep(
                    step_seq=step_seq,
                    trial_manager=trial_manager,
                    metadata=metadata,
                )
                return None

        if output is None:
            if sampling_enabled and restore_control_step:
                raise RuntimeError(
                    "request-owned RESTORE heartbeat sample_tokens returned None"
                )
            raise RuntimeError(
                "request-owned attention G1 does not support split sampling: "
                "execute_model returned None and no exact receipt FIFO exists."
            )

        # Async outputs can overlap subsequent steps.  Until receipt state is
        # kept in a step-keyed FIFO, decorating them here could drain events
        # into the wrong step, so fail explicitly rather than guessing.
        if isinstance(output, AsyncModelRunnerOutput):
            raise RuntimeError(
                "request-owned attention G1 does not support async model "
                "runner outputs without a step-keyed receipt FIFO."
            )
        if not isinstance(output, ModelRunnerOutput):
            raise RuntimeError(
                "request-owned attention worker returned an unexpected "
                f"output type {type(output).__name__}."
            )

        if sampling_enabled:
            # Synchronous token output: same terminal path as the deferred
            # sample_tokens completion (mark -> flush -> emit -> commit).
            result = self._complete_request_owned_step(
                step_seq, trial_manager, metadata, output
            )
            self._request_owned_restore_guard = None
            return result

        # Never mutate a worker-owned output or EMPTY_MODEL_RUNNER_OUTPUT.
        result = copy(output)
        # Post-execute completion fence: only now that the executing GPU step
        # finished are deferred physical PREEMPT/RELEASE frees returned to the
        # shared pool, so the receipt certifies physical free.  The attached
        # capacity snapshot is block-ID-free and is taken after the flush.
        store.flush()
        batch = trial_manager.emit_batch(step_seq)
        result.owner_receipt_batches = [
            replace(batch, cache_pool=store.pool_snapshot())
        ]
        self._request_owned_control_manager = trial_manager
        self._request_owned_restore_guard = None
        return result

    def _rollback_request_owned_restore_guard(self) -> None:
        """Discard every uncommitted restored destination, retaining host KV."""

        guard = self._request_owned_restore_guard
        self._request_owned_restore_guard = None
        if guard is None:
            return
        work, store, build_checkpoint, step_seq = guard
        errors: list[str] = []
        try:
            store.rollback_empty_step_build(build_checkpoint, step_seq)
        except BaseException as exc:
            errors.append(f"step-build rollback for {step_seq}: {exc!r}")
        for item in work:
            identity = item.plan.identity
            try:
                item.adapter.abort(identity)
            except BaseException as exc:
                errors.append(f"adapter abort for {identity.key!r}: {exc!r}")
            try:
                if not store.abort_restore(
                    identity.key, identity.allocation_generation
                ):
                    errors.append(
                        "physical destination did not match "
                        f"{identity.key!r} generation "
                        f"{identity.allocation_generation}"
                    )
            except BaseException as exc:
                errors.append(f"store abort for {identity.key!r}: {exc!r}")
        if errors:
            raise RuntimeError(
                "request-owned RESTORE rollback failed: " + "; ".join(errors)
            )

    def _request_owned_sampling_enabled(self) -> bool:
        """G3 sampling gate, default off.

        The config field lands separately, so the wrapper reads it safely
        with ``getattr(..., False)``: until then (and on every default
        config) the flag-off control-only path is preserved unchanged.
        The gate is strict: only a real ``bool`` admits the deferred
        sampling protocol, so accidental truthy values (``1``,
        ``"true"``) fail closed instead of silently enabling it.
        """
        value = getattr(
            self.vllm_config.scheduler_config,
            "enable_request_owned_sampling",
            False,
        )
        if not isinstance(value, bool):
            raise RuntimeError(
                "enable_request_owned_sampling must be a bool, got "
                f"{type(value).__name__} ({value!r})."
            )
        return value

    def _request_owned_fail_stop_guard(self) -> None:
        """Reject every request-owned call once the fail-stop latch is set.

        The latch is set only after a successful computed-batch mark was
        followed by a terminal completion failure: the step is already
        marked in the store and can never be retried (a retry would hit a
        duplicate mark), so further calls must fail closed instead of
        risking duplicate or out-of-order marks."""
        if self._request_owned_fail_stop is not None:
            raise RuntimeError(
                "request-owned attention is in an irreversible fail-stop "
                f"state: {self._request_owned_fail_stop}"
            )

    def _apply_request_owned_physical(
        self,
        command: OwnerCommand,
        store: RequestOwnedKVStore,
        step_seq: int,
    ) -> tuple[
        AllocationResult | DeferredFreeResult,
        RequestOwnedBulkRestoreWork | None,
    ]:
        """Dispatch one own-rank command to the corresponding physical
        store operation.  The store rejects any kind/state mismatch itself,
        so this seam never duplicates the logical state machine."""
        if command.kind is OwnerCommandKind.RESERVE:
            restored = bool(
                self._request_owned_offload_adapter is not None
                and store.is_restore_ready(command.key)
            )
            result = store.reserve(command)
            if result.accepted and restored:
                adapter = self._require_request_owned_offload_adapter()
                snapshot = store.snapshot(command.key)
                if snapshot is None:
                    raise RuntimeError("restored RESERVE lost its physical record")
                identity = OwnerOffloadIdentity.from_snapshot(snapshot)
                adapter.activate(identity)
                if not store.mark_reactivated(
                    command.key, snapshot.allocation_generation
                ):
                    raise RuntimeError(
                        "restored RESERVE could not close its physical HOT state"
                    )
            return result, None
        if command.kind is OwnerCommandKind.EXTEND:
            return store.extend(command), None
        if command.kind is OwnerCommandKind.PREEMPT:
            if self._request_owned_offload_adapter is None:
                return store.preempt(command), None
            return self._store_request_owned_preempt(command, store), None
        if command.kind is OwnerCommandKind.RELEASE:
            snapshot = (
                store.snapshot(command.key)
                if self._request_owned_offload_adapter is not None
                else None
            )
            result = store.release(command)
            if (
                result.accepted
                and snapshot is not None
                and self._request_owned_offload_adapter is not None
            ):
                self._request_owned_offload_adapter.release(snapshot)
            if result.accepted and self._request_owned_offload_adapter is not None:
                self._request_owned_offload_adapter.evict_owned_host_keys(command.key)
            return result, None
        if command.kind is OwnerCommandKind.RESTORE:
            result = store.restore(command)
            if not result.accepted:
                return result, None
            snapshot = store.snapshot(command.key)
            computed = store.computed_prefix_snapshot(command.key)
            if snapshot is None or computed is None:
                raise RuntimeError("accepted RESTORE did not create a destination")
            adapter = self._require_request_owned_offload_adapter()
            identity = OwnerOffloadIdentity.from_snapshot(computed)
            bound = False
            try:
                adapter.bind(computed, active=False)
                bound = True
                keys = make_request_owned_offload_keys(
                    computed, store.group_block_sizes
                )
                source_block_indices = store.restore_source_block_indices(
                    command.key, computed.allocation_generation
                )
                if source_block_indices is None:
                    raise RuntimeError(
                        "RESTORE destination lost its durable source block mask"
                    )
                plan = OwnerOffloadPlan.from_snapshot(
                    computed,
                    keys,
                    logical_block_indices=source_block_indices,
                )
                work = RequestOwnedBulkRestoreWork(
                    step_seq=step_seq,
                    adapter=adapter,
                    plan=plan,
                    zero_block_ids=snapshot.tables,
                )
            except BaseException:
                if bound:
                    adapter.abort(identity)
                store.abort_restore(command.key, snapshot.allocation_generation)
                raise
            return result, work
        raise RuntimeError(f"unknown owner command kind {command.kind}")

    def _require_request_owned_offload_adapter(
        self,
    ) -> RequestOwnedBulkOffloadAdapter:
        adapter = self._request_owned_offload_adapter
        if adapter is None:
            raise RuntimeError(
                "request-owned KV offload command requires the exclusive adapter"
            )
        return adapter

    def _store_request_owned_preempt(
        self, command: OwnerCommand, store: RequestOwnedKVStore
    ) -> DeferredFreeResult:
        adapter = self._require_request_owned_offload_adapter()
        snapshot = store.computed_prefix_snapshot(command.key)
        if snapshot is None:
            return DeferredFreeResult(
                accepted=False,
                key=command.key,
                error="PREEMPT has no owner-local computed-prefix source",
            )
        identity = adapter.bind(snapshot, active=True)
        adapter.retire(identity)
        plan = OwnerOffloadPlan.from_snapshot(
            snapshot,
            make_request_owned_offload_keys(snapshot, store.group_block_sizes),
        )
        job = adapter.submit_store(plan)
        adapter.wait((job,))
        receipts = adapter.poll()
        matching = tuple(item for item in receipts if item.job_id == job.job_id)
        if len(matching) != 1 or len(receipts) != 1:
            adapter.abort(identity)
            raise RequestOwnedOffloadError(
                f"PREEMPT store produced {len(matching)} exact and "
                f"{len(receipts)} total completion receipts"
            )
        receipt = matching[0]
        if not receipt.success:
            # The GPU source is still intact. Reopen the retired ledger state
            # so a later PREEMPT retry can submit the same generation again.
            adapter.activate(identity)
            return DeferredFreeResult(
                accepted=False,
                key=command.key,
                error=receipt.error,
            )
        reclaimable = adapter.take_reclaimable(identity)
        if reclaimable != plan.device_block_ids:
            raise RequestOwnedOffloadError(
                "PREEMPT durable receipt named the wrong physical source"
            )
        return store.preempt(command)

    def _execute_request_owned_bulk_restore(
        self,
        work: tuple[RequestOwnedBulkRestoreWork, ...],
        store: RequestOwnedKVStore,
    ) -> None:
        hook = getattr(self.worker, "execute_request_owned_bulk_restore", None)
        if not callable(hook):
            raise RuntimeError(
                "request-owned bulk RESTORE requires the worker's post-zero "
                "pre-forward restore hook"
            )
        hook(work)
        for item in work:
            identity = item.plan.identity
            if not store.mark_restore_ready(
                identity.key, identity.allocation_generation
            ):
                raise RuntimeError(
                    "bulk RESTORE completion did not match its destination generation"
                )

    def _build_request_owned_step_metadata(
        self,
        store: RequestOwnedKVStore,
        step_seq: int,
        scheduler_output: SchedulerOutput,
    ) -> RequestOwnedStepMetadata:
        """G3 wrapper seam: build the one-step immutable worker-local
        execution metadata after command+publication validation.  Execution
        tokens/counts are derived only from positive
        ``scheduler_output.num_scheduled_tokens`` for own-rank grants: a
        zero-token G2 heartbeat may carry newly published grants but builds
        empty execution metadata and retains allocation deltas for the later
        token-bearing step, while a token-bearing step must match own-rank
        authorization tokens and own-rank positive counts exactly.  The
        builder itself rejects wrong/missing/extra/duplicate/stale/
        pending-free/out-of-horizon lease and count state and is one-step
        fenced; a rejection here is fail-stop because the store retains
        every pending delta and the step is retryable.  No scheduler wire
        object is mutated."""
        own_rank_tokens = [
            token
            for token in scheduler_output.scheduled_owner_leases
            if token.owner_id == self.global_rank
        ]
        build = store.build_step_metadata(
            step_seq,
            own_rank_tokens,
            scheduler_output.num_scheduled_tokens,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        if not build.accepted:
            raise RuntimeError(
                "request-owned step metadata build failed: "
                f"{build.error or 'unknown error'}"
            )
        if build.metadata is None:
            raise RuntimeError(
                "request-owned step metadata build accepted without metadata"
            )
        return build.metadata

    def _deliver_request_owned_step_metadata(
        self, metadata: RequestOwnedStepMetadata | None
    ) -> None:
        """Hand the G3 step metadata to the concrete worker.

        ``None`` clears the worker's stale runner state at the start of a
        request-owned call; the immutable batch is delivered after a
        successful build.  The hook is a worker-private method call: the
        metadata is never attached to a SchedulerOutput or any other wire
        object, and no wire object is mutated.  Workers that do not
        implement the hook fail closed with an explicit error rather than
        silently dropping the handoff."""
        handoff = getattr(self.worker, "set_request_owned_step_metadata", None)
        if not callable(handoff):
            raise RuntimeError(
                f"worker {type(self.worker).__name__} does not support the "
                "request-owned G3 step metadata handoff; unsupported "
                "workers fail closed."
            )
        handoff(metadata)

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Explicit wrapper sampling seam.

        Default (request-owned attention disabled) delegates to the
        underlying worker exactly like the historical ``__getattr__``
        delegation.  With request-owned attention enabled, split sampling
        is only supported under ``enable_request_owned_sampling`` with a
        pending deferred step: the completion runs the same terminal path
        as a synchronous token output.  Calls after an irreversible
        post-mark failure also fail closed.  Any other call fails
        closed."""
        if not self.vllm_config.scheduler_config.enable_request_owned_attention:
            return self.worker.sample_tokens(grammar_output)
        # Irreversible post-mark failure latch: reject before any state.
        self._request_owned_fail_stop_guard()
        if not self._request_owned_sampling_enabled():
            raise RuntimeError(
                "request-owned attention does not support split sampling "
                "without enable_request_owned_sampling: execute_model never "
                "defers in this mode, so sample_tokens must not be called."
            )
        return self._sample_request_owned(grammar_output)

    def _sample_request_owned(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Complete one pending deferred request-owned sampling step.

        Requires exactly one pending record (replay after success and
        out-of-order calls fail closed), calls the underlying worker's
        ``sample_tokens``, rejects ``None``/async/unexpected outputs, then
        runs the shared terminal path (mark exactly once, flush, emit the
        exact receipt batch with the post-flush pool snapshot, commit the
        trial manager) on a copy of the worker output.  Worker-emitted
        ``owner_sampling_batches`` ride the copied output untouched.  Any
        failure leaves the pending record intact and the logical manager
        uncommitted; only full success clears the pending record.  A
        post-mark failure latches the wrapper fail-stop state: the step
        is already marked in the store and cannot be retried, and the
        pending record is never cleared."""
        pending = self._request_owned_deferred
        if pending is None:
            raise RuntimeError(
                "request-owned sample_tokens requires a pending deferred "
                "step: execute_model returned None but no deferred record "
                "exists (replay or out-of-order call)."
            )

        output = self.worker.sample_tokens(grammar_output)

        from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

        if output is None:
            raise RuntimeError(
                "request-owned deferred sampling failed: "
                "worker.sample_tokens returned None."
            )
        if isinstance(output, AsyncModelRunnerOutput):
            raise RuntimeError(
                "request-owned attention does not support async model "
                "runner outputs from sample_tokens without a step-keyed "
                "receipt FIFO."
            )
        if not isinstance(output, ModelRunnerOutput):
            raise RuntimeError(
                "request-owned attention worker returned an unexpected "
                f"output type {type(output).__name__} from sample_tokens."
            )

        result = self._complete_request_owned_step(
            pending.step_seq, pending.trial_manager, pending.metadata, output
        )
        # Clear the pending record only after the terminal path fully
        # succeeded (manager committed); a failure above leaves it intact.
        self._request_owned_deferred = None
        return result

    def _complete_request_owned_step(
        self,
        step_seq: int,
        trial_manager: AttentionLeaseManager,
        metadata: RequestOwnedStepMetadata,
        output: ModelRunnerOutput,
    ) -> ModelRunnerOutput:
        """Shared terminal decoration for request-owned sampling.

        Never mutates a worker-owned output or the shared empty singleton:
        the output is copied first.  Then the exact step metadata is marked
        exactly once (all-or-nothing in the store; marking the empty
        metadata of a zero-token heartbeat step is a valid accepted no-op),
        deferred physical frees are flushed, and the exact receipt batch
        with the post-flush pool snapshot is attached before the trial
        logical manager is committed.  A rejection of the mark fails
        atomically: the logical manager stays uncommitted and (for the
        deferred path) the pending record stays intact.  A failure after a
        real computed mark is irreversible and latches fail-stop.  The sole
        exception is an exclusive zero-token RESTORE heartbeat: its empty
        mark/build fences and physical destination are jointly reversible
        under the armed whole-step restore guard."""
        store = self._request_owned_kv_store
        if store is None:
            raise RuntimeError(
                "request-owned attention worker store is not initialized: "
                "the request-owned sampling completion requires the "
                "rank-local RequestOwnedKVStore."
            )

        # Never mutate a worker-owned output or EMPTY_MODEL_RUNNER_OUTPUT.
        result = copy(output)
        # Atomic computed-batch mark: validates every entry and full
        # coverage of the step's expectations before any record advances; a
        # rejection fails closed with no partial logical commit.  Marking
        # empty execution metadata (a zero-token heartbeat step) is a valid
        # no-op accepted by the store.
        committed_num_tokens = self._request_owned_committed_num_tokens(
            metadata, result
        )
        mark = store.mark_computed_batch(metadata, committed_num_tokens)
        if not mark.accepted:
            raise RuntimeError(
                "request-owned computed batch mark failed: "
                f"{mark.error or 'unknown error'}"
            )
        try:
            # Post-execute completion fence: only now that the executing GPU
            # step finished are deferred physical PREEMPT/RELEASE frees
            # returned to the shared pool, so the receipt certifies physical
            # free.  The attached capacity snapshot is block-ID-free and is
            # taken after the flush.
            store.flush()
            batch = trial_manager.emit_batch(step_seq)
            result.owner_receipt_batches = [
                replace(batch, cache_pool=store.pool_snapshot())
            ]
        except BaseException as exc:
            # A real computed batch is irreversible and must latch fail-stop.
            # An armed RESTORE guard necessarily covers an exclusive empty
            # heartbeat; its scalar empty-mark/build fences are rolled back
            # with the physical destination by the outer wrapper.
            if self._request_owned_restore_guard is None:
                self._request_owned_fail_stop = (
                    "computed batch mark succeeded but the irreversible "
                    f"terminal completion failed ({exc!r}); the step is "
                    "already marked and cannot be retried."
                )
            raise
        self._request_owned_control_manager = trial_manager
        return result

    @staticmethod
    def _request_owned_committed_num_tokens(
        metadata: RequestOwnedStepMetadata,
        output: ModelRunnerOutput,
    ) -> dict[OwnerLeaseKey, int]:
        """Derive verified logical KV commits from a terminal output.

        Execution writes the complete speculative target horizon, while
        sampling returns only the accepted prefix plus one terminal token.
        The returned mapping advances spec entries by exactly that emitted
        count.  Non-spec entries retain the store's exact-post contract and
        therefore need no override.
        """
        speculative_entries = tuple(
            entry for entry in metadata.entries if entry.num_speculative_tokens > 0
        )
        if not speculative_entries:
            return {}

        req_ids = output.req_ids
        sampled_token_ids = output.sampled_token_ids
        req_id_to_index = output.req_id_to_index
        if len(sampled_token_ids) != len(req_ids):
            raise RuntimeError(
                "request-owned speculative completion requires sampled-token "
                "rows to align 1:1 with req_ids."
            )
        if not isinstance(req_id_to_index, dict) or len(req_id_to_index) != len(
            req_ids
        ):
            raise RuntimeError(
                "request-owned speculative completion requires req_id_to_index "
                "to be bijective over req_ids."
            )
        seen_req_ids: set[str] = set()
        for expected_index, req_id in enumerate(req_ids):
            if not isinstance(req_id, str) or req_id in seen_req_ids:
                raise RuntimeError(
                    "request-owned speculative completion requires unique "
                    "string req_ids."
                )
            seen_req_ids.add(req_id)
            mapped_index = req_id_to_index.get(req_id)
            if (
                isinstance(mapped_index, bool)
                or not isinstance(mapped_index, int)
                or mapped_index != expected_index
            ):
                raise RuntimeError(
                    "request-owned speculative completion requires req_id_to_index "
                    "to be bijective and aligned with req_ids."
                )

        commits: dict[OwnerLeaseKey, int] = {}
        for entry in speculative_entries:
            req_id = entry.key.request_id
            index = req_id_to_index.get(req_id)
            if index is None:
                raise RuntimeError(
                    "request-owned speculative completion is missing the "
                    f"terminal output for {req_id!r}."
                )
            tokens = sampled_token_ids[index]
            if not isinstance(tokens, list):
                raise RuntimeError(
                    "request-owned speculative completion requires a list "
                    f"of sampled tokens for {req_id!r}, got "
                    f"{type(tokens).__name__}."
                )
            if not 1 <= len(tokens) <= entry.num_speculative_tokens + 1:
                raise RuntimeError(
                    "request-owned speculative completion emitted an "
                    f"invalid token count for {req_id!r}: got {len(tokens)}, "
                    f"expected 1..{entry.num_speculative_tokens + 1}."
                )
            commits[entry.key] = entry.pre_step_num_computed_tokens + len(tokens)
        return commits

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
