# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import replace
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
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.request_owned_kv import (
    AllocationResult,
    DeferredFreeResult,
    RequestOwnedKVStore,
    RequestOwnedStepMetadata,
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
        #: Immutable worker-local G3 execution metadata of the last step
        #: whose command+publication validation succeeded.  Cleared at the
        #: start of every request-owned call (the concrete worker is
        #: actively cleared with a ``None`` handoff) and handed to the
        #: worker through its private hook; never attached to a scheduler
        #: wire.
        self._request_owned_step_metadata: RequestOwnedStepMetadata | None = None

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

    def _create_request_owned_kv_store(
        self, kv_cache_config: KVCacheConfig
    ) -> RequestOwnedKVStore:
        """Build the rank-local physical KV store (G2).

        Scheduler/hash block sizes, DCP/PCP world sizes, and max batched
        tokens come from the same vllm_config facts the scheduler uses, so
        the store's block pool accounting matches the coordinator's.
        """
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
    ) -> ModelRunnerOutput:
        """Execute the G2 worker boundary over the rank-local physical store.

        Commands are composed failure-atomically: a deep-copied logical
        candidate is the preflight (fence/outbox), the physical store is
        consulted only when the candidate accepts, and a physical refusal
        advances the authoritative fences through the external-reject seam
        without any logical transition.  The positive step and zero-token
        terminal validation below are unchanged: G2 is still control-only
        at this milestone, so every token-bearing schedule is refused before
        the underlying worker runs.
        """
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
            physical = self._apply_request_owned_physical(command, store)
            if physical.accepted:
                trial_manager = candidate
                continue
            reject_error = physical.error or (
                "physical request-owned KV store rejected the command without an error"
            )
            trial_manager.apply(command, external_reject_error=reject_error)

        for token in scheduler_output.scheduled_owner_leases:
            if token.owner_id == self.global_rank:
                trial_manager.record_published(token)

        # G3 seam: after command+publication validation, freeze the
        # immutable worker-local execution metadata for this step.  The
        # zero-token terminal gate above still refuses every token-bearing
        # schedule before the model runner, so the handed metadata is
        # always the empty heartbeat batch at this milestone; no computed
        # progress is marked here.
        self._request_owned_step_metadata = self._build_request_owned_step_metadata(
            store, step_seq, scheduler_output
        )
        # G3 handoff: deliver the immutable metadata to the concrete worker
        # through its private hook, without attaching it to or mutating any
        # scheduler wire object; unsupported workers fail closed.
        self._deliver_request_owned_step_metadata(self._request_owned_step_metadata)

        self._apply_mm_cache(scheduler_output)
        output = self.worker.execute_model(scheduler_output)
        if output is None:
            raise RuntimeError(
                "request-owned attention G1 does not support split sampling: "
                "execute_model returned None and no exact receipt FIFO exists."
            )

        # Async outputs can overlap subsequent steps.  Until receipt state is
        # kept in a step-keyed FIFO, decorating them here could drain events
        # into the wrong step, so fail explicitly rather than guessing.
        from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

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
        return result

    @staticmethod
    def _apply_request_owned_physical(
        command: OwnerCommand, store: RequestOwnedKVStore
    ) -> AllocationResult | DeferredFreeResult:
        """Dispatch one own-rank command to the corresponding physical
        store operation.  The store rejects any kind/state mismatch itself,
        so this seam never duplicates the logical state machine."""
        if command.kind is OwnerCommandKind.RESERVE:
            return store.reserve(command)
        if command.kind is OwnerCommandKind.EXTEND:
            return store.extend(command)
        if command.kind is OwnerCommandKind.PREEMPT:
            return store.preempt(command)
        if command.kind is OwnerCommandKind.RELEASE:
            return store.release(command)
        if command.kind is OwnerCommandKind.RESTORE:
            return store.restore(command)
        raise RuntimeError(f"unknown owner command kind {command.kind}")

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
            step_seq, own_rank_tokens, scheduler_output.num_scheduled_tokens
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

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
