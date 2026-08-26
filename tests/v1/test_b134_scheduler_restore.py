# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-path tests for the B134 event chain (scheduler restore branch).

The scheduler restore branch lives deep inside ``Scheduler.schedule()``
(a 700-line method). We execute the REAL method body with a minimal mock
object graph: a PREEMPTED request is placed in the waiting queue, all
scheduler dependencies are replaced by controllable stand-ins, and the
module-level ``emit`` is replaced with a recorder. The test then asserts
the actual event sequence emitted for one restored request.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).parents[2]


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)

    def _fallback(attr):
        # Any symbol not explicitly listed is a fresh MagicMock, so
        # multi-line `from X import A, B, C` always resolves.
        return MagicMock()

    mod.__getattr__ = _fallback  # type: ignore[method-assign]
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _load_scheduler_module():
    """Load the REAL scheduler.py with stubbed heavy deps, restore after."""
    stubs: dict[str, types.ModuleType] = {}

    def add(name: str, **attrs) -> types.ModuleType:
        # Build the stub but do NOT write sys.modules yet: the snapshot of
        # pre-existing state must be taken before ANY key is overwritten.
        mod = _make_module(name, **attrs)
        stubs[name] = mod
        return mod

    # Every symbol scheduler.py imports must exist on its stub module.
    add("torch", Tensor=MagicMock)
    add("numpy", array=MagicMock(side_effect=lambda *a, **k: list(a[0])))
    add("vllm")
    add("vllm.compilation")
    add("vllm.compilation.cuda_graph", CUDAGraphStat=MagicMock)
    add("vllm.config", VllmConfig=MagicMock)
    add("vllm.distributed")
    add("vllm.distributed.ec_transfer")
    add("vllm.distributed.ec_transfer.ec_connector")
    add(
        "vllm.distributed.ec_transfer.ec_connector.base",
        ECTransferConnector=MagicMock,
        ECTransferMetadata=MagicMock,
    )
    add(
        "vllm.distributed.ec_transfer.ec_connector.factory",
        ECConnectorFactory=MagicMock,
    )
    add(
        "vllm.distributed.kv_events",
        KVEventType=MagicMock,
        KVEvent=MagicMock,
        kVCacheEventPublisher=MagicMock,
    )
    add("vllm.distributed.kv_transfer")
    add("vllm.distributed.kv_transfer.kv_connector")
    add(
        "vllm.distributed.kv_transfer.kv_connector.factory",
        KVConnectorFactory=MagicMock,
    )
    add(
        "vllm.distributed.kv_transfer.kv_connector.v1",
        KVConnectorBase_V1=MagicMock,
        KVConnectorExtension=MagicMock,
    )
    add(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorMetadata=MagicMock,
    )
    add(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        KVConnectorStats=MagicMock,
    )
    add("vllm.logger", init_logger=MagicMock())
    add("vllm.model_executor")
    add("vllm.model_executor.layers")
    add("vllm.model_executor.layers.fused_moe")
    add(
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer",
        CapturedRoutedExperts=MagicMock,
    )
    add(
        "vllm.multimodal", MULTIMODAL_REGISTRY=MagicMock(), MultiModalRegistry=MagicMock
    )
    add("vllm.multimodal.encoder_budget", MultiModalBudget=MagicMock)
    add("vllm.multimodal.utils", get_mm_features_in_window=MagicMock())
    add("vllm.v1")
    add("vllm.v1.core")
    add("vllm.v1.core.sched")
    add(
        "vllm.v1.core.sched.interface",
        PauseState=MagicMock,
        SchedulerInterface=MagicMock,
    )
    # PauseState is a real Enum (UNPAUSED / PAUSED_ALL); model faithfully.
    pause_state = types.new_class(
        "PauseState",
        (types.SimpleNamespace,),
        exec_body=lambda ns: ns.update(
            UNPAUSED="UNPAUSED",
            PAUSED_ALL="PAUSED_ALL",
        ),
    )
    stubs["vllm.v1.core.sched.interface"].PauseState = pause_state  # type: ignore[attr-defined]
    add(
        "vllm.v1.core.sched.output",
        SchedulerOutput=MagicMock,
        CachedRequestData=MagicMock,
        NewRequestData=MagicMock,
        PreemptedRequestData=MagicMock,
        ScheduledRequestData=MagicMock,
    )
    add("vllm.v1.core.sched.policy")
    add(
        "vllm.v1.core.sched.request_queue",
        RequestQueue=MagicMock,
        create_request_queue=MagicMock(side_effect=lambda policy: MagicMock()),
    )
    add("vllm.v1.core.sched.utils", check_stop=MagicMock, remove_all=MagicMock)
    add("vllm.v1.core.sched.victim_selector", VictimSelector=MagicMock)
    add(
        "vllm.v1.core.encoder_cache_manager",
        EncoderCacheManager=MagicMock,
        EncoderCacheManagerV1=MagicMock,
    )
    add("vllm.v1.core.kv_cache_coordinator", HybridKVCacheCoordinator=MagicMock)
    add(
        "vllm.v1.core.kv_cache_manager",
        KVCacheBlocks=MagicMock,
        KVCacheManager=MagicMock,
    )
    add("vllm.v1.core.kv_cache_metrics", KVCacheMetricsCollector=MagicMock)
    add("vllm.v1.core.kv_cache_utils", KVCacheBlock=MagicMock)
    add(
        "vllm.v1.engine",
        EngineCoreEventType=MagicMock(
            SCHEDULED="SCHEDULED", PREEMPTED="PREEMPTED"
        ),
        EngineCoreOutput=MagicMock,
        EngineCoreOutputs=MagicMock,
    )
    add("vllm.v1.kv_cache_interface", KVCacheConfig=MagicMock)
    add(
        "vllm.v1.kv_cache_compression",
        KVCacheCompressionError=MagicMock,
        KVCacheCompressionRuntimeSpec=MagicMock,
    )
    add("vllm.v1.metrics")
    add("vllm.v1.metrics.perf", ModelMetrics=MagicMock, PerfStats=MagicMock)
    add("vllm.v1.metrics.stats", PrefixCacheStats=MagicMock, SchedulerStats=MagicMock)
    add(
        "vllm.v1.outputs",
        DraftTokenIds=MagicMock,
        KVConnectorOutput=MagicMock,
        ModelRunnerOutput=MagicMock,
    )
    # RequestStatus is a plain enum; model it faithfully so status checks
    # in schedule() behave (PREEMPTED == PREEMPTED etc).
    request_status = types.new_class(
        "RequestStatus",
        (types.SimpleNamespace,),
        exec_body=lambda ns: ns.update(
            WAITING="WAITING",
            RUNNING="RUNNING",
            PREEMPTED="PREEMPTED",
            WAITING_FOR_REMOTE_KVS="WAITING_FOR_REMOTE_KVS",
            FINISHED_STOPPED="FINISHED_STOPPED",
            WAITING_FOR_STREAMING_REQ="WAITING_FOR_STREAMING_REQ",
        ),
    )
    add(
        "vllm.v1.request",
        Request=MagicMock,
        RequestStatus=request_status,
        StreamingUpdate=MagicMock,
    )
    add("vllm.v1.spec_decode")
    add("vllm.v1.spec_decode.dynamic")
    add("vllm.v1.spec_decode.dynamic.utils", build_dynamic_sd_schedule_lookup=MagicMock)
    add("vllm.v1.spec_decode.metrics", SpecDecodingStats=MagicMock)
    add("vllm.v1.structured_output", StructuredOutputManager=MagicMock)
    add("vllm.v1.utils", record_function_or_nullcontext=MagicMock())

    # Real events module (pure stdlib): scheduler imports EventBus from it.
    events_spec = importlib.util.spec_from_file_location(
        "vllm.v1.events",
        REPO_ROOT / "vllm" / "v1" / "events.py",
    )
    assert events_spec is not None and events_spec.loader is not None
    events_mod = importlib.util.module_from_spec(events_spec)

    scheduler_path = REPO_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
    scheduler_spec = importlib.util.spec_from_file_location(
        "vllm.v1.core.sched.scheduler", scheduler_path
    )
    assert scheduler_spec is not None and scheduler_spec.loader is not None
    scheduler_mod = importlib.util.module_from_spec(scheduler_spec)

    # Snapshot EVERY key we are about to touch — stubs plus the two real
    # modules loaded below — BEFORE overwriting anything. Restoring from
    # this snapshot puts sys.modules back to its pre-test identity.
    touched = set(stubs) | {"vllm.v1.events", "vllm.v1.core.sched.scheduler"}
    saved = {name: sys.modules.get(name) for name in touched}
    sys.modules.update(stubs)
    sys.modules["vllm.v1.events"] = events_mod
    events_spec.loader.exec_module(events_mod)
    sys.modules["vllm.v1.core.sched.scheduler"] = scheduler_mod
    try:
        scheduler_spec.loader.exec_module(scheduler_mod)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return scheduler_mod


def _make_request(scheduler_mod, request_id: str = "request-1"):
    """A PREEMPTED request with the minimal attributes schedule() touches."""
    req = MagicMock()
    req.request_id = request_id
    req.status = scheduler_mod.RequestStatus.PREEMPTED
    req.num_computed_tokens = 0
    req.num_tokens_with_spec = 16
    req.num_tokens = 16  # real int: used in scheduling math
    req.num_output_placeholders = 0
    req.spec_token_ids = []
    req.num_preemptions = 1
    req.has_encoder_inputs = False
    req.is_prefill_chunk = False
    req.next_decode_eligible_step = 0
    req.num_prompt_tokens = 16
    req.max_tokens = 128
    req.prompt_token_ids = [1] * 16
    req.output_token_ids = []
    req.lora_request = None
    return req


def _make_scheduler(scheduler_mod, request):
    """Controllable stand-in for Scheduler with the real schedule() bound."""
    sched = MagicMock()
    sched.log_stats = True  # scheduled event fires inside the log_stats branch
    sched.current_step = 0
    sched._pause_state = scheduler_mod.PauseState.UNPAUSED
    sched.max_num_scheduled_tokens = 4096
    sched.max_num_encoder_input_tokens = 0
    sched.max_num_running_reqs = 16
    sched.max_model_len = 4096
    sched.num_sampled_tokens_per_step = 1
    sched.prefill_capacity_bound = False
    sched.need_mamba_block_aligned_split = False
    sched.is_encoder_decoder = False
    sched.use_eagle = False
    sched.scheduler_reserve_full_isl = False
    sched.num_lookahead_tokens = 0
    sched.defer_block_free = False
    sched.sched_step_seq = 0
    sched.ec_connector = None
    sched.connector = None
    sched.connector_prefix_cache_stats = None
    sched.lora_config = None
    sched.use_v2_model_runner = False
    sched.observability_config = MagicMock(kv_cache_metrics=False)
    sched.kv_metrics_collector = None
    sched.victim_selector = MagicMock()
    sched.policy = MagicMock()
    sched.running = []  # real list: len() must yield an int
    sched.num_waiting_for_streaming_input = 0
    sched.num_spec_tokens = 0
    sched.dynamic_sd_lookup = None
    sched.num_tokens = 0  # used as a plain int in scheduling math
    sched._last_sd_lookup_schedule = 0
    sched.max_num_common_prefix_blocks = 0
    sched.scheduler_config = MagicMock(long_prefill_token_threshold=0)

    waiting_queue = MagicMock()
    waiting_queue.peek_request.return_value = request
    state = {"popped": False}

    def _pop():
        state["popped"] = True
        return request

    def _bool():
        # Queue is non-empty until the single request has been popped.
        return not state["popped"]

    waiting_queue.pop_request.side_effect = _pop
    waiting_queue.__bool__.side_effect = _bool
    waiting_queue.prepend_request = MagicMock()
    sched.waiting = waiting_queue
    sched.skipped_waiting = MagicMock()
    sched.skipped_waiting.__bool__.return_value = False

    sched._select_waiting_queue_for_scheduling = MagicMock(return_value=waiting_queue)
    sched._is_blocked_waiting_status = MagicMock(return_value=False)
    sched._try_promote_blocked_waiting_request = MagicMock(return_value=True)
    sched._inflight_prefills = set()
    sched._inflight_prefill_reserved_blocks = MagicMock(return_value=0)
    sched._get_num_common_prefix_blocks = MagicMock(return_value=[])

    kv = MagicMock()
    kv.new_step_starts = MagicMock()
    kv.allocate_slots = MagicMock(return_value=MagicMock())
    kv.get_blocks = MagicMock(return_value=[])
    kv.get_computed_blocks = MagicMock(return_value=([], 0))
    kv.create_kv_cache_blocks = MagicMock(return_value=[])
    kv.coordinator = None
    kv.log_stats = False
    kv.prefix_cache_stats = None
    sched.kv_cache_manager = kv
    sched.encoder_cache_manager = MagicMock()
    sched.encoder_cache_manager.free = MagicMock()
    sched.encoder_cache_manager.allocate = MagicMock()
    return sched


def _scheduler_touched_keys() -> set[str]:
    """Every sys.modules key the scheduler loader overwrites."""
    stub_names = {
        "torch",
        "numpy",
        "vllm",
        "vllm.compilation",
        "vllm.compilation.cuda_graph",
        "vllm.config",
        "vllm.distributed",
        "vllm.distributed.ec_transfer",
        "vllm.distributed.ec_transfer.ec_connector",
        "vllm.distributed.ec_transfer.ec_connector.base",
        "vllm.distributed.ec_transfer.ec_connector.factory",
        "vllm.distributed.kv_events",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.factory",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        "vllm.logger",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer",
        "vllm.multimodal",
        "vllm.multimodal.encoder_budget",
        "vllm.multimodal.utils",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.interface",
        "vllm.v1.core.sched.output",
        "vllm.v1.core.sched.policy",
        "vllm.v1.core.sched.request_queue",
        "vllm.v1.core.sched.utils",
        "vllm.v1.core.sched.victim_selector",
        "vllm.v1.core.encoder_cache_manager",
        "vllm.v1.core.kv_cache_coordinator",
        "vllm.v1.core.kv_cache_manager",
        "vllm.v1.core.kv_cache_metrics",
        "vllm.v1.core.kv_cache_utils",
        "vllm.v1.engine",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.kv_cache_compression",
        "vllm.v1.metrics",
        "vllm.v1.metrics.perf",
        "vllm.v1.metrics.stats",
        "vllm.v1.outputs",
        "vllm.v1.request",
        "vllm.v1.spec_decode",
        "vllm.v1.spec_decode.dynamic",
        "vllm.v1.spec_decode.dynamic.utils",
        "vllm.v1.spec_decode.metrics",
        "vllm.v1.structured_output",
        "vllm.v1.utils",
    }
    return stub_names | {"vllm.v1.events", "vllm.v1.core.sched.scheduler"}


def test_restore_request_emits_wakeup_admission_scheduled(monkeypatch) -> None:
    """One PREEMPTED request going through schedule() must emit the exact
    chain RequestResumed -> RequestAdmitted -> RequestScheduled."""
    scheduler_mod = _load_scheduler_module()

    emitted: list = []

    class _Sink:
        def emit(self, event):
            emitted.append(event)

    scheduler_mod.EventBus.register_sink(_Sink())
    try:
        request = _make_request(scheduler_mod)
        sched = _make_scheduler(scheduler_mod, request)

        result = scheduler_mod.Scheduler.schedule(sched)
        assert result is not None

        assert [type(e).__name__ for e in emitted] == [
            "RequestResumed",
            "RequestAdmitted",
            "RequestScheduled",
        ], f"restore chain order wrong: {emitted}"
    finally:
        scheduler_mod.EventBus._sinks = []
        scheduler_mod.EventBus.enabled = False


def test_scheduler_loader_restores_sys_modules_identity() -> None:
    """Running the loader must leave sys.modules in its exact prior state.

    Regression guard for test isolation: every module key the loader touches
    (stubs + dynamically loaded real modules) must be restored to the same
    object, or removed if it did not exist before.
    """
    touched = _scheduler_touched_keys()
    before = {name: sys.modules.get(name) for name in touched}

    _load_scheduler_module()

    after = {name: sys.modules.get(name) for name in touched}
    assert set(before) == set(after)
    for name in touched:
        assert after[name] is before[name], (
            f"sys.modules[{name!r}] identity changed by loader: "
            f"{before[name]!r} -> {after[name]!r}"
        )
