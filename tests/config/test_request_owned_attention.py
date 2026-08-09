# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the experimental request-owned attention (G2) config gate.

The feature flag lives on SchedulerConfig and defaults to False. When enabled,
VllmConfig must construct within the supported G2 envelope: Multiproc/non-Ray
execution, PP=1, eager execution without CUDA graphs, no speculative decoding,
no KV cache CPU offload/tiering or KV transfer/connectors, prefix caching
disabled, DCP=1, PCP=1, and synchronous scheduling. Every unsupported mode
must fail closed with a specific diagnostic: pipeline parallelism, Ray,
speculative decoding, non-eager compilation/CUDA graphs, KV cache CPU
offload, KV transfer, prefix caching, decode/prefill context parallelism,
async scheduling, and multimodal/encoder-decoder models.

The G3 envelope adds construction-time rejections for combinations whose
runtime semantics the request-owned path does not implement yet: Model
Runner V2, dynamic batch overlap / micro-batching (DBO), distributed EC
cache transfer, KV-sharing fast prefill, any executor other than Multiproc
(UniProc tolerated only for single-process host/control tests), non-
generative (pooling/draft) runner types, and sequence parallelism / async
TP. FlashComm and the optional O-projection TP split are plugin-side
switches with no core construction-time field, so they are not asserted
here and keep their runtime gates.
"""

import types

import pytest

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ECTransferConfig,
    KVTransferConfig,
    ParallelConfig,
    PassConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)


def _eager_compilation_config() -> CompilationConfig:
    return CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )


def _speculative_config() -> SpeculativeConfig:
    return SpeculativeConfig(model="[ngram]", num_speculative_tokens=3)


def _active_kv_transfer_config() -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
    )


def _vllm_config(
    *,
    enable_request_owned_attention: bool = True,
    scheduler_config: SchedulerConfig | None = None,
    compilation_config: CompilationConfig | None = None,
    parallel_config: ParallelConfig | None = None,
    cache_config: CacheConfig | None = None,
    speculative_config: SpeculativeConfig | None = None,
    kv_transfer_config: KVTransferConfig | None = None,
    ec_transfer_config: ECTransferConfig | None = None,
    async_scheduling: bool | None = False,
) -> VllmConfig:
    """Build a VllmConfig around the request-owned attention envelope.

    By default the supported G2 envelope is used: eager execution, PP=1,
    no speculative decoding, no KV cache CPU offload/tiering or KV transfer,
    prefix caching disabled, DCP=1, PCP=1, and synchronous scheduling.
    Passing any explicit config replaces exactly that part of the envelope.
    """
    return VllmConfig(
        scheduler_config=scheduler_config
        or SchedulerConfig.default_factory(
            enable_request_owned_attention=enable_request_owned_attention,
            async_scheduling=async_scheduling,
        ),
        compilation_config=compilation_config or _eager_compilation_config(),
        parallel_config=parallel_config or ParallelConfig(),
        cache_config=cache_config or CacheConfig(enable_prefix_caching=False),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
        ec_transfer_config=ec_transfer_config,
    )


def test_request_owned_attention_defaults_off():
    assert SchedulerConfig.default_factory().enable_request_owned_attention is False
    assert VllmConfig().scheduler_config.enable_request_owned_attention is False


def test_disabled_leaves_existing_validation_unchanged():
    # When disabled, every combination that constructs today must still
    # construct; the request-owned gate must not alter existing behavior.
    vllm_config = _vllm_config(
        enable_request_owned_attention=False,
        compilation_config=CompilationConfig(),
        parallel_config=ParallelConfig(pipeline_parallel_size=2),
        cache_config=CacheConfig(kv_offloading_size=1.0),
        speculative_config=_speculative_config(),
        kv_transfer_config=_active_kv_transfer_config(),
    )
    assert vllm_config.scheduler_config.enable_request_owned_attention is False


def test_enabled_supported_envelope_constructs():
    # The supported G2 envelope (Multiproc, PP=1, eager, no speculative
    # decoding, no KV offload/transfer, prefix caching disabled, DCP=1,
    # PCP=1, synchronous scheduling) must construct and keep the flag on.
    vllm_config = _vllm_config(
        enable_request_owned_attention=True,
        compilation_config=_eager_compilation_config(),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
        ),
        cache_config=CacheConfig(enable_prefix_caching=False),
        async_scheduling=False,
    )
    assert vllm_config.scheduler_config.enable_request_owned_attention is True
    assert vllm_config.scheduler_config.async_scheduling is False
    # The allowed envelope is Multiproc / eager / V1 / no-DBO.
    assert vllm_config.parallel_config.distributed_executor_backend == "mp"
    assert vllm_config.use_v2_model_runner is False
    assert vllm_config.parallel_config.use_ubatching is False


def test_enabled_rejects_pipeline_parallelism():
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        _vllm_config(parallel_config=ParallelConfig(pipeline_parallel_size=2))


def test_enabled_rejects_ray_executor():
    # Ray is intentionally not installed in this NPU environment. Mutate an
    # already validated disabled config, then invoke the feature validation
    # directly so the test reaches this gate without importing optional Ray.
    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.parallel_config.distributed_executor_backend = "ray"
    with pytest.raises(ValueError, match="does not support the Ray executor"):
        vllm_config._validate_request_owned_attention()


def test_enabled_rejects_speculative_decoding():
    with pytest.raises(ValueError, match="speculative"):
        _vllm_config(speculative_config=_speculative_config())


@pytest.mark.parametrize(
    "mode",
    [
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ],
)
def test_enabled_rejects_non_eager_compilation(mode):
    with pytest.raises(ValueError, match="compilation_config.mode"):
        _vllm_config(
            compilation_config=CompilationConfig(
                mode=mode,
                cudagraph_mode=CUDAGraphMode.NONE,
            )
        )


def test_enabled_rejects_default_compilation():
    # The default compilation config does not resolve to eager execution
    # (mode=None is resolved to a compile mode by VllmConfig), so enabling
    # request-owned attention must fail closed unless eager is explicit.
    resolved = VllmConfig(
        compilation_config=CompilationConfig(),
    ).compilation_config.mode
    if resolved == CompilationMode.NONE:
        pytest.skip("default compilation resolves to eager in this environment")
    with pytest.raises(ValueError, match="compilation_config.mode"):
        _vllm_config(compilation_config=CompilationConfig())


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ],
)
def test_enabled_rejects_cudagraph_execution(cudagraph_mode):
    # PIECEWISE-family modes are overridden to NONE by VllmConfig when the
    # compilation mode is not VLLM_COMPILE, so only modes that survive that
    # resolution can reject on cudagraph_mode itself.
    with pytest.raises(ValueError, match="cudagraph_mode"):
        _vllm_config(
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE,
                cudagraph_mode=cudagraph_mode,
            )
        )


def test_enabled_rejects_kv_cache_cpu_offload():
    with pytest.raises(ValueError, match="kv_offloading_size"):
        _vllm_config(cache_config=CacheConfig(kv_offloading_size=1.0))


def test_enabled_rejects_kv_transfer():
    # A manually configured KV connector enables transfer/offload/tiering
    # even with kv_offloading_size=None, so it must fail closed as well.
    with pytest.raises(ValueError, match="kv_transfer_config"):
        _vllm_config(kv_transfer_config=_active_kv_transfer_config())


def test_enabled_rejects_prefix_caching():
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        _vllm_config(cache_config=CacheConfig(enable_prefix_caching=True))


def test_enabled_rejects_decode_context_parallelism():
    # TP must be divisible by DCP for ParallelConfig to construct, so the
    # offending configuration uses DCP=2 over TP=2.
    with pytest.raises(ValueError, match="decode_context_parallel_size"):
        _vllm_config(
            parallel_config=ParallelConfig(
                tensor_parallel_size=2,
                decode_context_parallel_size=2,
            )
        )


def test_enabled_rejects_prefill_context_parallelism():
    with pytest.raises(ValueError, match="prefill_context_parallel_size"):
        _vllm_config(parallel_config=ParallelConfig(prefill_context_parallel_size=2))


def test_enabled_rejects_async_scheduling():
    with pytest.raises(ValueError, match="async_scheduling"):
        _vllm_config(async_scheduling=True)


def test_enabled_rejects_resolved_async_scheduling_default():
    # The default (async_scheduling=None) resolves to True on the Multiproc
    # executor before feature validation runs, so the enabled flag must fail
    # closed on the resolved value rather than only on an explicit request.
    with pytest.raises(ValueError, match="async_scheduling"):
        _vllm_config(async_scheduling=None)


def test_enabled_rejects_multimodal_model():
    # A real ModelConfig cannot be constructed without a model path, so the
    # feature validation is invoked directly on an already validated config
    # with a minimal model_config stub exposing the stable predicates.
    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.model_config = types.SimpleNamespace(
        is_multimodal_model=True,
        is_encoder_decoder=False,
    )
    with pytest.raises(ValueError, match="multimodal"):
        vllm_config._validate_request_owned_attention()


def test_enabled_rejects_encoder_decoder_model():
    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.model_config = types.SimpleNamespace(
        is_multimodal_model=False,
        is_encoder_decoder=True,
    )
    with pytest.raises(ValueError, match="encoder-decoder"):
        vllm_config._validate_request_owned_attention()


def test_enabled_rejects_model_runner_v2(monkeypatch):
    # Model Runner V2 is selected at construction through the
    # VLLM_USE_V2_MODEL_RUNNER switch; request-owned attention is implemented
    # on the V1 model runner and must fail closed on the resolved value.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with pytest.raises(ValueError, match="V1 model runner"):
        _vllm_config()


@pytest.mark.parametrize(
    "parallel_config",
    [
        ParallelConfig(enable_dbo=True),
        ParallelConfig(ubatch_size=2),
    ],
)
def test_enabled_rejects_dbo_ubatching(parallel_config):
    with pytest.raises(ValueError, match="DBO"):
        _vllm_config(parallel_config=parallel_config)


def test_enabled_rejects_ec_transfer():
    # A manually configured EC connector enables distributed EC cache
    # transfer, so it must fail closed as well.
    with pytest.raises(ValueError, match="ec_transfer_config"):
        _vllm_config(
            ec_transfer_config=ECTransferConfig(
                ec_connector="TorchDistributedConnector",
                ec_role="ec_producer",
            )
        )


def test_enabled_rejects_kv_sharing_fast_prefill():
    # CacheConfig defaults to enable_prefix_caching=True, which the G2 gate
    # already rejects; keep the rest of the envelope allowed so this test
    # isolates the KV-sharing fast-prefill rejection.
    with pytest.raises(ValueError, match="kv_sharing_fast_prefill"):
        _vllm_config(
            cache_config=CacheConfig(
                enable_prefix_caching=False,
                kv_sharing_fast_prefill=True,
            )
        )


def test_enabled_rejects_external_launcher_executor():
    # An external-launcher executor changes the process/runtime contract, so
    # it must fail closed. Mutate an already validated disabled config, then
    # invoke the feature validation directly (matching the Ray test pattern).
    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.parallel_config.distributed_executor_backend = "external_launcher"
    with pytest.raises(ValueError, match="Multiproc executor"):
        vllm_config._validate_request_owned_attention()


def test_enabled_rejects_custom_executor_class():
    # A custom executor backend is passed as an Executor subclass; the gate
    # must reject any non-Multiproc class. The stand-in class exercises the
    # custom-executor path without importing the runtime executor package.
    class CustomExecutor:
        uses_ray = False

    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.parallel_config.distributed_executor_backend = CustomExecutor
    with pytest.raises(ValueError, match="custom executor"):
        vllm_config._validate_request_owned_attention()


def test_enabled_rejects_uniproc_beyond_host_control():
    # UniProc is tolerated only for single-process host/control tests; an
    # explicit 'uni' backend over a multi-rank world must fail closed.
    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.parallel_config.distributed_executor_backend = "uni"
    vllm_config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="world_size"):
        vllm_config._validate_request_owned_attention()


@pytest.mark.parametrize("runner_type", ["pooling", "draft"])
def test_enabled_rejects_non_generate_runner(runner_type):
    # Pooling/draft runner types do not produce the generate-shaped output
    # the request-owned path expects and must fail closed at construction.
    with pytest.raises(ValueError, match="runner_type"):
        _vllm_config(
            scheduler_config=SchedulerConfig.default_factory(
                enable_request_owned_attention=True,
                async_scheduling=False,
                runner_type=runner_type,
            )
        )


def test_enabled_rejects_nested_mutation_kv_sharing_fast_prefill():
    # Post-init nested config mutation must be caught by explicit
    # re-validation on the same VllmConfig instance.
    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.cache_config.kv_sharing_fast_prefill = True
    with pytest.raises(ValueError, match="kv_sharing_fast_prefill"):
        vllm_config._validate_request_owned_attention()


def test_enabled_rejects_async_tp_fuse_gemm_comms():
    # Async TP (fuse_gemm_comms) implies sequence parallelism and changes
    # the collective semantics of the request-owned path; it must fail
    # closed at construction.
    with pytest.raises(ValueError, match="sequence parallelism"):
        _vllm_config(
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.NONE,
                pass_config=PassConfig(fuse_gemm_comms=True),
            )
        )


def test_enabled_rejects_sequence_parallelism_enable_sp():
    # The explicit enable_sp=True switch reaches the same gate. Real
    # construction with a model also reaches it; the post-init mutation path
    # is used here because the Ascend platform plugin's construction-time
    # defaults crash on a model_config-less test config before the
    # request-owned gate runs.
    vllm_config = _vllm_config(enable_request_owned_attention=False)
    vllm_config.scheduler_config.enable_request_owned_attention = True
    vllm_config.compilation_config.pass_config.enable_sp = True
    with pytest.raises(ValueError, match="sequence parallelism"):
        vllm_config._validate_request_owned_attention()


def test_hash_differs_when_enabled():
    # Request-owned attention changes the computation graph (per-request
    # attention kernel selection), so the scheduler config hash must
    # distinguish it. VllmConfig.compute_hash() folds in
    # scheduler_config.compute_hash(), and the enabled envelope constructs,
    # so the distinction is checked at the SchedulerConfig level.
    disabled = SchedulerConfig.default_factory()
    enabled = SchedulerConfig.default_factory(
        enable_request_owned_attention=True,
    )
    assert disabled.compute_hash() != enabled.compute_hash()
