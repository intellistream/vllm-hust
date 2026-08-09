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
"""

import types

import pytest

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    KVTransferConfig,
    ParallelConfig,
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
    compilation_config: CompilationConfig | None = None,
    parallel_config: ParallelConfig | None = None,
    cache_config: CacheConfig | None = None,
    speculative_config: SpeculativeConfig | None = None,
    kv_transfer_config: KVTransferConfig | None = None,
    async_scheduling: bool | None = False,
) -> VllmConfig:
    """Build a VllmConfig around the request-owned attention envelope.

    By default the supported G2 envelope is used: eager execution, PP=1,
    no speculative decoding, no KV cache CPU offload/tiering or KV transfer,
    prefix caching disabled, DCP=1, PCP=1, and synchronous scheduling.
    Passing any explicit config replaces exactly that part of the envelope.
    """
    return VllmConfig(
        scheduler_config=SchedulerConfig.default_factory(
            enable_request_owned_attention=enable_request_owned_attention,
            async_scheduling=async_scheduling,
        ),
        compilation_config=compilation_config or _eager_compilation_config(),
        parallel_config=parallel_config or ParallelConfig(),
        cache_config=cache_config or CacheConfig(enable_prefix_caching=False),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
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
        _vllm_config(
            parallel_config=ParallelConfig(prefill_context_parallel_size=2)
        )


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
