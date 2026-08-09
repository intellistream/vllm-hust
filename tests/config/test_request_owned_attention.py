# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the experimental request-owned attention (G0) config gate.

The feature flag lives on SchedulerConfig and defaults to False. When enabled,
VllmConfig must fail closed on unsupported modes (pipeline parallelism,
speculative decoding, non-eager/CUDA-graph execution, KV cache CPU offload).
"""

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
) -> VllmConfig:
    """Build a VllmConfig around the request-owned attention envelope.

    By default the supported G0 envelope is used: eager execution, PP=1,
    no speculative decoding, no KV cache CPU offload/tiering or KV transfer.
    """
    return VllmConfig(
        scheduler_config=SchedulerConfig.default_factory(
            enable_request_owned_attention=enable_request_owned_attention,
        ),
        compilation_config=compilation_config or _eager_compilation_config(),
        parallel_config=parallel_config or ParallelConfig(),
        cache_config=cache_config or CacheConfig(),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
    )


def test_request_owned_attention_defaults_off():
    assert SchedulerConfig.default_factory().enable_request_owned_attention is False
    assert VllmConfig().scheduler_config.enable_request_owned_attention is False


def test_disabled_leaves_existing_validation_unchanged():
    # When disabled, every combination that constructs today must still
    # construct; the G0 gate must not alter existing behavior.
    vllm_config = _vllm_config(
        enable_request_owned_attention=False,
        compilation_config=CompilationConfig(),
        parallel_config=ParallelConfig(pipeline_parallel_size=2),
        cache_config=CacheConfig(kv_offloading_size=1.0),
        speculative_config=_speculative_config(),
        kv_transfer_config=_active_kv_transfer_config(),
    )
    assert vllm_config.scheduler_config.enable_request_owned_attention is False


def test_enabled_accepts_supported_envelope():
    # Chunked prefill (default), async scheduling (default resolution),
    # TP>1 and prefix caching are all allowed inside the G0 envelope; the
    # gate must not reject them.
    vllm_config = _vllm_config(
        enable_request_owned_attention=True,
        compilation_config=_eager_compilation_config(),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
        ),
        cache_config=CacheConfig(enable_prefix_caching=True),
    )
    assert vllm_config.scheduler_config.enable_request_owned_attention is True
    assert vllm_config.scheduler_config.enable_chunked_prefill is True


def test_enabled_rejects_pipeline_parallelism():
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        _vllm_config(parallel_config=ParallelConfig(pipeline_parallel_size=2))


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


def test_hash_differs_when_enabled():
    # Request-owned attention changes the computation graph (per-request
    # attention kernel selection), so the config hash must distinguish it,
    # both at the SchedulerConfig level and through VllmConfig.compute_hash().
    disabled = SchedulerConfig.default_factory()
    enabled = SchedulerConfig.default_factory(
        enable_request_owned_attention=True,
    )
    assert disabled.compute_hash() != enabled.compute_hash()

    compilation_config = _eager_compilation_config()
    disabled_vllm = _vllm_config(
        enable_request_owned_attention=False,
        compilation_config=compilation_config,
    )
    enabled_vllm = _vllm_config(
        enable_request_owned_attention=True,
        compilation_config=compilation_config,
    )
    assert disabled_vllm.compute_hash() != enabled_vllm.compute_hash()
