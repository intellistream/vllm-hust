# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the experimental request-owned sampling transport (G3) config gate.

The flag lives on SchedulerConfig and defaults to False. When enabled it is a
transport-only opt-in on top of the request-owned attention envelope: it makes
the owner-sampling aggregator authoritative across the deferred ``execute_model
-> None -> sample_tokens`` Multiproc flow.  ``VllmConfig`` must therefore
require ``enable_request_owned_attention=True`` and strictly
``distributed_executor_backend='mp'``; UniProc (even at world_size=1), Ray,
``external_launcher``, and custom executor classes must fail closed.  The flag
changes no computation graph structure, so the scheduler graph hash must not
depend on it.
"""

import pytest

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)


def _eager_compilation_config() -> CompilationConfig:
    return CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )


def _sampling_vllm_config(
    *,
    enable_request_owned_attention: bool = True,
    enable_request_owned_sampling: bool = True,
    parallel_config: ParallelConfig | None = None,
) -> VllmConfig:
    """Build a VllmConfig inside the request-owned attention envelope with
    the Multiproc executor, ready for the sampling gate."""
    return VllmConfig(
        scheduler_config=SchedulerConfig.default_factory(
            enable_request_owned_attention=enable_request_owned_attention,
            enable_request_owned_sampling=enable_request_owned_sampling,
            async_scheduling=False,
        ),
        compilation_config=_eager_compilation_config(),
        parallel_config=parallel_config
        or ParallelConfig(distributed_executor_backend="mp"),
        cache_config=CacheConfig(enable_prefix_caching=False),
    )


def test_request_owned_sampling_defaults_off():
    assert SchedulerConfig.default_factory().enable_request_owned_sampling is False
    assert VllmConfig().scheduler_config.enable_request_owned_sampling is False


@pytest.mark.parametrize(
    "bad",
    [1, 0, "true", "1", None],
)
def test_sampling_flag_rejects_non_bool(bad):
    """The experimental flag is a strict bool gate: integer/string truthy
    values must fail closed instead of being silently coerced to enabled."""
    with pytest.raises(ValueError, match="must be a bool"):
        SchedulerConfig.default_factory(
            enable_request_owned_attention=False,
            enable_request_owned_sampling=bad,
        )


def test_sampling_requires_request_owned_attention():
    with pytest.raises(ValueError, match="enable_request_owned_attention"):
        _sampling_vllm_config(enable_request_owned_attention=False)


def test_supported_envelope_constructs():
    vllm_config = _sampling_vllm_config()
    assert vllm_config.scheduler_config.enable_request_owned_sampling is True
    assert vllm_config.scheduler_config.enable_request_owned_attention is True
    assert vllm_config.parallel_config.distributed_executor_backend == "mp"


def test_supported_envelope_tp2_constructs():
    # The sampling transport aggregates one receipt + one sampling batch per
    # process-global rank, so a multi-rank Multiproc world must construct.
    vllm_config = _sampling_vllm_config(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
        )
    )
    assert vllm_config.parallel_config.world_size == 2


def test_sampling_rejects_uniproc():
    # UniProc is tolerated by the attention envelope only for single-process
    # host/control tests, but the sampling state machine is proven only for
    # the Multiproc transport, so it must fail closed even at world_size=1.
    with pytest.raises(ValueError, match="distributed_executor_backend='mp'"):
        _sampling_vllm_config(
            parallel_config=ParallelConfig(distributed_executor_backend="uni")
        )


def test_sampling_rejects_ray_executor():
    # Ray is intentionally not installed in this NPU environment. Mutate an
    # already validated disabled config, then invoke the sampling validation
    # directly so the test reaches this gate without importing optional Ray.
    vllm_config = _sampling_vllm_config()
    vllm_config.parallel_config.distributed_executor_backend = "ray"
    with pytest.raises(ValueError, match="does not support the Ray executor"):
        vllm_config._validate_request_owned_sampling()


def test_sampling_rejects_external_launcher():
    vllm_config = _sampling_vllm_config()
    vllm_config.parallel_config.distributed_executor_backend = "external_launcher"
    with pytest.raises(ValueError, match="distributed_executor_backend='mp'"):
        vllm_config._validate_request_owned_sampling()


def test_sampling_rejects_custom_executor_class():
    class CustomExecutor:
        uses_ray = False

    vllm_config = _sampling_vllm_config()
    vllm_config.parallel_config.distributed_executor_backend = CustomExecutor
    with pytest.raises(ValueError, match="distributed_executor_backend='mp'"):
        vllm_config._validate_request_owned_sampling()


def test_sampling_off_leaves_existing_attention_validation_unchanged():
    # With the sampling flag off, the attention envelope keeps its existing
    # tolerance: the same 'uni' single-process config still constructs.
    vllm_config = _sampling_vllm_config(
        enable_request_owned_sampling=False,
        parallel_config=ParallelConfig(distributed_executor_backend="uni"),
    )
    assert vllm_config.scheduler_config.enable_request_owned_attention is True
    assert vllm_config.scheduler_config.enable_request_owned_sampling is False


def test_sampling_flag_does_not_change_graph_hash():
    # The sampling flag selects no computation graph structure, so the
    # SchedulerConfig graph hash must be identical with it on or off.
    on = SchedulerConfig.default_factory(enable_request_owned_sampling=True)
    off = SchedulerConfig.default_factory(enable_request_owned_sampling=False)
    assert on.compute_hash() == off.compute_hash()
