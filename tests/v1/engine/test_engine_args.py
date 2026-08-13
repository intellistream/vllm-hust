# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError

import pytest

from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms.hardware_defaults import (
    get_current_accelerator_scheduling_defaults,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import _xxhash


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching, (
        "V1 turns on prefix caching by default."
    )

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching

    # HUST defaults to xxhash for lower prefix-cache block hashing overhead.
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)

    from vllm.platforms import current_platform

    if current_platform.is_cpu():
        default_llm_tokens = 4096
        default_server_tokens = 2048
        default_max_num_seqs = 256
    elif current_platform.is_tpu():
        chip_name = current_platform.get_device_name()
        if chip_name == "V6E":
            default_llm_tokens = 2048
            default_server_tokens = 1024
        elif chip_name == "V5E":
            default_llm_tokens = 1024
            default_server_tokens = 512
        elif chip_name == "V5P":
            default_llm_tokens = 512
            default_server_tokens = 256
        else:
            defaults = get_current_accelerator_scheduling_defaults()
            default_llm_tokens = defaults.llm_class_max_num_batched_tokens
            default_server_tokens = defaults.api_server_max_num_batched_tokens
        default_max_num_seqs = 256
    else:
        defaults = get_current_accelerator_scheduling_defaults()
        default_llm_tokens = defaults.llm_class_max_num_batched_tokens
        default_server_tokens = defaults.api_server_max_num_batched_tokens
        default_max_num_seqs = defaults.max_num_seqs

    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501


def test_mm_prefix_lm_raises_batched_tokens_floor():
    """Verify that prefix-LM multimodal models auto-raise
    max_num_batched_tokens to fit at least one multimodal item.

    Regression test for https://github.com/vllm-project/vllm/issues/42687
    """
    from unittest.mock import patch

    # Simulate a prefix-LM multimodal model whose largest modality
    # (video) requires 2496 tokens — more than the 2048 default.
    fake_mm_min = (2496, "video")

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_model_len=2048,
        enforce_eager=True,
    )

    with (
        patch.object(
            type(engine_args),
            "_get_min_mm_batched_tokens",
            staticmethod(lambda _mc: fake_mm_min),
        ),
        patch(
            "vllm.config.ModelConfig.is_multimodal_model",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "vllm.config.ModelConfig.is_mm_prefix_lm",
            new_callable=lambda: property(lambda self: True),
        ),
    ):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.scheduler_config.max_num_batched_tokens >= 2496


def test_request_owned_flags_default_false():
    """All experimental request-owned flags default to False and never
    change the scheduler config unless explicitly enabled."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    engine_args = EngineArgs.from_cli_args(args=parser.parse_args([]))
    assert engine_args.enable_request_owned_attention is False
    assert engine_args.enable_request_owned_q_wkv_fanin is False
    assert engine_args.enable_request_owned_sampling is False
    assert engine_args.enable_request_owned_graph is False
    assert engine_args.enable_request_owned_kv_offload is False
    assert engine_args.enable_request_owned_windows is False
    assert engine_args.request_owned_decode_window_steps == 1
    assert engine_args.request_owned_decode_reservation_tokens is None

    vllm_config = engine_args.create_engine_config()
    assert vllm_config.scheduler_config.enable_request_owned_attention is False
    assert vllm_config.scheduler_config.enable_request_owned_q_wkv_fanin is False
    assert vllm_config.scheduler_config.enable_request_owned_sampling is False
    assert vllm_config.scheduler_config.enable_request_owned_graph is False
    assert vllm_config.scheduler_config.enable_request_owned_kv_offload is False
    assert vllm_config.scheduler_config.enable_request_owned_windows is False
    assert vllm_config.scheduler_config.request_owned_decode_window_steps == 1
    assert vllm_config.scheduler_config.request_owned_decode_reservation_tokens is None


def test_request_owned_kv_offload_cli_is_explicit_opt_in():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--enable-request-owned-kv-offload"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_request_owned_kv_offload is True


def test_request_owned_windows_cli_parses_quantum_without_model_loading():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--enable-request-owned-attention",
            "--enable-request-owned-graph",
            "--enable-request-owned-windows",
            "--request-owned-decode-window-steps",
            "7",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_request_owned_windows is True
    assert engine_args.request_owned_decode_window_steps == 7


def test_request_owned_decode_reservation_cli_parses_without_model_loading():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--request-owned-decode-reservation-tokens", "2"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.request_owned_decode_reservation_tokens == 2


def test_request_owned_attention_flag_propagates_independently(monkeypatch):
    """--enable-request-owned-attention alone maps exactly to the
    SchedulerConfig field, leaving sampling off."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--distributed-executor-backend",
            "mp",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--enable-request-owned-attention",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_request_owned_attention is True
    assert engine_args.enable_request_owned_q_wkv_fanin is False
    assert engine_args.enable_request_owned_sampling is False
    assert engine_args.enable_request_owned_graph is False

    vllm_config = engine_args.create_engine_config()
    assert vllm_config.scheduler_config.enable_request_owned_attention is True
    assert vllm_config.scheduler_config.enable_request_owned_q_wkv_fanin is False
    assert vllm_config.scheduler_config.enable_request_owned_sampling is False
    assert vllm_config.scheduler_config.enable_request_owned_graph is False


def test_request_owned_sampling_flag_propagates_independently():
    """--enable-request-owned-sampling alone reaches its own EngineArgs field,
    and existing VllmConfig validation still rejects it without the
    request-owned attention envelope."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--distributed-executor-backend",
            "mp",
            "--enable-request-owned-sampling",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_request_owned_attention is False
    assert engine_args.enable_request_owned_sampling is True

    with pytest.raises(ValueError, match="enable_request_owned_attention=True"):
        engine_args.create_engine_config()


def test_request_owned_q_wkv_fanin_cli_reaches_scheduler_config(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--distributed-executor-backend",
            "mp",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--enable-request-owned-attention",
            "--enable-request-owned-q-wkv-fanin",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_request_owned_attention is True
    assert engine_args.enable_request_owned_q_wkv_fanin is True

    vllm_config = engine_args.create_engine_config()
    assert vllm_config.scheduler_config.enable_request_owned_attention is True
    assert vllm_config.scheduler_config.enable_request_owned_q_wkv_fanin is True


def test_request_owned_flags_together_reach_scheduler_config(monkeypatch):
    """The vllm serve CLI form (Multiproc + both flags) parses and reaches
    scheduler config construction with both booleans enabled."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--distributed-executor-backend",
            "mp",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--enable-request-owned-attention",
            "--enable-request-owned-sampling",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_request_owned_attention is True
    assert engine_args.enable_request_owned_sampling is True
    assert engine_args.enable_request_owned_graph is False

    vllm_config = engine_args.create_engine_config()
    assert vllm_config.scheduler_config.enable_request_owned_attention is True
    assert vllm_config.scheduler_config.enable_request_owned_sampling is True
    assert vllm_config.scheduler_config.enable_request_owned_graph is False


def test_request_owned_graph_cli_reaches_exact_piecewise_lane(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--distributed-executor-backend",
            "mp",
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--enable-request-owned-attention",
            "--enable-request-owned-sampling",
            "--enable-request-owned-graph",
            "--compilation-config",
            '{"mode":3,"cudagraph_mode":"PIECEWISE"}',
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_request_owned_graph is True

    vllm_config = engine_args.create_engine_config()
    assert vllm_config.scheduler_config.enable_request_owned_graph is True
    assert vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_request_owned_invalid_envelope_still_rejected():
    """Enabling the flags does not widen the fail-closed envelope: an
    unsupported pipeline shape is still rejected by existing VllmConfig
    validation."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--distributed-executor-backend",
            "mp",
            "--pipeline-parallel-size",
            "2",
            "--enable-request-owned-attention",
            "--enable-request-owned-sampling",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args=args)

    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        engine_args.create_engine_config()
