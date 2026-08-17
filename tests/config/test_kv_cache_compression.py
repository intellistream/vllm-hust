# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm.config import KVCacheCompressionConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _config(**overrides) -> KVCacheCompressionConfig:
    provider_config = {
        "max_capacity_prompt": 512,
        "window_size": 8,
        "kernel_size": 7,
        "pooling": "maxpool",
        "beta": 20,
        "kv_cache_granularity": "kv_head",
        "gqa_score_aggregation": "mean",
        "merge": None,
    }
    provider_config.update(overrides)
    return KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config=provider_config,
    )


def test_disabled_by_default_and_does_not_change_hash() -> None:
    baseline = VllmConfig()
    disabled = VllmConfig(kv_cache_compression_config=None)

    assert baseline.kv_cache_compression_config is None
    assert disabled.kv_cache_compression_config is None
    assert baseline.compute_hash() == disabled.compute_hash()


def test_new_config_fields_are_appended_for_positional_compatibility() -> None:
    assert fields(VllmConfig)[-1].name == "kv_cache_compression_config"
    assert fields(EngineArgs)[-1].name == "kv_cache_compression_config"


def test_enabled_config_changes_vllm_hash() -> None:
    disabled_hash = VllmConfig().compute_hash()
    enabled_hash = VllmConfig(kv_cache_compression_config=_config()).compute_hash()

    assert enabled_hash != disabled_hash


def test_hash_is_stable_across_provider_option_order() -> None:
    config = _config()
    reversed_options = dict(reversed(list(config.provider_config.items())))
    equivalent = KVCacheCompressionConfig(
        provider=config.provider,
        provider_config=reversed_options,
    )

    assert config.compute_hash() == equivalent.compute_hash()


def test_json_round_trip() -> None:
    adapter = TypeAdapter(KVCacheCompressionConfig)
    config = _config()

    restored = adapter.validate_json(adapter.dump_json(config))

    assert restored == config
    assert restored.provider_config["merge"] is None


def test_cli_and_python_use_same_config_type(tmp_path: Path) -> None:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    payload = {
        "schema_version": 1,
        "provider": "pyramidkv_ascend",
        "provider_config": _config().provider_config,
    }

    args = parser.parse_args(
        [
            "--model",
            str(tmp_path),
            "--kv-cache-compression-config",
            json.dumps(payload),
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)

    assert args.kv_cache_compression_config == _config()
    assert engine_args.kv_cache_compression_config == _config()


def test_cli_is_disabled_by_default(tmp_path: Path) -> None:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    args = parser.parse_args(["--model", str(tmp_path)])

    assert args.kv_cache_compression_config is None
    assert EngineArgs.from_cli_args(args).kv_cache_compression_config is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": ""},
        {"provider": "pyramidkv_ascend", "schema_version": 2},
        {
            "provider": "pyramidkv_ascend",
            "provider_config": {"nested": {"not": "allowed"}},
        },
        {
            "provider": "pyramidkv_ascend",
            "provider_config": {"items": [1, 2]},
        },
        {
            "provider": "pyramidkv_ascend",
            "provider_config": {"beta": float("nan")},
        },
    ],
)
def test_invalid_config(kwargs) -> None:
    with pytest.raises((ValidationError, ValueError)):
        KVCacheCompressionConfig(**kwargs)


def test_unknown_top_level_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(KVCacheCompressionConfig).validate_python(
            {
                "provider": "pyramidkv_ascend",
                "provider_config": {},
                "unknown": True,
            }
        )
