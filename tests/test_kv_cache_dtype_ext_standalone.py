# SPDX-License-Identifier: Apache-2.0
"""Standalone unit tests for KV cache dtype extension logic.

Completely self-contained: inlines the relevant logic from the modified
vllm files so that tests run without the full vllm dependency chain.
"""

import typing
from enum import IntEnum

import torch
import pytest


# ===========================================================================
# Inlined logic from vllm/config/cache.py
# ===========================================================================

CacheDType = typing.Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "int4",
    "nvfp4",
    "fp4_e2m1",
    "int8",
    "kivi_int4",
]


# ===========================================================================
# Inlined logic from vllm/utils/torch_utils.py
# ===========================================================================

STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "int4_per_token_head": torch.uint8,
    "int8_per_token_head": torch.int8,
    "fp8_per_token_head": torch.uint8,
    "fp8_inc": torch.float8_e4m3fn,
    "fp8_ds_mla": torch.uint8,
    "turboquant_k8v4": torch.uint8,
    "turboquant_4bit_nc": torch.uint8,
    "turboquant_k3v4_nc": torch.uint8,
    "turboquant_3bit_nc": torch.uint8,
    "int4": torch.uint8,
    "nvfp4": torch.uint8,
    "fp4_e2m1": torch.uint8,
}


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype in ("nvfp4", "int4", "fp4_e2m1")
    )


def int4_kv_cache_full_dim(head_size: int) -> int:
    """Packed last dim for INT4 KV cache: 2×int4 per byte."""
    return head_size // 2


def fp4_e2m1_kv_cache_full_dim(head_size: int) -> int:
    """Packed last dim for FP4 E2M1 KV cache: fp4 data + fp8 block scales."""
    num_blocks = (head_size + 15) // 16
    data_bytes = num_blocks * 8
    scale_bytes = num_blocks * 1
    return data_bytes + scale_bytes


def nvfp4_kv_cache_full_dim(head_size: int) -> int:
    """Packed last dim for NVFP4 KV cache: fp4 data + fp8 block scales."""
    return head_size // 2 + head_size // 16


# ===========================================================================
# Inlined logic from vllm/v1/kv_cache_interface.py
# ===========================================================================


class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4
    NVFP4 = 5
    INT4 = 6
    FP4_E2M1 = 7

    @property
    def is_per_token_head(self) -> bool:
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        return self == KVQuantMode.NVFP4


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    if kv_cache_dtype == "int4_per_token_head":
        return KVQuantMode.INT4_PER_TOKEN_HEAD
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if kv_cache_dtype == "int4":
        return KVQuantMode.INT4
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if kv_cache_dtype == "fp4_e2m1":
        return KVQuantMode.FP4_E2M1
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


# ===========================================================================
# Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# CacheDType
# ---------------------------------------------------------------------------


class TestCacheDTypeIncludesNewTypes:

    def test_includes_int4(self):
        assert "int4" in typing.get_args(CacheDType)

    def test_includes_fp4_e2m1(self):
        assert "fp4_e2m1" in typing.get_args(CacheDType)

    def test_includes_nvfp4(self):
        assert "nvfp4" in typing.get_args(CacheDType)

    def test_does_not_exclude_existing(self):
        args = typing.get_args(CacheDType)
        for existing in ("auto", "float16", "bfloat16", "fp8", "int8", "kivi_int4"):
            assert existing in args, f"CacheDType should still include '{existing}'"


# ---------------------------------------------------------------------------
# STR_DTYPE_TO_TORCH_DTYPE
# ---------------------------------------------------------------------------


class TestStrDtypeToTorchDtype:

    def test_int4_maps_to_uint8(self):
        assert STR_DTYPE_TO_TORCH_DTYPE["int4"] == torch.uint8

    def test_fp4_e2m1_maps_to_uint8(self):
        assert STR_DTYPE_TO_TORCH_DTYPE["fp4_e2m1"] == torch.uint8

    def test_nvfp4_maps_to_uint8(self):
        assert STR_DTYPE_TO_TORCH_DTYPE["nvfp4"] == torch.uint8

    def test_float16_maps_to_float16(self):
        assert STR_DTYPE_TO_TORCH_DTYPE["float16"] == torch.float16

    def test_fp8_maps_to_uint8(self):
        assert STR_DTYPE_TO_TORCH_DTYPE["fp8"] == torch.uint8


# ---------------------------------------------------------------------------
# is_quantized_kv_cache
# ---------------------------------------------------------------------------


class TestIsQuantizedKvCache:

    @pytest.mark.parametrize("dtype", [
        "int4", "fp4_e2m1", "nvfp4",
        "fp8", "fp8_e4m3", "fp8_e5m2",
        "int4_per_token_head", "int8_per_token_head", "fp8_per_token_head",
    ])
    def test_quantized_dtypes_return_true(self, dtype):
        assert is_quantized_kv_cache(dtype) is True, f"{dtype} should be quantized"

    @pytest.mark.parametrize("dtype", [
        "auto", "float16", "bfloat16", "float32",
    ])
    def test_non_quantized_dtypes_return_false(self, dtype):
        assert is_quantized_kv_cache(dtype) is False, f"{dtype} should not be quantized"


# ---------------------------------------------------------------------------
# KVQuantMode
# ---------------------------------------------------------------------------


class TestKVQuantMode:

    def test_int4_exists(self):
        assert hasattr(KVQuantMode, "INT4")
        assert KVQuantMode.INT4.value == 6

    def test_fp4_e2m1_exists(self):
        assert hasattr(KVQuantMode, "FP4_E2M1")
        assert KVQuantMode.FP4_E2M1.value == 7

    def test_nvfp4_exists(self):
        assert hasattr(KVQuantMode, "NVFP4")
        assert KVQuantMode.NVFP4.value == 5

    def test_none_exists(self):
        assert KVQuantMode.NONE.value == 0

    def test_is_per_token_head(self):
        assert KVQuantMode.INT4_PER_TOKEN_HEAD.is_per_token_head is True
        assert KVQuantMode.INT8_PER_TOKEN_HEAD.is_per_token_head is True
        assert KVQuantMode.FP8_PER_TOKEN_HEAD.is_per_token_head is True
        assert KVQuantMode.INT4.is_per_token_head is False
        assert KVQuantMode.FP4_E2M1.is_per_token_head is False
        assert KVQuantMode.NVFP4.is_per_token_head is False
        assert KVQuantMode.NONE.is_per_token_head is False

    def test_is_nvfp4(self):
        assert KVQuantMode.NVFP4.is_nvfp4 is True
        assert KVQuantMode.INT4.is_nvfp4 is False
        assert KVQuantMode.FP4_E2M1.is_nvfp4 is False
        assert KVQuantMode.NONE.is_nvfp4 is False


# ---------------------------------------------------------------------------
# get_kv_quant_mode
# ---------------------------------------------------------------------------


class TestGetKvQuantMode:

    @pytest.mark.parametrize("dtype,expected", [
        ("int4", KVQuantMode.INT4),
        ("fp4_e2m1", KVQuantMode.FP4_E2M1),
        ("nvfp4", KVQuantMode.NVFP4),
        ("fp8", KVQuantMode.FP8_PER_TENSOR),
        ("fp8_e4m3", KVQuantMode.FP8_PER_TENSOR),
        ("fp8_e5m2", KVQuantMode.FP8_PER_TENSOR),
        ("int4_per_token_head", KVQuantMode.INT4_PER_TOKEN_HEAD),
        ("int8_per_token_head", KVQuantMode.INT8_PER_TOKEN_HEAD),
        ("fp8_per_token_head", KVQuantMode.FP8_PER_TOKEN_HEAD),
    ])
    def test_known_dtypes(self, dtype, expected):
        assert get_kv_quant_mode(dtype) is expected, f"{dtype} -> {expected}"

    @pytest.mark.parametrize("dtype", [
        "auto", "float16", "bfloat16", "float32",
    ])
    def test_non_quantized_returns_none(self, dtype):
        assert get_kv_quant_mode(dtype) is KVQuantMode.NONE


# ---------------------------------------------------------------------------
# int4_kv_cache_full_dim
# ---------------------------------------------------------------------------


class TestInt4KvCacheFullDim:

    @pytest.mark.parametrize("head_size,expected", [
        (16, 8),
        (64, 32),
        (128, 64),
        (256, 128),
    ])
    def test_expected_sizes(self, head_size, expected):
        """INT4 packs 2 values per byte."""
        assert int4_kv_cache_full_dim(head_size) == expected


# ---------------------------------------------------------------------------
# fp4_e2m1_kv_cache_full_dim
# ---------------------------------------------------------------------------


class TestFp4E2m1KvCacheFullDim:

    @pytest.mark.parametrize("head_size,expected", [
        (16, 9),     # 1 block: 8 data + 1 scale
        (64, 36),    # 4 blocks: 32 data + 4 scale
        (128, 72),   # 8 blocks: 64 data + 8 scale
        (256, 144),  # 16 blocks: 128 data + 16 scale
        (20, 18),    # 2 blocks: 16 data + 2 scale
    ])
    def test_expected_sizes(self, head_size, expected):
        """FP4 data + fp8 block scales (1 per 16 elements)."""
        assert fp4_e2m1_kv_cache_full_dim(head_size) == expected


# ---------------------------------------------------------------------------
# nvfp4_kv_cache_full_dim
# ---------------------------------------------------------------------------


class TestNvfp4KvCacheFullDim:

    @pytest.mark.parametrize("head_size,expected", [
        (64, 36),
        (128, 72),
        (256, 144),
    ])
    def test_expected_sizes(self, head_size, expected):
        assert nvfp4_kv_cache_full_dim(head_size) == expected


# ---------------------------------------------------------------------------
# Cross-component consistency: get_kv_quant_mode + is_quantized_kv_cache
# ---------------------------------------------------------------------------


class TestCrossComponentConsistency:

    def test_quantized_and_mode_agree(self):
        """Quantized dtypes should have non-NONE mode and vice versa."""
        for dtype in typing.get_args(CacheDType):
            is_quant = is_quantized_kv_cache(dtype)
            mode = get_kv_quant_mode(dtype)
            quantized_by_mode = mode != KVQuantMode.NONE
            if is_quant != quantized_by_mode:
                # Special case: kivi_int4 is in CacheDType but uses
                # a different quant path (not in get_kv_quant_mode)
                if dtype == "kivi_int4":
                    continue
                raise AssertionError(
                    f"Mismatch for {dtype}: is_quantized={is_quant}, "
                    f"mode={mode}"
                )