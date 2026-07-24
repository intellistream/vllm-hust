# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the new KV cache dtype extensions in vllm-hust.

Tests cover int4 and fp4_e2m1 KV cache dtype support across the dtype
mapping, quantization detection, and interface layers.
"""

import typing

import pytest
import torch

from vllm.config.cache import CacheDType
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    fp4_e2m1_kv_cache_full_dim,
    int4_kv_cache_full_dim,
    is_quantized_kv_cache,
)
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode


# ---------------------------------------------------------------------------
# CacheDType
# ---------------------------------------------------------------------------


class TestCacheDTypeIncludesNewTypes:

    def test_includes_int4(self):
        """Verify that CacheDType includes 'int4'."""
        assert "int4" in typing.get_args(CacheDType)

    def test_includes_fp4_e2m1(self):
        """Verify that CacheDType includes 'fp4_e2m1'."""
        assert "fp4_e2m1" in typing.get_args(CacheDType)


# ---------------------------------------------------------------------------
# STR_DTYPE_TO_TORCH_DTYPE
# ---------------------------------------------------------------------------


class TestStrDtypeToTorchDtype:

    def test_int4_maps_to_uint8(self):
        assert STR_DTYPE_TO_TORCH_DTYPE["int4"] == torch.uint8

    def test_fp4_e2m1_maps_to_uint8(self):
        assert STR_DTYPE_TO_TORCH_DTYPE["fp4_e2m1"] == torch.uint8


# ---------------------------------------------------------------------------
# is_quantized_kv_cache
# ---------------------------------------------------------------------------


class TestIsQuantizedKvCache:

    @pytest.mark.parametrize(
        "dtype",
        [
            "int4",
            "fp4_e2m1",
            "nvfp4",
        ],
    )
    def test_quantized_dtypes_return_true(self, dtype):
        assert is_quantized_kv_cache(dtype) is True

    @pytest.mark.parametrize(
        "dtype",
        [
            "auto",
            "float16",
            "bfloat16",
            "float32",
        ],
    )
    def test_non_quantized_dtypes_return_false(self, dtype):
        assert is_quantized_kv_cache(dtype) is False


# ---------------------------------------------------------------------------
# KVQuantMode
# ---------------------------------------------------------------------------


class TestKVQuantMode:

    def test_int4_exists(self):
        assert hasattr(KVQuantMode, "INT4")
        assert isinstance(KVQuantMode.INT4, KVQuantMode)

    def test_fp4_e2m1_exists(self):
        assert hasattr(KVQuantMode, "FP4_E2M1")
        assert isinstance(KVQuantMode.FP4_E2M1, KVQuantMode)

    def test_int4_value(self):
        assert KVQuantMode.INT4.value == 6

    def test_fp4_e2m1_value(self):
        assert KVQuantMode.FP4_E2M1.value == 7


# ---------------------------------------------------------------------------
# get_kv_quant_mode
# ---------------------------------------------------------------------------


class TestGetKvQuantMode:

    def test_int4_returns_int4_mode(self):
        assert get_kv_quant_mode("int4") is KVQuantMode.INT4

    def test_fp4_e2m1_returns_fp4_e2m1_mode(self):
        assert get_kv_quant_mode("fp4_e2m1") is KVQuantMode.FP4_E2M1

    def test_auto_returns_none(self):
        assert get_kv_quant_mode("auto") is KVQuantMode.NONE

    def test_float16_returns_none(self):
        assert get_kv_quant_mode("float16") is KVQuantMode.NONE

    def test_bfloat16_returns_none(self):
        assert get_kv_quant_mode("bfloat16") is KVQuantMode.NONE

    def test_nvfp4_returns_nvfp4_mode(self):
        assert get_kv_quant_mode("nvfp4") is KVQuantMode.NVFP4


# ---------------------------------------------------------------------------
# Helper functions: int4_kv_cache_full_dim / fp4_e2m1_kv_cache_full_dim
# ---------------------------------------------------------------------------


class TestInt4KvCacheFullDim:

    def test_head_size_128(self):
        """INT4 packs 2 values per byte, so 128 elements -> 64 bytes."""
        assert int4_kv_cache_full_dim(128) == 64

    def test_head_size_64(self):
        assert int4_kv_cache_full_dim(64) == 32

    def test_head_size_256(self):
        assert int4_kv_cache_full_dim(256) == 128


class TestFp4E2m1KvCacheFullDim:

    def test_head_size_128(self):
        """FP4 data + fp8 block scales (1 per 16 elements).

        For head_size=128:
          data  = 128 // 2 = 64
          scale = 128 // 16 = 8
          total = 72
        """
        assert fp4_e2m1_kv_cache_full_dim(128) == 72

    def test_head_size_16(self):
        """Small head_size: 16 // 2 + 16 // 16 = 8 + 1 = 9."""
        assert fp4_e2m1_kv_cache_full_dim(16) == 9

    def test_head_size_64(self):
        # 64 elements: 4 blocks of 16 -> 4*8=32 data + 4*1=4 scale = 36
        assert fp4_e2m1_kv_cache_full_dim(64) == 36

    def test_head_size_256(self):
        assert fp4_e2m1_kv_cache_full_dim(256) == 144