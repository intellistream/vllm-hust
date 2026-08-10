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


# ---------------------------------------------------------------------------
# int8 / kivi_int4 consistency (reviewer point: config face internal
# consistency)
# ---------------------------------------------------------------------------


class TestInt8KiviInt4Consistency:
    def test_int8_is_quantized(self):
        assert is_quantized_kv_cache("int8") is True

    def test_kivi_int4_is_quantized(self):
        assert is_quantized_kv_cache("kivi_int4") is True

    def test_int8_str_dtype_mapping(self):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        assert STR_DTYPE_TO_TORCH_DTYPE["int8"] == torch.int8

    def test_kivi_int4_str_dtype_mapping(self):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        assert STR_DTYPE_TO_TORCH_DTYPE["kivi_int4"] == torch.uint8

    def test_int8_get_kv_quant_mode(self):
        assert get_kv_quant_mode("int8") is KVQuantMode.INT8_PER_TENSOR

    def test_kivi_int4_get_kv_quant_mode(self):
        assert get_kv_quant_mode("kivi_int4") is KVQuantMode.KIVI_INT4

    def test_int8_in_cache_dtype(self):
        assert "int8" in typing.get_args(CacheDType)

    def test_kivi_int4_in_cache_dtype(self):
        assert "kivi_int4" in typing.get_args(CacheDType)


# ---------------------------------------------------------------------------
# KVQuantMode new properties
# ---------------------------------------------------------------------------


class TestKVQuantModeProperties:
    def test_is_int4(self):
        assert KVQuantMode.INT4.is_int4 is True
        assert KVQuantMode.NONE.is_int4 is False

    def test_is_fp4_e2m1(self):
        assert KVQuantMode.FP4_E2M1.is_fp4_e2m1 is True
        assert KVQuantMode.NONE.is_fp4_e2m1 is False

    def test_is_int8_per_tensor(self):
        assert KVQuantMode.INT8_PER_TENSOR.is_int8_per_tensor is True
        assert KVQuantMode.NONE.is_int8_per_tensor is False

    def test_is_kivi_int4(self):
        assert KVQuantMode.KIVI_INT4.is_kivi_int4 is True
        assert KVQuantMode.NONE.is_kivi_int4 is False

    def test_is_packed_4bit(self):
        assert KVQuantMode.INT4.is_packed_4bit is True
        assert KVQuantMode.FP4_E2M1.is_packed_4bit is True
        assert KVQuantMode.NVFP4.is_packed_4bit is True
        assert KVQuantMode.NONE.is_packed_4bit is False

    def test_int8_per_tensor_value(self):
        assert KVQuantMode.INT8_PER_TENSOR.value == 8

    def test_kivi_int4_value(self):
        assert KVQuantMode.KIVI_INT4.value == 9


# ---------------------------------------------------------------------------
# Cache sizing: real_page_size_bytes for INT4 / FP4_E2M1
# ---------------------------------------------------------------------------


class TestAttentionSpecCacheSizing:
    """Verify that INT4 and FP4_E2M1 produce correct packed page sizes."""

    @pytest.fixture
    def base_kwargs(self):
        return dict(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )

    def test_int4_page_size_smaller_than_none(self, base_kwargs):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec_none = AttentionSpec(kv_quant_mode=KVQuantMode.NONE, **base_kwargs)
        spec_int4 = AttentionSpec(kv_quant_mode=KVQuantMode.INT4, **base_kwargs)
        assert spec_int4.real_page_size_bytes < spec_none.real_page_size_bytes

    def test_fp4_e2m1_page_size_smaller_than_none(self, base_kwargs):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec_none = AttentionSpec(kv_quant_mode=KVQuantMode.NONE, **base_kwargs)
        spec_fp4 = AttentionSpec(kv_quant_mode=KVQuantMode.FP4_E2M1, **base_kwargs)
        assert spec_fp4.real_page_size_bytes < spec_none.real_page_size_bytes

    def test_int4_page_size_exact(self, base_kwargs):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(kv_quant_mode=KVQuantMode.INT4, **base_kwargs)
        # head_size=128, int4 packs 2/byte -> packed dim = 64
        # page = 2 * block_size * num_kv_heads * 64 * 1
        expected = 2 * 16 * 8 * 64 * 1
        assert spec.real_page_size_bytes == expected

    def test_fp4_e2m1_page_size_exact(self, base_kwargs):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(kv_quant_mode=KVQuantMode.FP4_E2M1, **base_kwargs)
        # head_size=128, fp4_e2m1 dim = 128//2 + 128//16 = 64+8 = 72
        # page = 2 * block_size * num_kv_heads * 72 * 1
        expected = 2 * 16 * 8 * 72 * 1
        assert spec.real_page_size_bytes == expected


class TestFullAttentionSpecCacheSizing:
    @pytest.fixture
    def base_kwargs(self):
        return dict(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
        )

    def test_int4_full_attn_page_size(self, base_kwargs):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = FullAttentionSpec(kv_quant_mode=KVQuantMode.INT4, **base_kwargs)
        # K: int4(128//2=64) + V: int4(128//2=64) = 128
        expected = 16 * 8 * 128 * 1
        assert spec.real_page_size_bytes == expected

    def test_fp4_e2m1_full_attn_page_size(self, base_kwargs):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = FullAttentionSpec(kv_quant_mode=KVQuantMode.FP4_E2M1, **base_kwargs)
        # K: fp4_e2m1(128)=72 + V: fp4_e2m1(128)=72 = 144
        expected = 16 * 8 * 144 * 1
        assert spec.real_page_size_bytes == expected


# ---------------------------------------------------------------------------
# Backend capability fail-closed
# ---------------------------------------------------------------------------


class TestBackendCapabilityFailClosed:
    def test_int4_odd_head_size_raises(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        with pytest.raises(ValueError, match="even head_size"):
            AttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=127,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.INT4,
            )

    def test_fp4_e2m1_non_div16_head_size_raises(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        with pytest.raises(ValueError, match="divisible by 16"):
            AttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=100,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.FP4_E2M1,
            )

    def test_kivi_int4_odd_head_size_raises(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        with pytest.raises(ValueError, match="even head_size"):
            AttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=127,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.KIVI_INT4,
            )

    def test_none_mode_does_not_raise(self):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=127,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NONE,
        )
        assert spec.real_page_size_bytes > 0


# ---------------------------------------------------------------------------
# KiVi INT4 cache sizing (asymmetric K/V — no 2x double-count)
# ---------------------------------------------------------------------------


class TestAttentionSpecKiViCacheSizing:
    """Verify KiVi produces correct page sizes without the 2x double-count bug."""

    @pytest.fixture
    def base_kwargs(self):
        return dict(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )

    def test_kivi_int4_page_size_exact(self, base_kwargs):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(kv_quant_mode=KVQuantMode.KIVI_INT4, **base_kwargs)
        # K: int4 (128//2=64) + V: int8 (128) = 192 per head, no 2x multiplier
        expected = 16 * 8 * (64 + 128) * 1
        assert spec.real_page_size_bytes == expected

    def test_kivi_int4_no_double_count(self, base_kwargs):
        """KiVi has asymmetric K/V; the 2x K+V multiplier must not apply."""
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec = AttentionSpec(kv_quant_mode=KVQuantMode.KIVI_INT4, **base_kwargs)
        # Correct: 16 * 8 * 192 = 24576
        # Buggy (2x): 2 * 16 * 8 * 192 = 49152
        assert spec.real_page_size_bytes == 24576

    def test_kivi_int4_smaller_than_none(self, base_kwargs):
        from vllm.v1.kv_cache_interface import AttentionSpec

        spec_none = AttentionSpec(kv_quant_mode=KVQuantMode.NONE, **base_kwargs)
        spec_kivi = AttentionSpec(kv_quant_mode=KVQuantMode.KIVI_INT4, **base_kwargs)
        assert spec_kivi.real_page_size_bytes < spec_none.real_page_size_bytes


class TestFullAttentionSpecKiViCacheSizing:
    @pytest.fixture
    def base_kwargs(self):
        return dict(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
        )

    def test_kivi_int4_full_attn_page_size(self, base_kwargs):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = FullAttentionSpec(kv_quant_mode=KVQuantMode.KIVI_INT4, **base_kwargs)
        # K: int4(64) + V: int8(128) = 192
        expected = 16 * 8 * 192 * 1
        assert spec.real_page_size_bytes == expected


# ---------------------------------------------------------------------------
# FullAttentionSpec fail-closed (verifies __post_init__ chain is not broken)
# ---------------------------------------------------------------------------


class TestFullAttentionSpecFailClosed:
    """Verify FullAttentionSpec inherits fail-closed validation via super()."""

    def test_int4_odd_head_size_raises(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        with pytest.raises(ValueError, match="even head_size"):
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=127,
                head_size_v=127,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.INT4,
            )

    def test_fp4_e2m1_non_div16_head_size_raises(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        with pytest.raises(ValueError, match="divisible by 16"):
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=100,
                head_size_v=100,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.FP4_E2M1,
            )

    def test_kivi_int4_odd_head_size_raises(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        with pytest.raises(ValueError, match="even head_size"):
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=127,
                head_size_v=127,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.KIVI_INT4,
            )

    def test_none_mode_does_not_raise(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=127,
            head_size_v=127,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NONE,
        )
        assert spec.real_page_size_bytes > 0
