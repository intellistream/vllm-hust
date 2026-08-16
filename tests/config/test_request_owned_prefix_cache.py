# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.config.test_request_owned_attention import _vllm_config
from vllm.config import CacheConfig


def test_prefix_cache_and_bulk_offload_fail_closed_until_images_are_unified():
    with pytest.raises(ValueError, match="not yet compatible"):
        _vllm_config(
            enable_request_owned_kv_offload=True,
            cache_config=CacheConfig(
                kv_offloading_size=1.0,
                enable_prefix_caching=True,
            ),
        )
