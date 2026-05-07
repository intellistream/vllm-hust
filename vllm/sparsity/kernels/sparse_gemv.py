# SPDX-License-Identifier: Apache-2.0
"""Triton sparse GEMV kernel for TEAL (Phase 2 experimental).

This module is a placeholder for the TEAL Triton sparse GEMV kernel.
When ``use_sparse_gemv=True`` and all constraints are satisfied:
  - tp_size == 1
  - quant_config is None
  - LoRA is disabled
  - dtype in (float16, bfloat16)
  - decode-only batch

a custom op registered via ``direct_register_custom_op`` will be called.
"""

from typing import Any, Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def sparse_gemv_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: torch.Tensor,
    sparsity_bin: int = 0,
) -> torch.Tensor:
    """Placeholder implementation: falls back to dense GEMV.

    The real Triton kernel will be added in Phase 2.
    """
    logger.warning_once(
        "sparse_gemv_impl is a placeholder; using dense fallback. "
        "Set use_sparse_gemv=False to silence this warning."
    )
    return torch.matmul(x, weight.t())


def can_use_sparse_gemv(
    tp_size: int,
    quant_config: Optional[Any],
    dtype: torch.dtype,
    use_sparse_gemv_flag: bool = False,
) -> bool:
    """Return True if the strict constraints for sparse GEMV are met."""
    if not use_sparse_gemv_flag:
        return False
    if tp_size != 1:
        return False
    if quant_config is not None:
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    return True
