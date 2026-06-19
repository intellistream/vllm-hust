# SPDX-License-Identifier: Apache-2.0
"""Backend sparse GEMV kernel for TEAL / La RoSA.

When ``use_sparse_gemv=True`` and all constraints are satisfied:
  - tp_size == 1
  - quant_config is None
  - LoRA is disabled
  - dtype in (float16, bfloat16)
  - all rows in the input batch are eligible for sparsification

the Ascend plugin's ``activation_sparse_linear`` custom op is called when
available. Other backends use a dense masked fallback for correctness.
"""

import json
import os
from typing import Any, Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SPARSE_GEMV_INVOCATIONS = 0
_SPARSE_GEMV_MARKED = False


def _requires_backend_kernel() -> bool:
    value = os.environ.get("VLLM_SPARSE_GEMV_REQUIRE_KERNEL", "")
    return value.lower() in {"1", "true", "yes", "on"}


def reset_sparse_gemv_invocation_count() -> None:
    global _SPARSE_GEMV_INVOCATIONS, _SPARSE_GEMV_MARKED
    _SPARSE_GEMV_INVOCATIONS = 0
    _SPARSE_GEMV_MARKED = False


def get_sparse_gemv_invocation_count() -> int:
    return _SPARSE_GEMV_INVOCATIONS


def _tensor_marker_payload(name: str, tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {f"{name}_provided": False}
    return {
        f"{name}_provided": True,
        f"{name}_shape": list(tensor.shape),
        f"{name}_dtype": str(tensor.dtype),
        f"{name}_device": str(tensor.device),
        f"{name}_numel": int(tensor.numel()),
    }


def _activity_marker_payload(
    x: torch.Tensor | None,
    threshold: torch.Tensor | None,
    inclusive: bool | None,
) -> dict[str, Any]:
    if x is None or threshold is None or inclusive is None:
        return {}
    if x.dim() == 0:
        return {"active_stats_error": "x must have at least one dimension"}
    if x.numel() == 0 or x.shape[-1] == 0:
        return {"active_row_count": 0, "active_hidden_size": int(x.shape[-1])}

    with torch.no_grad():
        threshold = threshold.detach().to(dtype=torch.float32, device=x.device)
        if threshold.numel() == 1:
            compare_threshold = threshold.reshape(1, 1)
        elif x.dim() == 2 and threshold.numel() == x.shape[0]:
            compare_threshold = threshold.reshape(x.shape[0], 1)
        else:
            return {
                "active_stats_error": (
                    "threshold must be scalar or have one value per input row"
                )
            }

        compare = torch.ge if inclusive else torch.gt
        active = compare(x.detach().abs().to(dtype=torch.float32), compare_threshold)
        hidden_size = int(x.shape[-1])
        active_counts = active.reshape(-1, hidden_size).sum(dim=1)
        active_counts_cpu = active_counts.to(dtype=torch.float32, device="cpu")
        densities = active_counts_cpu / float(hidden_size)

    return {
        "active_row_count": int(active_counts_cpu.numel()),
        "active_hidden_size": hidden_size,
        "active_count_min": int(active_counts_cpu.min().item()),
        "active_count_max": int(active_counts_cpu.max().item()),
        "active_count_mean": float(active_counts_cpu.mean().item()),
        "active_density_min": float(densities.min().item()),
        "active_density_max": float(densities.max().item()),
        "active_density_mean": float(densities.mean().item()),
        "active_sparsity_mean": float(1.0 - densities.mean().item()),
    }


def _record_sparse_gemv_invocation(
    *,
    x: torch.Tensor | None = None,
    threshold: torch.Tensor | None = None,
    inclusive: bool | None = None,
    weight_t: torch.Tensor | None = None,
) -> None:
    global _SPARSE_GEMV_INVOCATIONS, _SPARSE_GEMV_MARKED
    _SPARSE_GEMV_INVOCATIONS += 1

    marker_path = os.environ.get("VLLM_SPARSE_GEMV_MARKER_PATH")
    if not marker_path or _SPARSE_GEMV_MARKED:
        return
    _SPARSE_GEMV_MARKED = True
    try:
        marker_dir = os.path.dirname(marker_path)
        if marker_dir:
            os.makedirs(marker_dir, exist_ok=True)
        payload: dict[str, Any] = {
            "pid": os.getpid(),
            "invocations": _SPARSE_GEMV_INVOCATIONS,
        }
        payload.update(_tensor_marker_payload("x", x))
        payload.update(_tensor_marker_payload("threshold", threshold))
        payload.update(_tensor_marker_payload("weight_t", weight_t))
        try:
            payload.update(_activity_marker_payload(x, threshold, inclusive))
        except Exception as err:
            payload["active_stats_error"] = f"{type(err).__name__}: {err}"
        if inclusive is not None:
            payload["inclusive"] = bool(inclusive)
        with open(marker_path, "a", encoding="utf-8") as marker_file:
            marker_file.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        logger.warning("Failed to write sparse GEMV marker to %s", marker_path)


def sparse_gemv_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: torch.Tensor,
    sparsity_bin: int = 0,
    inclusive: bool = False,
    weight_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply activation-threshold sparse GEMV with an AscendC fast path."""
    del sparsity_bin
    _record_sparse_gemv_invocation(
        x=x,
        threshold=threshold,
        inclusive=inclusive,
        weight_t=weight_t,
    )
    try:
        from vllm_ascend.ops.sparse_linear import activation_sparse_linear
    except (ImportError, ModuleNotFoundError):
        activation_sparse_linear = None

    if activation_sparse_linear is not None:
        return activation_sparse_linear(
            x,
            weight,
            threshold,
            inclusive=inclusive,
            weight_t=weight_t,
        )

    if _requires_backend_kernel():
        raise RuntimeError(
            "use_sparse_gemv=True requires vllm_ascend.ops.sparse_linear."
            "activation_sparse_linear, but the backend op wrapper is not "
            "importable. Unset VLLM_SPARSE_GEMV_REQUIRE_KERNEL to allow the "
            "dense masked fallback."
        )

    logger.warning(
        "Ascend activation_sparse_linear is unavailable; using dense masked "
        "fallback. Set use_sparse_gemv=False to silence this warning."
    )
    threshold = threshold.to(dtype=torch.float32, device=x.device)
    if threshold.numel() == 1:
        threshold = threshold.reshape(1, 1)
    else:
        threshold = threshold.reshape(x.shape[0], 1)
    compare = torch.ge if inclusive else torch.gt
    sparse_x = torch.where(
        compare(x.abs().to(dtype=torch.float32), threshold),
        x,
        torch.zeros_like(x),
    )
    return torch.matmul(sparse_x, weight.t())


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
