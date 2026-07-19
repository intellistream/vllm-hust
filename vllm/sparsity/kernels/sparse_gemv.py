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
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SPARSE_GEMV_INVOCATIONS = 0
_SPARSE_GEMV_MARKED = False
_SPARSE_GEMV_MARKER_RECORDS = 0
_ASCEND_SPARSE_LINEAR_MARKED = False
_ASCEND_SPARSE_LINEAR_MARKER_RECORDS = 0
_ASCEND_SPARSE_LINEAR_IMPORT_ATTEMPTED = False
_ASCEND_SPARSE_LINEAR = None
_ASCEND_DIRECT_T_OP = None
_ASCEND_SILU_AND_MUL_DIRECT_T_OP = None
_ASCEND_PACK_OP = None
_ASCEND_SILU_AND_MUL_PACKED_T_OP = None
_ASCEND_CUSTOM_OP_IMPORT_ATTEMPTED = False
_ASCEND_CUSTOM_OP_ENABLED = False


def _requires_backend_kernel() -> bool:
    value = os.environ.get("VLLM_SPARSE_GEMV_REQUIRE_KERNEL", "")
    return value.lower() in {"1", "true", "yes", "on"}


def reset_sparse_gemv_invocation_count() -> None:
    global _SPARSE_GEMV_INVOCATIONS, _SPARSE_GEMV_MARKED
    global _SPARSE_GEMV_MARKER_RECORDS
    global _ASCEND_SPARSE_LINEAR_MARKED, _ASCEND_SPARSE_LINEAR_MARKER_RECORDS
    _SPARSE_GEMV_INVOCATIONS = 0
    _SPARSE_GEMV_MARKED = False
    _SPARSE_GEMV_MARKER_RECORDS = 0
    _ASCEND_SPARSE_LINEAR_MARKED = False
    _ASCEND_SPARSE_LINEAR_MARKER_RECORDS = 0


def get_sparse_gemv_invocation_count() -> int:
    return _SPARSE_GEMV_INVOCATIONS


def _get_ascend_sparse_linear():
    global _ASCEND_SPARSE_LINEAR_IMPORT_ATTEMPTED, _ASCEND_SPARSE_LINEAR
    if not _ASCEND_SPARSE_LINEAR_IMPORT_ATTEMPTED:
        _ASCEND_SPARSE_LINEAR_IMPORT_ATTEMPTED = True
        try:
            from vllm_ascend.ops.sparse_linear import activation_sparse_linear
        except (ImportError, ModuleNotFoundError):
            _ASCEND_SPARSE_LINEAR = None
        else:
            _ASCEND_SPARSE_LINEAR = activation_sparse_linear
    return _ASCEND_SPARSE_LINEAR


def _ensure_ascend_custom_ops_registered() -> bool:
    global _ASCEND_CUSTOM_OP_IMPORT_ATTEMPTED, _ASCEND_CUSTOM_OP_ENABLED
    if not _ASCEND_CUSTOM_OP_IMPORT_ATTEMPTED:
        _ASCEND_CUSTOM_OP_IMPORT_ATTEMPTED = True
        try:
            from vllm_ascend.utils import enable_custom_op
        except (ImportError, ModuleNotFoundError):
            _ASCEND_CUSTOM_OP_ENABLED = False
        else:
            _ASCEND_CUSTOM_OP_ENABLED = bool(enable_custom_op())
    return _ASCEND_CUSTOM_OP_ENABLED


def _get_ascend_direct_t_op():
    global _ASCEND_DIRECT_T_OP
    if _ASCEND_DIRECT_T_OP is None:
        if not _ensure_ascend_custom_ops_registered():
            return None
        try:
            _ASCEND_DIRECT_T_OP = torch.ops._C_ascend.activation_sparse_linear_direct_t
        except (AttributeError, RuntimeError):
            return None
    return _ASCEND_DIRECT_T_OP


def _get_ascend_silu_and_mul_direct_t_op():
    global _ASCEND_SILU_AND_MUL_DIRECT_T_OP
    if _ASCEND_SILU_AND_MUL_DIRECT_T_OP is None:
        if not _ensure_ascend_custom_ops_registered():
            return None
        try:
            _ASCEND_SILU_AND_MUL_DIRECT_T_OP = (
                torch.ops._C_ascend.activation_sparse_silu_and_mul_direct_t
            )
        except (AttributeError, RuntimeError):
            return None
    return _ASCEND_SILU_AND_MUL_DIRECT_T_OP


def _get_ascend_pack_op():
    global _ASCEND_PACK_OP
    if _ASCEND_PACK_OP is None:
        if not _ensure_ascend_custom_ops_registered():
            return None
        try:
            _ASCEND_PACK_OP = torch.ops._C_ascend.activation_sparse_pack
        except (AttributeError, RuntimeError):
            return None
    return _ASCEND_PACK_OP


def _get_ascend_silu_and_mul_packed_t_op():
    global _ASCEND_SILU_AND_MUL_PACKED_T_OP
    if _ASCEND_SILU_AND_MUL_PACKED_T_OP is None:
        if not _ensure_ascend_custom_ops_registered():
            return None
        try:
            _ASCEND_SILU_AND_MUL_PACKED_T_OP = (
                torch.ops._C_ascend.activation_sparse_silu_and_mul_packed_t
            )
        except (AttributeError, RuntimeError):
            return None
    return _ASCEND_SILU_AND_MUL_PACKED_T_OP


def _should_use_packed_silu_and_mul_t() -> bool:
    mode = os.environ.get("VLLM_ASCEND_SPARSE_LINEAR_IMPL", "auto").lower()
    return mode in {"packed", "packed_t", "silu_packed", "silu_packed_t"}


def should_use_topk_matmul_silu() -> bool:
    mode = os.environ.get("VLLM_ASCEND_SPARSE_LINEAR_IMPL", "auto").lower()
    return mode in {"topk_matmul", "topk-matmul", "larosa_topk_matmul"}


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


def _marker_metadata_payload(
    marker_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if not marker_metadata:
        return {}
    payload: dict[str, Any] = {}
    for key, value in marker_metadata.items():
        key = str(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def _marker_record_limit() -> int:
    value = os.environ.get("VLLM_SPARSE_GEMV_MARKER_LIMIT")
    if value is None:
        return 1
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def _is_torch_compiling() -> bool:
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if is_compiling is None:
        return False
    try:
        return bool(is_compiling())
    except Exception:
        return False


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    if getattr(tensor, "is_npu", False):
        return True
    if not _is_torch_compiling():
        return False
    return getattr(getattr(tensor, "device", None), "type", None) == "npu"


def _same_supported_sparse_dtype(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    return x.dtype in (torch.float16, torch.bfloat16) and weight.dtype == x.dtype


def _threshold_numel_is_supported(
    threshold: torch.Tensor,
    x: torch.Tensor,
) -> bool:
    if threshold.numel() == 1:
        return True
    if _is_torch_compiling():
        return True
    return threshold.numel() == x.shape[0]


def _record_sparse_gemv_invocation(
    *,
    x: torch.Tensor | None = None,
    threshold: torch.Tensor | None = None,
    inclusive: bool | None = None,
    weight_t: torch.Tensor | None = None,
    marker_metadata: dict[str, Any] | None = None,
) -> None:
    global _SPARSE_GEMV_INVOCATIONS, _SPARSE_GEMV_MARKED
    global _SPARSE_GEMV_MARKER_RECORDS
    _SPARSE_GEMV_INVOCATIONS += 1

    if _is_torch_compiling():
        return

    marker_limit = _marker_record_limit()
    if marker_limit <= _SPARSE_GEMV_MARKER_RECORDS:
        _SPARSE_GEMV_MARKED = marker_limit > 0
        return
    marker_path = os.environ.get("VLLM_SPARSE_GEMV_MARKER_PATH")
    if not marker_path:
        return
    try:
        marker_dir = os.path.dirname(marker_path)
        if marker_dir:
            os.makedirs(marker_dir, exist_ok=True)
        payload: dict[str, Any] = {
            "pid": os.getpid(),
            "invocations": _SPARSE_GEMV_INVOCATIONS,
            "marker_record": _SPARSE_GEMV_MARKER_RECORDS + 1,
        }
        payload.update(_marker_metadata_payload(marker_metadata))
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
        _SPARSE_GEMV_MARKER_RECORDS += 1
        _SPARSE_GEMV_MARKED = marker_limit <= _SPARSE_GEMV_MARKER_RECORDS
    except OSError:
        logger.warning("Failed to write sparse GEMV marker to %s", marker_path)


def _record_ascend_sparse_linear_marker(
    *,
    x: torch.Tensor,
    threshold: torch.Tensor,
    inclusive: bool,
    weight_t: torch.Tensor,
    op_name: str = "activation_sparse_linear_direct_t",
    marker_metadata: dict[str, Any] | None = None,
) -> None:
    global _ASCEND_SPARSE_LINEAR_MARKED, _ASCEND_SPARSE_LINEAR_MARKER_RECORDS
    if _is_torch_compiling():
        return

    marker_limit = _marker_record_limit()
    if marker_limit <= _ASCEND_SPARSE_LINEAR_MARKER_RECORDS:
        _ASCEND_SPARSE_LINEAR_MARKED = marker_limit > 0
        return

    marker_path = os.environ.get("VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH")
    if not marker_path:
        return
    try:
        marker_dir = os.path.dirname(marker_path)
        if marker_dir:
            os.makedirs(marker_dir, exist_ok=True)
        payload: dict[str, Any] = {
            "op": op_name,
            "pid": os.getpid(),
            "inclusive": bool(inclusive),
            "marker_record": _ASCEND_SPARSE_LINEAR_MARKER_RECORDS + 1,
        }
        payload.update(_marker_metadata_payload(marker_metadata))
        payload.update(_tensor_marker_payload("x", x))
        payload.update(_tensor_marker_payload("threshold", threshold))
        payload.update(_tensor_marker_payload("weight_t", weight_t))
        with open(marker_path, "a", encoding="utf-8") as marker_file:
            marker_file.write(json.dumps(payload, sort_keys=True) + "\n")
        _ASCEND_SPARSE_LINEAR_MARKER_RECORDS += 1
        _ASCEND_SPARSE_LINEAR_MARKED = (
            marker_limit <= _ASCEND_SPARSE_LINEAR_MARKER_RECORDS
        )
    except OSError:
        return


def _try_ascend_direct_t_fast_path(
    x: torch.Tensor,
    weight_t: torch.Tensor | None,
    threshold: torch.Tensor,
    inclusive: bool,
    marker_metadata: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    if weight_t is None:
        return None
    if not _is_npu_tensor(x):
        return None
    if not _same_supported_sparse_dtype(x, weight_t):
        return None
    if x.dim() != 2 or weight_t.dim() != 2:
        return None
    if x.shape[1] != weight_t.shape[0]:
        return None
    if not _threshold_numel_is_supported(threshold, x):
        return None

    op = _get_ascend_direct_t_op()
    if op is None:
        return None

    if threshold.dtype != torch.float32 or threshold.device != x.device:
        threshold = threshold.to(dtype=torch.float32, device=x.device)
    if not threshold.is_contiguous():
        threshold = threshold.contiguous()
    _record_ascend_sparse_linear_marker(
        x=x,
        threshold=threshold,
        inclusive=inclusive,
        weight_t=weight_t,
        marker_metadata=marker_metadata,
    )
    return op(
        x if x.is_contiguous() else x.contiguous(),
        weight_t if weight_t.is_contiguous() else weight_t.contiguous(),
        threshold,
        inclusive,
    )


def _try_ascend_silu_and_mul_direct_t_fast_path(
    x: torch.Tensor,
    weight_t: torch.Tensor | None,
    threshold: torch.Tensor,
    inclusive: bool,
    marker_metadata: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    if weight_t is None:
        return None
    if not _is_npu_tensor(x):
        return None
    if not _same_supported_sparse_dtype(x, weight_t):
        return None
    if x.dim() != 2 or weight_t.dim() != 2:
        return None
    if x.shape[1] != weight_t.shape[0] or weight_t.shape[1] % 2 != 0:
        return None
    if not _threshold_numel_is_supported(threshold, x):
        return None

    op = _get_ascend_silu_and_mul_direct_t_op()
    if op is None:
        return None

    if threshold.dtype != torch.float32 or threshold.device != x.device:
        threshold = threshold.to(dtype=torch.float32, device=x.device)
    if not threshold.is_contiguous():
        threshold = threshold.contiguous()
    if _should_use_packed_silu_and_mul_t():
        pack_op = _get_ascend_pack_op()
        packed_op = _get_ascend_silu_and_mul_packed_t_op()
        if pack_op is not None and packed_op is not None:
            _record_ascend_sparse_linear_marker(
                x=x,
                threshold=threshold,
                inclusive=inclusive,
                weight_t=weight_t,
                op_name="activation_sparse_silu_and_mul_packed_t",
                marker_metadata=marker_metadata,
            )
            values, indices, counts = pack_op(
                x if x.is_contiguous() else x.contiguous(),
                threshold,
                inclusive,
            )
            return packed_op(
                values,
                indices,
                counts,
                weight_t if weight_t.is_contiguous() else weight_t.contiguous(),
            )

    _record_ascend_sparse_linear_marker(
        x=x,
        threshold=threshold,
        inclusive=inclusive,
        weight_t=weight_t,
        op_name="activation_sparse_silu_and_mul_direct_t",
        marker_metadata=marker_metadata,
    )
    return op(
        x if x.is_contiguous() else x.contiguous(),
        weight_t if weight_t.is_contiguous() else weight_t.contiguous(),
        threshold,
        inclusive,
    )


def sparse_gemv_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: torch.Tensor,
    sparsity_bin: int = 0,
    inclusive: bool = False,
    weight_t: torch.Tensor | None = None,
    marker_metadata: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply activation-threshold sparse GEMV with an AscendC fast path."""
    del sparsity_bin
    _record_sparse_gemv_invocation(
        x=x,
        threshold=threshold,
        inclusive=inclusive,
        weight_t=weight_t,
        marker_metadata=marker_metadata,
    )
    fast_output = _try_ascend_direct_t_fast_path(
        x,
        weight_t,
        threshold,
        inclusive,
        marker_metadata,
    )
    if fast_output is not None:
        return fast_output

    activation_sparse_linear = _get_ascend_sparse_linear()
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
    if threshold.dtype != torch.float32 or threshold.device != x.device:
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


def sparse_gemv_silu_and_mul_direct_t_cached_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_t: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
    marker_metadata: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """Apply fused sparse gate/up projection plus SiLU*up on Ascend."""
    del weight
    _record_sparse_gemv_invocation(
        x=x,
        threshold=threshold,
        inclusive=inclusive,
        weight_t=weight_t,
        marker_metadata=marker_metadata,
    )
    return _try_ascend_silu_and_mul_direct_t_fast_path(
        x,
        weight_t,
        threshold,
        inclusive,
        marker_metadata,
    )


def sparse_gemv_topk_matmul_silu_impl(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    keep: int,
    marker_metadata: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """Apply La RoSA top-k sparse gate/up via selected-weight matmul on NPU."""
    if not _is_npu_tensor(x):
        return None
    if not _same_supported_sparse_dtype(x, weight_t):
        return None
    if x.dim() != 2 or x.shape[0] != 1:
        return None
    if weight_t.dim() != 2 or x.shape[1] != weight_t.shape[0]:
        return None
    if weight_t.shape[1] % 2 != 0:
        return None
    if keep <= 0 or keep > x.shape[1]:
        return None

    magnitudes = x.abs().to(dtype=torch.float32)
    topk_values, topk_indices = torch.topk(magnitudes, keep, dim=-1)
    threshold = topk_values[..., -1].contiguous()
    _record_sparse_gemv_invocation(
        x=x,
        threshold=threshold,
        inclusive=True,
        weight_t=weight_t,
        marker_metadata=marker_metadata,
    )
    _record_ascend_sparse_linear_marker(
        x=x,
        threshold=threshold,
        inclusive=True,
        weight_t=weight_t,
        op_name="activation_sparse_topk_matmul_silu",
        marker_metadata=marker_metadata,
    )

    selected_indices = topk_indices[0].to(dtype=torch.long).contiguous()
    selected_x = x[:, selected_indices].contiguous()
    selected_weight_t = weight_t.index_select(0, selected_indices)
    gate_up = torch.matmul(selected_x, selected_weight_t)
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def sparse_gemv_direct_t_cached_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_t: torch.Tensor,
    threshold: torch.Tensor,
    *,
    inclusive: bool = False,
    marker_metadata: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """Fastest cached-plan path for already validated Ascend direct-T calls."""
    if not _is_npu_tensor(x):
        return None
    if not _same_supported_sparse_dtype(x, weight_t):
        return None
    if threshold.dtype != torch.float32 or threshold.device != x.device:
        return None
    if not x.is_contiguous() or not weight_t.is_contiguous():
        return None
    if not threshold.is_contiguous():
        return None
    op = _get_ascend_direct_t_op()
    if op is None:
        return None

    _record_sparse_gemv_invocation(
        x=x,
        threshold=threshold,
        inclusive=inclusive,
        weight_t=weight_t,
        marker_metadata=marker_metadata,
    )
    _record_ascend_sparse_linear_marker(
        x=x,
        threshold=threshold,
        inclusive=inclusive,
        weight_t=weight_t,
        marker_metadata=marker_metadata,
    )
    return op(x, weight_t, threshold, inclusive)


def can_use_sparse_gemv(
    tp_size: int,
    quant_config: Any | None,
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
    return dtype in (torch.float16, torch.bfloat16)
