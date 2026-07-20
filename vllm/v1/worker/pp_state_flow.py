# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded diagnostics for PP scheduler/model state flow."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any

import vllm.envs as envs

_write_lock = threading.Lock()
_event_seq = 0


def enabled() -> bool:
    """Return whether state-flow diagnostics are explicitly enabled."""
    return bool(
        envs.VLLM_DIAGNOSE_PP_STATE_FLOW
        and envs.VLLM_DIAGNOSE_PP_STATE_FLOW_OUTPUT_PATH
    )


def _output_path() -> Path:
    base = Path(str(envs.VLLM_DIAGNOSE_PP_STATE_FLOW_OUTPUT_PATH))
    suffix = base.suffix or ".jsonl"
    return base.with_name(f"{base.stem}.pid-{os.getpid()}{suffix}")


def emit(event: str, **payload: Any) -> None:
    """Append one JSON event to a process-local trace file."""
    if not enabled():
        return

    global _event_seq
    with _write_lock:
        _event_seq += 1
        record = {
            "schema": "vllm.pp_state_flow.v1",
            "event": event,
            "event_seq": _event_seq,
            "pid": os.getpid(),
            "monotonic_ns": time.monotonic_ns(),
            **payload,
        }
        path = _output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def tensor_summary(tensor: Any, *, include_values_limit: int = 64) -> dict[str, Any]:
    """Return a bounded byte checksum and optional small values for a tensor."""
    import torch

    if tensor is None:
        return {"is_none": True}
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor)!r}")

    detached = tensor.detach().contiguous()
    cpu = detached.to(device="cpu")
    try:
        raw = cpu.view(torch.uint8).numpy().tobytes()
        checksum_encoding = "native-bytes"
    except (RuntimeError, TypeError):
        raw = cpu.to(torch.float32).numpy().tobytes()
        checksum_encoding = "float32-bytes"

    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": detached.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checksum_encoding": checksum_encoding,
    }
    if detached.numel() <= include_values_limit:
        result["values"] = cpu.reshape(-1).tolist()
    return result


def tensor_dict_summary(tensors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Summarize a PP tensor dictionary without retaining tensor contents."""
    return {name: tensor_summary(tensor) for name, tensor in sorted(tensors.items())}


def rank_info() -> dict[str, int]:
    """Return PP/TP ranks after distributed groups have initialized."""
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    return {
        "pp_rank": get_pp_group().rank_in_group,
        "tp_rank": get_tp_group().rank_in_group,
    }


def _block_tables(runner: Any) -> dict[str, list[dict[str, Any]]]:
    input_batch = runner.input_batch
    num_reqs = input_batch.num_reqs
    result: dict[str, list[dict[str, Any]]] = {}
    for group_id, block_table in enumerate(input_batch.block_table.block_tables):
        table = block_table.get_numpy_array()
        rows = []
        for request_index in range(num_reqs):
            count = int(block_table.num_blocks_per_row[request_index])
            rows.append(
                {
                    "request_index": request_index,
                    "block_ids": table[request_index, :count].tolist(),
                }
            )
        result[str(group_id)] = rows
    return result


def _slot_mappings(runner: Any, num_tokens: int) -> dict[str, dict[str, Any]]:
    return {
        str(group_id): tensor_summary(block_table.slot_mapping.gpu[:num_tokens])
        for group_id, block_table in enumerate(
            runner.input_batch.block_table.block_tables
        )
    }


def _logits_payload(logits: Any) -> dict[str, Any]:
    import torch

    logits_float = logits.detach().to(torch.float32)
    top_values, top_ids = torch.topk(logits_float, k=5, dim=-1)
    margins = top_values[:, 0] - top_values[:, 1]
    return {
        "logits": tensor_summary(logits),
        "top5_token_ids": top_ids.to(device="cpu").tolist(),
        "top5_values": top_values.to(device="cpu").tolist(),
        "top1_top2_margin": margins.to(device="cpu").tolist(),
    }


def _sampled_payload(sampled: Any) -> Any:
    import torch

    if isinstance(sampled, torch.Tensor):
        return tensor_summary(sampled)
    return sampled


def install_model_runner_wrappers(runner: Any) -> None:
    """Wrap the effective runner methods, including plugin overrides."""
    if not enabled() or getattr(runner, "_pp_state_flow_wrapped", False):
        return
    runner._pp_state_flow_wrapped = True
    runner._pp_state_flow_step = 0
    runner._pp_state_flow_scheduler_output = None

    original_execute = runner.execute_model

    @wraps(original_execute)
    def execute_wrapper(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        runner._pp_state_flow_step += 1
        runner._pp_state_flow_scheduler_output = scheduler_output
        return original_execute(scheduler_output, *args, **kwargs)

    runner.execute_model = execute_wrapper

    original_forward = runner._model_forward
    forward_signature = inspect.signature(original_forward)

    @wraps(original_forward)
    def forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_arguments = forward_signature.bind_partial(*args, **kwargs).arguments
        scheduler_output = runner._pp_state_flow_scheduler_output
        num_tokens = (
            scheduler_output.total_num_scheduled_tokens
            if scheduler_output is not None
            else 0
        )
        input_ids = bound_arguments.get("input_ids")
        positions = bound_arguments.get("positions")
        intermediate_tensors = bound_arguments.get("intermediate_tensors")
        if intermediate_tensors is not None:
            wait_for_comm = getattr(intermediate_tensors, "wait_for_comm", None)
            if wait_for_comm is not None:
                wait_for_comm()
            emit(
                "pp_recv_complete",
                component="model_runner_wrapper",
                diagnostic_step=runner._pp_state_flow_step,
                microbatch_id=getattr(scheduler_output, "microbatch_id", -1),
                tensors=tensor_dict_summary(intermediate_tensors.tensors),
                **rank_info(),
            )
        emit(
            "attention_input_before_forward",
            component="model_runner_wrapper",
            diagnostic_step=runner._pp_state_flow_step,
            microbatch_id=getattr(scheduler_output, "microbatch_id", -1),
            request_ids=list(runner.input_batch.req_ids),
            num_scheduled_tokens=(
                dict(scheduler_output.num_scheduled_tokens)
                if scheduler_output is not None
                else {}
            ),
            block_tables=_block_tables(runner),
            input_ids=tensor_summary(input_ids),
            positions=tensor_summary(positions),
            slot_mappings=_slot_mappings(runner, num_tokens),
            **rank_info(),
        )
        output = original_forward(*args, **kwargs)
        tensors = getattr(output, "tensors", None)
        if tensors is not None:
            emit(
                "pp_send_start",
                component="model_runner_wrapper",
                diagnostic_step=runner._pp_state_flow_step,
                microbatch_id=getattr(scheduler_output, "microbatch_id", -1),
                tensors=tensor_dict_summary(tensors),
                **rank_info(),
            )
        return output

    runner._model_forward = forward_wrapper

    original_sample_tokens = runner.sample_tokens

    @wraps(original_sample_tokens)
    def sample_tokens_wrapper(*args: Any, **kwargs: Any) -> Any:
        state = runner.execute_model_state
        scheduler_output = runner._pp_state_flow_scheduler_output
        logits = getattr(state, "logits", None) if state is not None else None
        if logits is not None:
            emit(
                "logits_before_sampling",
                component="model_runner_wrapper",
                diagnostic_step=runner._pp_state_flow_step,
                microbatch_id=getattr(scheduler_output, "microbatch_id", -1),
                request_ids=list(runner.input_batch.req_ids),
                **_logits_payload(logits),
                **rank_info(),
            )
        output = original_sample_tokens(*args, **kwargs)
        sampled = getattr(output, "sampled_token_ids", None)
        if sampled is not None:
            emit(
                "sampled_tokens_synchronized",
                component="model_runner_wrapper",
                diagnostic_step=runner._pp_state_flow_step,
                microbatch_id=getattr(scheduler_output, "microbatch_id", -1),
                request_ids=list(runner.input_batch.req_ids),
                sampled_token_ids=_sampled_payload(sampled),
                **rank_info(),
            )
        return output

    runner.sample_tokens = sample_tokens_wrapper
