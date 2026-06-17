# SPDX-License-Identifier: Apache-2.0
"""Compare dense vs activation-sparse vLLM generation throughput.

This benchmark is intentionally narrow: it uses deterministic token-id prompts
and toggles only ``activation_sparsity_config`` so the sparse run can be gated
against the dense run with the same request shape.
"""

import argparse
import gc
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["teal", "larosa"], required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, default=0.4)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--warmup-prompts", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--shutdown-timeout", type=int, default=60)
    parser.add_argument("--vllm-prefill-sparsify", default="none")
    parser.add_argument("--allow-sparse-gemv-fallback", action="store_true")
    parser.add_argument(
        "--no-require-sparse-invocations",
        action="store_true",
        help="Do not fail when the vLLM sparse GEMV shim is never invoked.",
    )
    parser.add_argument("--min-total-token-speedup", type=float, default=1.0)
    parser.add_argument("--min-output-token-speedup", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def synchronize() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_backend_cache() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_prompts(
    model: str,
    num_prompts: int,
    input_len: int,
    seed: int,
    trust_remote_code: bool,
) -> list[dict[str, list[int]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )
    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError(f"Tokenizer for {model!r} has invalid vocab_size={vocab_size}")

    excluded_ids = {
        token_id
        for token_id in tokenizer.all_special_ids
        if isinstance(token_id, int) and 0 <= token_id < vocab_size
    }
    candidate_ids = [i for i in range(vocab_size) if i not in excluded_ids]
    if not candidate_ids:
        raise ValueError(f"Tokenizer for {model!r} has no non-special token ids")

    rng = random.Random(seed)
    prompts: list[dict[str, list[int]]] = []
    for _ in range(num_prompts):
        prompt_token_ids = [rng.choice(candidate_ids) for _ in range(input_len)]
        prompts.append({"prompt_token_ids": prompt_token_ids})
    return prompts


def build_sparse_config(args: argparse.Namespace) -> dict[str, Any]:
    common: dict[str, Any] = {
        "enable": True,
        "method": args.method,
        "uniform_sparsity": args.sparsity,
        "calibration_path": str(args.calibration_path.resolve()),
        "decode_only": False,
        "strict_unsupported_check": True,
        "use_sparse_gemv": True,
    }
    if args.method == "teal":
        common.update(
            {
                "apply_all_tokens": False,
                "prefill_sparsify": args.vllm_prefill_sparsify,
            }
        )
    else:
        common.update(
            {
                "apply_all_tokens": args.vllm_prefill_sparsify == "all",
                "prefill_sparsify": args.vllm_prefill_sparsify,
            }
        )
    return common


def build_llm(
    args: argparse.Namespace,
    activation_sparsity_config: dict[str, Any] | None,
) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len or (args.input_len + args.output_len + 8),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "shutdown_timeout": args.shutdown_timeout,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if activation_sparsity_config is not None:
        kwargs["activation_sparsity_config"] = activation_sparsity_config
    return LLM(**kwargs)


def run_case(
    args: argparse.Namespace,
    prompts: list[dict[str, list[int]]],
    activation_sparsity_config: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    from vllm import SamplingParams

    llm = build_llm(args, activation_sparsity_config)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
        detokenize=False,
    )

    warmup_prompts = prompts[: args.warmup_prompts]
    if warmup_prompts:
        llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
        synchronize()

    synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    synchronize()
    elapsed = time.perf_counter() - start

    prompt_tokens = sum(len(output.prompt_token_ids or []) for output in outputs)
    output_tokens = sum(
        len(completion.token_ids)
        for output in outputs
        for completion in output.outputs
        if completion is not None
    )
    expected_output_tokens = len(prompts) * args.output_len
    total_tokens = prompt_tokens + output_tokens
    del llm
    gc.collect()
    empty_backend_cache()

    return {
        "label": label,
        "elapsed_s": elapsed,
        "num_prompts": len(prompts),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "expected_output_tokens": expected_output_tokens,
        "output_token_completion_ratio": output_tokens / expected_output_tokens,
        "total_tokens": total_tokens,
        "requests_per_second": len(prompts) / elapsed,
        "total_tokens_per_second": total_tokens / elapsed,
        "output_tokens_per_second": output_tokens / elapsed,
    }


def reset_sparse_invocation_counter() -> None:
    from vllm.sparsity.kernels.sparse_gemv import (
        reset_sparse_gemv_invocation_count,
    )

    reset_sparse_gemv_invocation_count()


def get_sparse_invocation_counter() -> int:
    from vllm.sparsity.kernels.sparse_gemv import (
        get_sparse_gemv_invocation_count,
    )

    return get_sparse_gemv_invocation_count()


def read_marker_records(marker_path: Path) -> list[dict[str, Any]]:
    if not marker_path.exists():
        return []
    records = []
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        records.append(json.loads(line))
    return records


def unique_marker_values(
    records: list[dict[str, Any]],
    key: str,
) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for record in records:
        if key not in record:
            continue
        value = record[key]
        marker = json.dumps(value, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        values.append(value)
    return values


def main() -> int:
    args = parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("--input-len and --output-len must be positive")
    if args.warmup_prompts < 0 or args.warmup_prompts > args.num_prompts:
        raise ValueError("--warmup-prompts must be in [0, num-prompts]")
    if args.sparsity < 0.0 or args.sparsity >= 1.0:
        raise ValueError("--sparsity must be in [0, 1)")
    if args.method == "larosa" and args.sparsity * 1.2 >= 1.0:
        raise ValueError("La RoSA h2 sparsity is sparsity * 1.2 and must be < 1")

    prompts = build_prompts(
        args.model,
        args.num_prompts,
        args.input_len,
        args.seed,
        args.trust_remote_code,
    )
    dense = run_case(args, prompts, None, "dense")

    previous_require_kernel = os.environ.get("VLLM_SPARSE_GEMV_REQUIRE_KERNEL")
    previous_marker_path = os.environ.get("VLLM_SPARSE_GEMV_MARKER_PATH")
    previous_ascend_marker_path = os.environ.get(
        "VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"
    )
    os.environ["VLLM_SPARSE_GEMV_REQUIRE_KERNEL"] = (
        "0" if args.allow_sparse_gemv_fallback else "1"
    )
    marker_tmpdir = tempfile.TemporaryDirectory(prefix="vllm_sparse_gemv_")
    marker_path = Path(marker_tmpdir.name) / "invocations.jsonl"
    ascend_marker_path = Path(marker_tmpdir.name) / "ascend_custom_ops.jsonl"
    os.environ["VLLM_SPARSE_GEMV_MARKER_PATH"] = str(marker_path)
    os.environ["VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"] = str(ascend_marker_path)
    reset_sparse_invocation_counter()
    try:
        sparse = run_case(args, prompts, build_sparse_config(args), "sparse")
    finally:
        if previous_require_kernel is None:
            os.environ.pop("VLLM_SPARSE_GEMV_REQUIRE_KERNEL", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_REQUIRE_KERNEL"] = previous_require_kernel
        if previous_marker_path is None:
            os.environ.pop("VLLM_SPARSE_GEMV_MARKER_PATH", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_MARKER_PATH"] = previous_marker_path
        if previous_ascend_marker_path is None:
            os.environ.pop("VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH", None)
        else:
            os.environ["VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"] = (
                previous_ascend_marker_path
            )

    sparse_invocations_current_process = get_sparse_invocation_counter()
    sparse_marker_records = read_marker_records(marker_path)
    ascend_marker_records = read_marker_records(ascend_marker_path)
    sparse_marker_pids = sorted(
        {
            int(record["pid"])
            for record in sparse_marker_records
            if "pid" in record
        }
    )
    ascend_marker_pids = sorted(
        {
            int(record["pid"])
            for record in ascend_marker_records
            if "pid" in record
        }
    )
    ascend_marker_ops = sorted(
        {
            str(record["op"])
            for record in ascend_marker_records
            if "op" in record
        }
    )
    sparse.update(
        {
            "sparse_gemv_invocations_current_process": (
                sparse_invocations_current_process
            ),
            "sparse_gemv_marker_records": len(sparse_marker_records),
            "sparse_gemv_marker_pids": sparse_marker_pids,
            "sparse_gemv_marker_x_shapes": unique_marker_values(
                sparse_marker_records,
                "x_shape",
            ),
            "sparse_gemv_marker_threshold_shapes": unique_marker_values(
                sparse_marker_records,
                "threshold_shape",
            ),
            "sparse_gemv_marker_threshold_numels": unique_marker_values(
                sparse_marker_records,
                "threshold_numel",
            ),
            "sparse_gemv_marker_inclusive": unique_marker_values(
                sparse_marker_records,
                "inclusive",
            ),
            "sparse_gemv_marker_weight_t_provided": unique_marker_values(
                sparse_marker_records,
                "weight_t_provided",
            ),
            "ascend_sparse_linear_marker_records": len(ascend_marker_records),
            "ascend_sparse_linear_marker_pids": ascend_marker_pids,
            "ascend_sparse_linear_marker_ops": ascend_marker_ops,
            "ascend_sparse_linear_marker_x_shapes": unique_marker_values(
                ascend_marker_records,
                "x_shape",
            ),
            "ascend_sparse_linear_marker_threshold_shapes": unique_marker_values(
                ascend_marker_records,
                "threshold_shape",
            ),
            "ascend_sparse_linear_marker_threshold_numels": unique_marker_values(
                ascend_marker_records,
                "threshold_numel",
            ),
            "ascend_sparse_linear_marker_inclusive": unique_marker_values(
                ascend_marker_records,
                "inclusive",
            ),
            "ascend_sparse_linear_marker_weight_t_provided": unique_marker_values(
                ascend_marker_records,
                "weight_t_provided",
            ),
        }
    )
    marker_tmpdir.cleanup()

    total_token_speedup = (
        sparse["total_tokens_per_second"] / dense["total_tokens_per_second"]
    )
    output_token_speedup = (
        sparse["output_tokens_per_second"] / dense["output_tokens_per_second"]
    )

    failures = []
    for case in (dense, sparse):
        if case["output_tokens"] != case["expected_output_tokens"]:
            failures.append(
                f"{case['label']} generated {case['output_tokens']} output "
                f"tokens, expected {case['expected_output_tokens']}"
            )
    if (
        not args.no_require_sparse_invocations
        and sparse_invocations_current_process == 0
        and not sparse_marker_records
    ):
        failures.append("sparse run did not invoke vLLM sparse GEMV shim")
    if not args.allow_sparse_gemv_fallback and not ascend_marker_records:
        failures.append(
            "sparse run did not invoke Ascend sparse linear custom-op wrapper"
        )
    if total_token_speedup < args.min_total_token_speedup:
        failures.append(
            "total token throughput speedup "
            f"{total_token_speedup:.6g} < {args.min_total_token_speedup:.6g}"
        )
    if output_token_speedup < args.min_output_token_speedup:
        failures.append(
            "output token throughput speedup "
            f"{output_token_speedup:.6g} < {args.min_output_token_speedup:.6g}"
        )

    result = {
        "method": args.method,
        "model": args.model,
        "calibration_path": str(args.calibration_path.resolve()),
        "sparsity": args.sparsity,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "dtype": args.dtype,
        "require_sparse_gemv_kernel": not args.allow_sparse_gemv_fallback,
        "dense": dense,
        "sparse": sparse,
        "total_token_speedup": total_token_speedup,
        "output_token_speedup": output_token_speedup,
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
