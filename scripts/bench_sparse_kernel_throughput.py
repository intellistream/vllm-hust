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


SPARSE_MARKER_ENV_NAMES = (
    "VLLM_SPARSE_GEMV_MARKER_PATH",
    "VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH",
)

LAROSA_SECOND_SITE_PROJECTIONS = {"self_attn.o", "mlp.down"}


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
        "--text-file",
        type=Path,
        default=None,
        help=(
            "Optional plain-text prompt source. When set, prompts are "
            "deterministic token windows from this text instead of random "
            "token ids."
        ),
    )
    parser.add_argument(
        "--inline-text",
        action="append",
        default=None,
        help=(
            "Inline prompt source. Can be passed multiple times. Ignored when "
            "--text-file is set."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Optional vLLM scheduler max_num_batched_tokens override. This is "
            "useful for single-batch kernel checks where the default scheduler "
            "token buffer is much larger than the measured request."
        ),
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--shutdown-timeout", type=int, default=60)
    parser.add_argument("--vllm-prefill-sparsify", default="none")
    parser.add_argument(
        "--target-projection",
        action="append",
        choices=["self_attn.qkv", "self_attn.o", "mlp.gate_up", "mlp.down"],
        default=None,
        help=(
            "Restrict activation sparsity to a projection. Can be repeated. "
            "Useful for Ascend-friendly gate/up-only throughput checks."
        ),
    )
    parser.add_argument(
        "--target-layer",
        action="append",
        type=int,
        default=None,
        help=(
            "Restrict activation sparsity to this zero-based layer id. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--sparse-gemv-dense-fallback-policy",
        choices=["mask", "identity"],
        default="mask",
        help=(
            "Policy for projections rejected by the sparse GEMV kernel policy. "
            "'mask' preserves TEAL/La RoSA dense fallback semantics; 'identity' "
            "leaves all-row rejected projections dense for kernel-only "
            "throughput probes."
        ),
    )
    parser.add_argument(
        "--sparse-gemv-min-sparsity",
        type=float,
        default=None,
        help=(
            "Minimum expected sparsity for auto sparse GEMV dispatch. If unset, "
            "the runtime default is used."
        ),
    )
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


def build_text_prompt_ids(args: argparse.Namespace, tokenizer: Any) -> list[int] | None:
    if args.text_file is None and not args.inline_text:
        return None
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = "\n\n".join(args.inline_text)
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    token_ids = [int(token_id) for token_id in token_ids]
    if not token_ids:
        raise ValueError("Text prompt source produced no tokens.")
    return token_ids


def build_prompts(args: argparse.Namespace) -> list[dict[str, list[int]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    text_prompt_ids = build_text_prompt_ids(args, tokenizer)
    if text_prompt_ids is not None:
        if len(text_prompt_ids) < args.input_len:
            repeats = (
                args.input_len + len(text_prompt_ids) - 1
            ) // len(text_prompt_ids)
            text_prompt_ids = (text_prompt_ids * repeats)[: args.input_len]
        prompts = []
        for prompt_idx in range(args.num_prompts):
            start = (prompt_idx * args.input_len) % len(text_prompt_ids)
            prompt_token_ids = text_prompt_ids[start : start + args.input_len]
            if len(prompt_token_ids) < args.input_len:
                prompt_token_ids += text_prompt_ids[
                    : args.input_len - len(prompt_token_ids)
                ]
            prompts.append({"prompt_token_ids": prompt_token_ids})
        return prompts

    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError(
            f"Tokenizer for {args.model!r} has invalid vocab_size={vocab_size}"
        )

    excluded_ids = {
        token_id
        for token_id in tokenizer.all_special_ids
        if isinstance(token_id, int) and 0 <= token_id < vocab_size
    }
    candidate_ids = [i for i in range(vocab_size) if i not in excluded_ids]
    if not candidate_ids:
        raise ValueError(f"Tokenizer for {args.model!r} has no non-special token ids")

    rng = random.Random(args.seed)
    prompts: list[dict[str, list[int]]] = []
    for _ in range(args.num_prompts):
        prompt_token_ids = [rng.choice(candidate_ids) for _ in range(args.input_len)]
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
    if args.target_projection:
        common["target_projections"] = args.target_projection
    if args.target_layer:
        common["target_layers"] = args.target_layer
    return common


def targets_larosa_second_site(target_projections: list[str] | None) -> bool:
    if target_projections is None:
        return True
    return any(proj in LAROSA_SECOND_SITE_PROJECTIONS for proj in target_projections)


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
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    return LLM(**kwargs)


def run_case(
    args: argparse.Namespace,
    prompts: list[dict[str, list[int]]],
    activation_sparsity_config: dict[str, Any] | None,
    label: str,
    disable_sparse_markers_during_timing: bool = False,
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

    disabled_marker_env: dict[str, str | None] | None = None
    if disable_sparse_markers_during_timing:
        marker_paths = [
            Path(os.environ[name])
            for name in SPARSE_MARKER_ENV_NAMES
            if os.environ.get(name)
        ]
        markers_ready = len(marker_paths) == len(SPARSE_MARKER_ENV_NAMES) and all(
            marker_path.exists() and marker_path.stat().st_size > 0
            for marker_path in marker_paths
        )
        if markers_ready:
            disabled_marker_env = {
                name: os.environ.get(name) for name in SPARSE_MARKER_ENV_NAMES
            }
            for name in SPARSE_MARKER_ENV_NAMES:
                os.environ.pop(name, None)

    synchronize()
    try:
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        synchronize()
        elapsed = time.perf_counter() - start
    finally:
        if disabled_marker_env is not None:
            for name, value in disabled_marker_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

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
        "sparse_markers_disabled_during_timing": disabled_marker_env is not None,
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
    if (
        args.sparse_gemv_min_sparsity is not None
        and not 0.0 <= args.sparse_gemv_min_sparsity <= 1.0
    ):
        raise ValueError("--sparse-gemv-min-sparsity must be in [0, 1]")
    if (
        args.method == "larosa"
        and targets_larosa_second_site(args.target_projection)
        and args.sparsity * 1.2 >= 1.0
    ):
        raise ValueError("La RoSA h2 sparsity is sparsity * 1.2 and must be < 1")

    prompts = build_prompts(args)
    dense = run_case(args, prompts, None, "dense")

    previous_require_kernel = os.environ.get("VLLM_SPARSE_GEMV_REQUIRE_KERNEL")
    previous_marker_path = os.environ.get("VLLM_SPARSE_GEMV_MARKER_PATH")
    previous_ascend_marker_path = os.environ.get(
        "VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"
    )
    previous_dense_fallback_policy = os.environ.get(
        "VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY"
    )
    previous_min_sparsity = os.environ.get("VLLM_SPARSE_GEMV_MIN_SPARSITY")
    os.environ["VLLM_SPARSE_GEMV_REQUIRE_KERNEL"] = (
        "0" if args.allow_sparse_gemv_fallback else "1"
    )
    os.environ["VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY"] = (
        args.sparse_gemv_dense_fallback_policy
    )
    if args.sparse_gemv_min_sparsity is not None:
        os.environ["VLLM_SPARSE_GEMV_MIN_SPARSITY"] = str(
            args.sparse_gemv_min_sparsity
        )
    active_min_sparsity = os.environ.get("VLLM_SPARSE_GEMV_MIN_SPARSITY", "0.70")
    marker_tmpdir = tempfile.TemporaryDirectory(prefix="vllm_sparse_gemv_")
    marker_path = Path(marker_tmpdir.name) / "invocations.jsonl"
    ascend_marker_path = Path(marker_tmpdir.name) / "ascend_custom_ops.jsonl"
    os.environ["VLLM_SPARSE_GEMV_MARKER_PATH"] = str(marker_path)
    os.environ["VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH"] = str(ascend_marker_path)
    reset_sparse_invocation_counter()
    try:
        sparse = run_case(
            args,
            prompts,
            build_sparse_config(args),
            "sparse",
            disable_sparse_markers_during_timing=True,
        )
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
        if previous_dense_fallback_policy is None:
            os.environ.pop("VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_DENSE_FALLBACK_POLICY"] = (
                previous_dense_fallback_policy
            )
        if previous_min_sparsity is None:
            os.environ.pop("VLLM_SPARSE_GEMV_MIN_SPARSITY", None)
        else:
            os.environ["VLLM_SPARSE_GEMV_MIN_SPARSITY"] = previous_min_sparsity

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
            "sparse_gemv_marker_active_row_counts": unique_marker_values(
                sparse_marker_records,
                "active_row_count",
            ),
            "sparse_gemv_marker_active_hidden_sizes": unique_marker_values(
                sparse_marker_records,
                "active_hidden_size",
            ),
            "sparse_gemv_marker_active_count_mins": unique_marker_values(
                sparse_marker_records,
                "active_count_min",
            ),
            "sparse_gemv_marker_active_count_maxes": unique_marker_values(
                sparse_marker_records,
                "active_count_max",
            ),
            "sparse_gemv_marker_active_count_means": unique_marker_values(
                sparse_marker_records,
                "active_count_mean",
            ),
            "sparse_gemv_marker_active_density_mins": unique_marker_values(
                sparse_marker_records,
                "active_density_min",
            ),
            "sparse_gemv_marker_active_density_maxes": unique_marker_values(
                sparse_marker_records,
                "active_density_max",
            ),
            "sparse_gemv_marker_active_density_means": unique_marker_values(
                sparse_marker_records,
                "active_density_mean",
            ),
            "sparse_gemv_marker_active_sparsity_means": unique_marker_values(
                sparse_marker_records,
                "active_sparsity_mean",
            ),
            "sparse_gemv_marker_active_stats_errors": unique_marker_values(
                sparse_marker_records,
                "active_stats_error",
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
        and not ascend_marker_records
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
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "prompt_source": (
            "text" if args.text_file is not None or args.inline_text else "random"
        ),
        "dtype": args.dtype,
        "sparse_linear_policy": os.environ.get("VLLM_SPARSE_GEMV_LINEAR_POLICY"),
        "target_layers": args.target_layer,
        "sparse_gemv_dense_fallback_policy": (
            args.sparse_gemv_dense_fallback_policy
        ),
        "sparse_gemv_min_sparsity": active_min_sparsity,
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
