# SPDX-License-Identifier: Apache-2.0
"""Run same-config sparse PPL and generation-throughput gates.

This script intentionally orchestrates the existing evaluation scripts instead
of duplicating model logic. It produces a single summary artifact that fails
unless the PPL run and throughput run use the same sparse config, invoke the
Ascend sparse custom op, and meet the requested quality/speed thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path(".cache/sparse_kernel_results")
SUPPORTED_PROJECTIONS = (
    "self_attn.qkv",
    "self_attn.o",
    "mlp.gate_up",
    "mlp.down",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a decode-aligned PPL check and a generation-throughput check "
            "for one shared sparse config."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["teal", "larosa"], required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument(
        "--teal-artifact-root",
        type=Path,
        default=None,
        help=(
            "TEAL artifact root for histogram discovery. Required by "
            "eval_teal_ppl_alignment.py unless --teal-hf-histogram-path is set."
        ),
    )
    parser.add_argument("--teal-hf-histogram-path", type=Path, default=None)
    parser.add_argument("--teal-repo-path", type=Path, default=None)
    parser.add_argument("--sparsity", type=float, default=0.4)
    parser.add_argument(
        "--target-projection",
        action="append",
        choices=SUPPORTED_PROJECTIONS,
        default=None,
    )
    parser.add_argument("--target-layer", action="append", type=int, default=None)
    parser.add_argument(
        "--vllm-prefill-sparsify",
        choices=["half", "all", "none"],
        default="none",
    )
    parser.add_argument(
        "--vllm-score-mode",
        choices=[
            "prompt_logprobs",
            "decode_next_token",
            "forced_decode_next_token",
        ],
        default="forced_decode_next_token",
    )
    parser.add_argument(
        "--sparse-gemv-dense-fallback-policy",
        choices=["mask", "identity"],
        default="mask",
    )
    parser.add_argument("--sparse-gemv-min-sparsity", type=float, default=None)
    parser.add_argument(
        "--allow-sparse-gemv-fallback",
        action="store_true",
        help="Allow runs without the Ascend sparse custom-op wrapper.",
    )

    dataset = parser.add_argument_group("PPL dataset")
    dataset.add_argument("--dataset-name", default=None)
    dataset.add_argument("--dataset-subset", default=None)
    dataset.add_argument("--dataset-split", default=None)
    dataset.add_argument("--dataset-text-field", default=None)
    dataset.add_argument("--dataset-size", type=int, default=8)
    dataset.add_argument("--dataset-sample", choices=["first", "random"], default=None)
    dataset.add_argument("--dataset-seed", type=int, default=0)
    dataset.add_argument("--text-file", type=Path, default=None)
    dataset.add_argument("--inline-text", action="append", default=None)
    dataset.add_argument("--context-size", type=int, default=256)
    dataset.add_argument("--window-size", type=int, default=64)
    dataset.add_argument("--max-windows", type=int, default=2)

    generation = parser.add_argument_group("generation throughput")
    generation.add_argument("--num-prompts", type=int, default=1)
    generation.add_argument(
        "--warmup-prompts",
        type=int,
        default=1,
        help=(
            "Prompts used to warm dense and sparse runs before timed "
            "generation. A sparse warmup also records marker evidence before "
            "marker env vars are disabled for timing."
        ),
    )
    generation.add_argument("--input-len", type=int, default=128)
    generation.add_argument("--output-len", type=int, default=128)
    generation.add_argument("--seed", type=int, default=0)
    generation.add_argument("--max-num-batched-tokens", type=int, default=None)
    generation.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    generation.add_argument(
        "--keep-sparse-markers-during-timing",
        action="store_true",
    )
    generation.add_argument("--sparse-marker-limit", type=int, default=None)
    generation.add_argument("--sparse-marker-sequence-limit", type=int, default=512)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--python", default=sys.executable)
    runtime.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    runtime.add_argument("--max-model-len", type=int, default=None)
    runtime.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    runtime.add_argument("--shutdown-timeout", type=int, default=60)
    runtime.add_argument("--trust-remote-code", action="store_true")
    runtime.add_argument("--dry-run", action="store_true")
    runtime.add_argument(
        "--sparse-linear-policy",
        choices=["auto", "all", "none"],
        default=None,
        help=(
            "Optional VLLM_SPARSE_GEMV_LINEAR_POLICY override shared by the "
            "PPL and throughput child processes. Use 'all' when forced-decode "
            "PPL batches multiple decode rows but the Ascend sparse op should "
            "still be required and marker-checked."
        ),
    )

    gate = parser.add_argument_group("gate thresholds")
    gate.add_argument("--max-ppl-ratio", type=float, default=1.05)
    gate.add_argument("--max-delta-nll", type=float, default=None)
    gate.add_argument("--min-total-token-speedup", type=float, default=0.0)
    gate.add_argument("--min-output-token-speedup", type=float, default=1.0)
    gate.add_argument("--max-decode-marker-rows", type=int, default=8)
    gate.add_argument(
        "--no-require-decode-marker-shape",
        action="store_true",
        help="Do not require a small-row decode marker in each sparse run.",
    )
    gate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    gate.add_argument("--ppl-json-output", type=Path, default=None)
    gate.add_argument("--throughput-json-output", type=Path, default=None)
    gate.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def sparsity_slug(value: float) -> str:
    return f"s{value:.2f}".replace(".", "p")


def default_artifact_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    projections = "all" if args.target_projection is None else "_".join(
        proj.replace(".", "_") for proj in args.target_projection
    )
    layers = "all" if args.target_layer is None else "layers" + "_".join(
        str(layer) for layer in args.target_layer
    )
    prefix = (
        f"{args.method}_{sparsity_slug(args.sparsity)}_{projections}_{layers}_"
        f"{args.vllm_prefill_sparsify}_{args.vllm_score_mode}"
    )
    output_root = args.output_root
    ppl_path = args.ppl_json_output or output_root / f"{prefix}_ppl.json"
    throughput_path = (
        args.throughput_json_output or output_root / f"{prefix}_throughput.json"
    )
    summary_path = args.json_output or output_root / f"{prefix}_gate.json"
    return ppl_path, throughput_path, summary_path


def append_if_value(command: list[str], option: str, value: Any) -> None:
    if value is not None:
        command.extend([option, str(value)])


def append_repeated(command: list[str], option: str, values: list[Any] | None) -> None:
    if values is None:
        return
    for value in values:
        command.extend([option, str(value)])


def append_common_dataset_args(
    command: list[str],
    args: argparse.Namespace,
    include_text: bool = True,
) -> None:
    append_if_value(command, "--dataset-name", args.dataset_name)
    append_if_value(command, "--dataset-subset", args.dataset_subset)
    append_if_value(command, "--dataset-split", args.dataset_split)
    append_if_value(command, "--dataset-text-field", args.dataset_text_field)
    command.extend(["--dataset-size", str(args.dataset_size)])
    append_if_value(command, "--dataset-sample", args.dataset_sample)
    command.extend(["--dataset-seed", str(args.dataset_seed)])
    if include_text:
        append_if_value(command, "--text-file", args.text_file)
        append_repeated(command, "--inline-text", args.inline_text)
    command.extend(["--context-size", str(args.context_size)])
    command.extend(["--window-size", str(args.window_size)])
    append_if_value(command, "--max-windows", args.max_windows)


def build_ppl_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    if args.method == "teal":
        script = SCRIPT_DIR / "eval_teal_ppl_alignment.py"
        command = [
            args.python,
            str(script),
            "--model",
            args.model,
            "--sparsity",
            str(args.sparsity),
            "--backend",
            "vllm",
            "--vllm-calibration-path",
            str(args.calibration_path),
        ]
        append_if_value(command, "--artifact-root", args.teal_artifact_root)
        append_if_value(command, "--hf-histogram-path", args.teal_hf_histogram_path)
        append_if_value(command, "--teal-repo-path", args.teal_repo_path)
    else:
        script = SCRIPT_DIR / "eval_larosa_ppl_alignment.py"
        command = [
            args.python,
            str(script),
            "--model",
            args.model,
            "--sparsity",
            str(args.sparsity),
            "--backend",
            "vllm",
            "--calibration-path",
            str(args.calibration_path),
        ]

    append_common_dataset_args(command, args)
    command.extend(["--vllm-score-mode", args.vllm_score_mode])
    command.extend(["--vllm-prefill-sparsify", args.vllm_prefill_sparsify])
    append_repeated(command, "--target-projection", args.target_projection)
    append_repeated(command, "--target-layer", args.target_layer)
    command.append("--vllm-use-sparse-gemv")
    if args.allow_sparse_gemv_fallback:
        command.append("--vllm-allow-sparse-gemv-fallback")
    command.extend(
        [
            "--vllm-sparse-gemv-dense-fallback-policy",
            args.sparse_gemv_dense_fallback_policy,
        ]
    )
    append_if_value(
        command,
        "--vllm-sparse-gemv-min-sparsity",
        args.sparse_gemv_min_sparsity,
    )
    command.extend(["--dtype", args.dtype])
    command.extend(["--device", args.device])
    command.extend(["--vllm-gpu-memory-utilization", str(args.gpu_memory_utilization)])
    append_if_value(command, "--vllm-max-model-len", args.max_model_len)
    command.append(
        "--vllm-enforce-eager" if args.vllm_enforce_eager else "--no-vllm-enforce-eager"
    )
    command.extend(["--vllm-shutdown-timeout", str(args.shutdown_timeout)])
    command.extend(["--json-output", str(output_path)])
    return command


def build_throughput_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "bench_sparse_kernel_throughput.py"),
        "--model",
        args.model,
        "--method",
        args.method,
        "--calibration-path",
        str(args.calibration_path),
        "--sparsity",
        str(args.sparsity),
        "--num-prompts",
        str(args.num_prompts),
        "--warmup-prompts",
        str(args.warmup_prompts),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--shutdown-timeout",
        str(args.shutdown_timeout),
        "--vllm-prefill-sparsify",
        args.vllm_prefill_sparsify,
        "--sparse-gemv-dense-fallback-policy",
        args.sparse_gemv_dense_fallback_policy,
        "--min-total-token-speedup",
        str(args.min_total_token_speedup),
        "--min-output-token-speedup",
        str(args.min_output_token_speedup),
        "--sparse-marker-sequence-limit",
        str(args.sparse_marker_sequence_limit),
        "--json-output",
        str(output_path),
    ]
    append_if_value(command, "--max-model-len", args.max_model_len)
    append_if_value(command, "--max-num-batched-tokens", args.max_num_batched_tokens)
    append_if_value(command, "--kv-cache-memory-bytes", args.kv_cache_memory_bytes)
    append_if_value(command, "--text-file", args.text_file)
    append_repeated(command, "--inline-text", args.inline_text)
    append_repeated(command, "--target-projection", args.target_projection)
    append_repeated(command, "--target-layer", args.target_layer)
    append_if_value(command, "--sparse-gemv-min-sparsity", args.sparse_gemv_min_sparsity)
    append_if_value(command, "--sparse-marker-limit", args.sparse_marker_limit)
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.vllm_enforce_eager:
        command.append("--enforce-eager")
    if args.allow_sparse_gemv_fallback:
        command.append("--allow-sparse-gemv-fallback")
    if args.keep_sparse_markers_during_timing:
        command.append("--keep-sparse-markers-during-timing")
    return command


def run_command(command: list[str], args: argparse.Namespace) -> int:
    print("+ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.sparse_linear_policy is not None:
        env["VLLM_SPARSE_GEMV_LINEAR_POLICY"] = args.sparse_linear_policy
    completed = subprocess.run(command, check=False, env=env)
    return completed.returncode


def load_json(path: Path, failures: list[str], label: str) -> dict[str, Any] | None:
    if not path.exists():
        failures.append(f"{label} JSON was not created: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"{label} JSON is invalid: {exc}")
        return None


def normalize_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def compare_optional_list(
    failures: list[str],
    label: str,
    expected: list[Any] | None,
    actual: Any,
) -> None:
    normalized_actual = normalize_list(actual)
    if expected != normalized_actual:
        failures.append(
            f"{label} mismatch: expected {expected}, got {normalized_actual}"
        )


def shape_has_decode_rows(shapes: Any, max_rows: int) -> bool:
    if not isinstance(shapes, list):
        return False
    for shape in shapes:
        if (
            isinstance(shape, list)
            and shape
            and isinstance(shape[0], int)
            and 0 < shape[0] <= max_rows
        ):
            return True
    return False


def marker_failures(
    markers: dict[str, Any] | None,
    args: argparse.Namespace,
    label: str,
) -> list[str]:
    failures: list[str] = []
    if not markers:
        return [f"{label} has no sparse kernel marker summary"]
    ascend_records = int(markers.get("ascend_sparse_linear_marker_records") or 0)
    if not args.allow_sparse_gemv_fallback and ascend_records <= 0:
        failures.append(f"{label} did not invoke Ascend sparse linear custom op")
    sparse_records = int(markers.get("sparse_gemv_marker_records") or 0)
    if sparse_records <= 0 and ascend_records <= 0:
        failures.append(f"{label} did not record sparse GEMV activity")
    if (
        not args.no_require_decode_marker_shape
        and not shape_has_decode_rows(
            markers.get("ascend_sparse_linear_marker_x_shapes"),
            args.max_decode_marker_rows,
        )
    ):
        failures.append(
            f"{label} has no decode-sized Ascend marker x_shape "
            f"(max rows {args.max_decode_marker_rows})"
        )
    return failures


def validate_ppl_result(
    data: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    failures: list[str] = []
    if data is None:
        return metrics, failures

    setup = data.get("setup") or {}
    vllm = data.get("vllm") or {}
    if not vllm:
        failures.append("PPL JSON has no vLLM result")
        return metrics, failures

    if setup.get("model") != args.model:
        failures.append(f"PPL model mismatch: expected {args.model}, got {setup.get('model')}")
    if not math.isclose(float(setup.get("sparsity", -1.0)), args.sparsity):
        failures.append(
            f"PPL sparsity mismatch: expected {args.sparsity}, got {setup.get('sparsity')}"
        )
    if setup.get("vllm_score_mode") != args.vllm_score_mode:
        failures.append(
            "PPL score mode mismatch: "
            f"expected {args.vllm_score_mode}, got {setup.get('vllm_score_mode')}"
        )
    if setup.get("vllm_prefill_sparsify") != args.vllm_prefill_sparsify:
        failures.append(
            "PPL prefill policy mismatch: "
            f"expected {args.vllm_prefill_sparsify}, "
            f"got {setup.get('vllm_prefill_sparsify')}"
        )

    setup_layers = setup.get("vllm_target_layers", setup.get("target_layers"))
    setup_projections = setup.get(
        "vllm_target_projections",
        setup.get("target_projections"),
    )
    compare_optional_list(failures, "PPL target layers", args.target_layer, setup_layers)
    compare_optional_list(
        failures,
        "PPL target projections",
        args.target_projection,
        setup_projections,
    )
    if setup.get("vllm_sparse_linear_policy") != args.sparse_linear_policy:
        failures.append(
            "PPL sparse linear policy mismatch: "
            f"expected {args.sparse_linear_policy}, "
            f"got {setup.get('vllm_sparse_linear_policy')}"
        )

    ppl_ratio = vllm.get("ppl_ratio")
    delta_nll = vllm.get("delta_nll")
    metrics["dense_ppl"] = (vllm.get("dense") or {}).get("ppl")
    metrics["sparse_ppl"] = (vllm.get("sparse") or {}).get("ppl")
    metrics["ppl_ratio"] = ppl_ratio
    metrics["delta_nll"] = delta_nll
    if ppl_ratio is None or not math.isfinite(float(ppl_ratio)):
        failures.append(f"PPL ratio is invalid: {ppl_ratio}")
    elif float(ppl_ratio) > args.max_ppl_ratio:
        failures.append(
            f"PPL ratio {float(ppl_ratio):.6g} > {args.max_ppl_ratio:.6g}"
        )
    if args.max_delta_nll is not None:
        if delta_nll is None or not math.isfinite(float(delta_nll)):
            failures.append(f"delta_nll is invalid: {delta_nll}")
        elif float(delta_nll) > args.max_delta_nll:
            failures.append(
                f"delta_nll {float(delta_nll):.6g} > {args.max_delta_nll:.6g}"
            )
    failures.extend(
        marker_failures(vllm.get("sparse_kernel_markers"), args, "PPL sparse run")
    )
    return metrics, failures


def validate_throughput_result(
    data: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    failures: list[str] = []
    if data is None:
        return metrics, failures

    if data.get("method") != args.method:
        failures.append(
            f"throughput method mismatch: expected {args.method}, got {data.get('method')}"
        )
    if data.get("model") != args.model:
        failures.append(
            f"throughput model mismatch: expected {args.model}, got {data.get('model')}"
        )
    if not math.isclose(float(data.get("sparsity", -1.0)), args.sparsity):
        failures.append(
            "throughput sparsity mismatch: "
            f"expected {args.sparsity}, got {data.get('sparsity')}"
        )
    if data.get("vllm_prefill_sparsify") != args.vllm_prefill_sparsify:
        failures.append(
            "throughput prefill policy mismatch: "
            f"expected {args.vllm_prefill_sparsify}, "
            f"got {data.get('vllm_prefill_sparsify')}"
        )
    compare_optional_list(
        failures,
        "throughput target layers",
        args.target_layer,
        data.get("target_layers"),
    )
    compare_optional_list(
        failures,
        "throughput target projections",
        args.target_projection,
        data.get("target_projections"),
    )
    if data.get("sparse_linear_policy") != args.sparse_linear_policy:
        failures.append(
            "throughput sparse linear policy mismatch: "
            f"expected {args.sparse_linear_policy}, "
            f"got {data.get('sparse_linear_policy')}"
        )

    output_speedup = data.get("output_token_speedup")
    total_speedup = data.get("total_token_speedup")
    metrics["output_token_speedup"] = output_speedup
    metrics["total_token_speedup"] = total_speedup
    if output_speedup is None or not math.isfinite(float(output_speedup)):
        failures.append(f"output token speedup is invalid: {output_speedup}")
    elif float(output_speedup) < args.min_output_token_speedup:
        failures.append(
            "output token throughput speedup "
            f"{float(output_speedup):.6g} < {args.min_output_token_speedup:.6g}"
        )
    if total_speedup is None or not math.isfinite(float(total_speedup)):
        failures.append(f"total token speedup is invalid: {total_speedup}")
    elif float(total_speedup) < args.min_total_token_speedup:
        failures.append(
            "total token throughput speedup "
            f"{float(total_speedup):.6g} < {args.min_total_token_speedup:.6g}"
        )
    failures.extend(data.get("failures") or [])
    failures.extend(marker_failures(data.get("sparse"), args, "throughput sparse run"))
    return metrics, failures


def summary_payload(
    args: argparse.Namespace,
    ppl_path: Path,
    throughput_path: Path,
    ppl_command: list[str],
    throughput_command: list[str],
    ppl_returncode: int | None,
    throughput_returncode: int | None,
    ppl_metrics: dict[str, Any],
    throughput_metrics: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    return {
        "method": args.method,
        "model": args.model,
        "calibration_path": str(args.calibration_path.resolve()),
        "sparsity": args.sparsity,
        "target_layers": args.target_layer,
        "target_projections": args.target_projection,
        "vllm_prefill_sparsify": args.vllm_prefill_sparsify,
        "vllm_score_mode": args.vllm_score_mode,
        "sparse_linear_policy": args.sparse_linear_policy,
        "max_ppl_ratio": args.max_ppl_ratio,
        "max_delta_nll": args.max_delta_nll,
        "min_total_token_speedup": args.min_total_token_speedup,
        "min_output_token_speedup": args.min_output_token_speedup,
        "artifacts": {
            "ppl": str(ppl_path),
            "throughput": str(throughput_path),
        },
        "commands": {
            "ppl": ppl_command,
            "throughput": throughput_command,
        },
        "returncodes": {
            "ppl": ppl_returncode,
            "throughput": throughput_returncode,
        },
        "ppl": ppl_metrics,
        "throughput": throughput_metrics,
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    if args.sparse_gemv_min_sparsity is not None and not (
        0.0 <= args.sparse_gemv_min_sparsity <= 1.0
    ):
        raise ValueError("--sparse-gemv-min-sparsity must be in [0, 1]")
    if args.max_decode_marker_rows <= 0:
        raise ValueError("--max-decode-marker-rows must be positive")

    ppl_path, throughput_path, summary_path = default_artifact_paths(args)
    ppl_path.parent.mkdir(parents=True, exist_ok=True)
    throughput_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ppl_command = build_ppl_command(args, ppl_path)
    throughput_command = build_throughput_command(args, throughput_path)
    ppl_returncode: int | None = None
    throughput_returncode: int | None = None

    if not args.dry_run:
        ppl_returncode = run_command(ppl_command, args)
        throughput_returncode = run_command(throughput_command, args)

    failures: list[str] = []
    if args.dry_run:
        failures.append("dry run: commands were not executed")
        ppl_metrics: dict[str, Any] = {}
        throughput_metrics: dict[str, Any] = {}
    else:
        if ppl_returncode != 0:
            failures.append(f"PPL command exited with {ppl_returncode}")
        if throughput_returncode != 0:
            failures.append(
                f"throughput command exited with {throughput_returncode}"
            )
        ppl_data = load_json(ppl_path, failures, "PPL")
        throughput_data = load_json(throughput_path, failures, "throughput")
        ppl_metrics, ppl_failures = validate_ppl_result(ppl_data, args)
        throughput_metrics, throughput_failures = validate_throughput_result(
            throughput_data,
            args,
        )
        failures.extend(ppl_failures)
        failures.extend(throughput_failures)

    payload = summary_payload(
        args,
        ppl_path,
        throughput_path,
        ppl_command,
        throughput_command,
        ppl_returncode,
        throughput_returncode,
        ppl_metrics,
        throughput_metrics,
        failures,
    )
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
