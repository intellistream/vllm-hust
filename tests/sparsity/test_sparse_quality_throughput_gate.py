# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sparse_quality_throughput_gate import (  # noqa: E402
    build_ppl_command,
    build_throughput_command,
    validate_ppl_result,
    validate_throughput_result,
)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python="python",
        method="teal",
        model="test-model",
        calibration_path=tmp_path / "thresholds",
        teal_artifact_root=tmp_path / "teal",
        teal_hf_histogram_path=None,
        teal_repo_path=None,
        sparsity=0.4,
        target_projection=["mlp.gate_up"],
        target_layer=[0, 26],
        vllm_prefill_sparsify="none",
        vllm_score_mode="forced_decode_next_token",
        sparse_gemv_dense_fallback_policy="mask",
        sparse_gemv_min_sparsity=None,
        allow_sparse_gemv_fallback=False,
        dataset_name=None,
        dataset_subset=None,
        dataset_split=None,
        dataset_text_field=None,
        dataset_size=2,
        dataset_sample=None,
        dataset_seed=0,
        text_file=None,
        inline_text=["alpha beta gamma"],
        context_size=32,
        window_size=8,
        max_windows=1,
        num_prompts=1,
        warmup_prompts=0,
        input_len=16,
        output_len=8,
        seed=0,
        max_num_batched_tokens=None,
        kv_cache_memory_bytes=None,
        keep_sparse_markers_during_timing=False,
        sparse_marker_limit=None,
        sparse_marker_sequence_limit=64,
        dtype="float16",
        device="npu",
        gpu_memory_utilization=0.75,
        max_model_len=None,
        vllm_enforce_eager=True,
        shutdown_timeout=30,
        trust_remote_code=False,
        sparse_linear_policy=None,
        max_ppl_ratio=1.05,
        max_delta_nll=None,
        min_total_token_speedup=0.0,
        min_output_token_speedup=1.0,
        max_decode_marker_rows=8,
        no_require_decode_marker_shape=False,
    )


def _markers(rows: int = 3) -> dict:
    return {
        "sparse_gemv_marker_records": 1,
        "ascend_sparse_linear_marker_records": 1,
        "ascend_sparse_linear_marker_ops": [
            "activation_sparse_silu_and_mul_direct_t"
        ],
        "ascend_sparse_linear_marker_x_shapes": [[rows, 3584]],
    }


def test_gate_builds_same_config_commands(tmp_path: Path) -> None:
    args = _args(tmp_path)
    ppl_command = build_ppl_command(args, tmp_path / "ppl.json")
    throughput_command = build_throughput_command(args, tmp_path / "throughput.json")

    for command in (ppl_command, throughput_command):
        assert "--target-projection" in command
        assert "mlp.gate_up" in command
        assert "--target-layer" in command
        assert "26" in command
        assert "--vllm-prefill-sparsify" in command
        assert "none" in command

    assert "--vllm-score-mode" in ppl_command
    assert "forced_decode_next_token" in ppl_command
    assert "--method" in throughput_command
    assert "teal" in throughput_command


def test_gate_accepts_matching_artifacts(tmp_path: Path) -> None:
    args = _args(tmp_path)
    ppl = {
        "setup": {
            "model": args.model,
            "sparsity": args.sparsity,
            "vllm_prefill_sparsify": "none",
            "vllm_score_mode": "forced_decode_next_token",
            "vllm_target_layers": [0, 26],
            "vllm_target_projections": ["mlp.gate_up"],
        },
        "vllm": {
            "dense": {"ppl": 100.0},
            "sparse": {"ppl": 102.0},
            "ppl_ratio": 1.02,
            "delta_nll": 0.02,
            "sparse_kernel_markers": _markers(),
        },
    }
    throughput = {
        "method": "teal",
        "model": args.model,
        "sparsity": args.sparsity,
        "vllm_prefill_sparsify": "none",
        "target_layers": [0, 26],
        "target_projections": ["mlp.gate_up"],
        "output_token_speedup": 1.01,
        "total_token_speedup": 0.99,
        "failures": [],
        "sparse": _markers(rows=1),
    }

    _, ppl_failures = validate_ppl_result(ppl, args)
    _, throughput_failures = validate_throughput_result(throughput, args)

    assert ppl_failures == []
    assert throughput_failures == []


def test_gate_rejects_mismatched_projection(tmp_path: Path) -> None:
    args = _args(tmp_path)
    ppl = {
        "setup": {
            "model": args.model,
            "sparsity": args.sparsity,
            "vllm_prefill_sparsify": "none",
            "vllm_score_mode": "forced_decode_next_token",
            "vllm_target_layers": [0, 26],
            "vllm_target_projections": ["mlp.down"],
        },
        "vllm": {
            "ppl_ratio": 1.01,
            "delta_nll": 0.01,
            "sparse_kernel_markers": _markers(),
        },
    }

    _, failures = validate_ppl_result(ppl, args)

    assert any("target projections mismatch" in failure for failure in failures)


def test_gate_rejects_mismatched_sparse_linear_policy(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.sparse_linear_policy = "all"
    ppl = {
        "setup": {
            "model": args.model,
            "sparsity": args.sparsity,
            "vllm_prefill_sparsify": "none",
            "vllm_score_mode": "forced_decode_next_token",
            "vllm_target_layers": [0, 26],
            "vllm_target_projections": ["mlp.gate_up"],
            "vllm_sparse_linear_policy": "auto",
        },
        "vllm": {
            "ppl_ratio": 1.01,
            "delta_nll": 0.01,
            "sparse_kernel_markers": _markers(),
        },
    }

    _, failures = validate_ppl_result(ppl, args)

    assert any("sparse linear policy mismatch" in failure for failure in failures)


def test_gate_rejects_missing_decode_marker_shape(tmp_path: Path) -> None:
    args = _args(tmp_path)
    throughput = {
        "method": "teal",
        "model": args.model,
        "sparsity": args.sparsity,
        "vllm_prefill_sparsify": "none",
        "target_layers": [0, 26],
        "target_projections": ["mlp.gate_up"],
        "output_token_speedup": 1.01,
        "total_token_speedup": 1.0,
        "failures": [],
        "sparse": _markers(rows=8192),
    }

    _, failures = validate_throughput_result(throughput, args)

    assert any("decode-sized Ascend marker" in failure for failure in failures)
