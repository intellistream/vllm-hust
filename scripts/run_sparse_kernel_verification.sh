#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/workspace/wanyao/seas0/vllm-hust-feat-sparse-pth}"
ASCEND_ROOT="${ASCEND_ROOT:-/workspace/wanyao/seas0/vllm-ascend-hust}"
PYTHON_BIN="${PYTHON_BIN:-${VLLM_ROOT}/.venv/bin/python}"
RESULT_DIR="${RESULT_DIR:-${VLLM_ROOT}/.cache/sparse_kernel_verification}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"

MODEL="${MODEL:-/workspace/wanyao/seas0/models/Qwen2.5-7B}"
TEAL_ARTIFACT_ROOT="${TEAL_ARTIFACT_ROOT:-${VLLM_ROOT}/.cache/teal_qwen25_7b_base_wikitext2_random500_seed0_s0.4}"
LAROSA_ARTIFACT_ROOT="${LAROSA_ARTIFACT_ROOT:-${VLLM_ROOT}/.cache/larosa_qwen25_7b_base_wikitext2_random500_seed0_selfattn}"

SKIP_BUILD="${SKIP_BUILD:-0}"
RUN_THROUGHPUT="${RUN_THROUGHPUT:-1}"
RUN_QWEN_PPL="${RUN_QWEN_PPL:-1}"
BENCH_WARMUP="${BENCH_WARMUP:-20}"
BENCH_ITERS="${BENCH_ITERS:-100}"
BENCH_DTYPE="${BENCH_DTYPE:-float16}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"
MAX_SPARSE_ERR="${MAX_SPARSE_ERR:-0.5}"
MAX_DIRECT_ERR="${MAX_DIRECT_ERR:-0.5}"
MAX_DIRECT_T_ERR="${MAX_DIRECT_T_ERR:-0.5}"
BENCH_SKIP_DIRECT="${BENCH_SKIP_DIRECT:-1}"
MIN_PACKED_TOTAL_SPEEDUP="${MIN_PACKED_TOTAL_SPEEDUP:-1.0}"
MIN_PACKED_TOTAL_WITH_THRESHOLD_SPEEDUP="${MIN_PACKED_TOTAL_WITH_THRESHOLD_SPEEDUP:-1.0}"
MIN_PACKED_COMPUTE_SPEEDUP="${MIN_PACKED_COMPUTE_SPEEDUP:-1.0}"
THROUGHPUT_NUM_PROMPTS="${THROUGHPUT_NUM_PROMPTS:-1}"
BENCH_BATCH_SIZES="${BENCH_BATCH_SIZES:-1 ${THROUGHPUT_NUM_PROMPTS}}"
THROUGHPUT_WARMUP_PROMPTS="${THROUGHPUT_WARMUP_PROMPTS:-0}"
THROUGHPUT_INPUT_LEN="${THROUGHPUT_INPUT_LEN:-128}"
THROUGHPUT_OUTPUT_LEN="${THROUGHPUT_OUTPUT_LEN:-128}"
MIN_TOTAL_TOKEN_SPEEDUP="${MIN_TOTAL_TOKEN_SPEEDUP:-1.0}"
MIN_OUTPUT_TOKEN_SPEEDUP="${MIN_OUTPUT_TOKEN_SPEEDUP:-1.0}"
THROUGHPUT_SPARSE_LINEAR_POLICY="${THROUGHPUT_SPARSE_LINEAR_POLICY:-auto}"
PPL_SPARSE_LINEAR_POLICY="${PPL_SPARSE_LINEAR_POLICY:-all}"

export ASCEND_RT_VISIBLE_DEVICES
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-npu}"
export VLLM_VERSION="${VLLM_VERSION:-0.17.2}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ASCEND_FORCE_CPU_SLOT_MAPPING="${VLLM_ASCEND_FORCE_CPU_SLOT_MAPPING:-1}"
export PYTHONPATH="${VLLM_ROOT}:${ASCEND_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RESULT_DIR}"

SOURCE_MANIFEST="${RESULT_DIR}/source_manifest.json"
export SOURCE_MANIFEST

check_throughput_marker_metadata() {
  local json_path="$1"
  local method="$2"
  "${PYTHON_BIN}" - "${json_path}" "${method}" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
method = sys.argv[2]
result = json.loads(json_path.read_text(encoding="utf-8"))
sparse = result["sparse"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"{json_path}: {message}")


def marker_values(prefix: str, key: str) -> list:
    return sparse.get(f"{prefix}_marker_{key}", [])


for prefix in ("sparse_gemv", "ascend_sparse_linear"):
    require(
        sparse.get(f"{prefix}_marker_records", 0) > 0,
        f"{prefix} marker records are missing",
    )
    threshold_numels = marker_values(prefix, "threshold_numels")
    x_shapes = marker_values(prefix, "x_shapes")
    inclusives = marker_values(prefix, "inclusive")
    require(threshold_numels, f"{prefix} threshold_numels are missing")
    require(x_shapes, f"{prefix} x_shapes are missing")
    require(inclusives, f"{prefix} inclusive values are missing")

    if method == "teal":
        require(
            all(int(numel) == 1 for numel in threshold_numels),
            f"{prefix} TEAL should use scalar threshold, got {threshold_numels}",
        )
        require(
            inclusives == [False],
            f"{prefix} TEAL should use exclusive compare, got {inclusives}",
        )
    elif method == "larosa":
        rows = {
            int(shape[0])
            for shape in x_shapes
            if isinstance(shape, list) and shape
        }
        require(rows, f"{prefix} La RoSA x_shapes do not expose batch rows")
        require(
            all(int(numel) in rows for numel in threshold_numels),
            (
                f"{prefix} La RoSA should use one threshold per row, got "
                f"threshold_numels={threshold_numels}, x_shapes={x_shapes}"
            ),
        )
        require(
            any(int(numel) > 1 for numel in threshold_numels)
            or any(
                isinstance(shape, list)
                and shape
                and int(shape[0]) == 1
                for shape in x_shapes
            ),
            (
                f"{prefix} La RoSA marker did not prove batched or "
                f"single-row top-k threshold, got "
                f"threshold_numels={threshold_numels}, x_shapes={x_shapes}"
            ),
        )
        require(
            inclusives == [True],
            f"{prefix} La RoSA should use inclusive top-k compare, got {inclusives}",
        )
    else:
        raise SystemExit(f"unknown method {method!r}")

ops = sparse.get("ascend_sparse_linear_marker_ops", [])
require(
    any(
        op in ops
        for op in (
            "activation_sparse_linear_packed_t",
            "activation_sparse_linear_direct_t",
        )
    ),
    f"Ascend sparse custom op marker is missing, got {ops}",
)
weight_t_values = sparse.get("ascend_sparse_linear_marker_weight_t_provided", [])
require(
    True in weight_t_values,
    f"Ascend marker did not prove transposed-weight path, got {weight_t_values}",
)
print(f"{json_path}: marker metadata validated for {method}")
PY
}

check_ppl_marker_metadata() {
  local json_path="$1"
  local method="$2"
  "${PYTHON_BIN}" - "${json_path}" "${method}" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
method = sys.argv[2]
result = json.loads(json_path.read_text(encoding="utf-8"))
vllm = result.get("vllm")
if vllm is None:
    raise SystemExit(f"{json_path}: vLLM result is missing")
markers = vllm.get("sparse_kernel_markers")
if markers is None:
    raise SystemExit(f"{json_path}: sparse kernel marker summary is missing")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"{json_path}: {message}")


def marker_values(prefix: str, key: str) -> list:
    return markers.get(f"{prefix}_marker_{key}", [])


for prefix in ("sparse_gemv", "ascend_sparse_linear"):
    require(
        markers.get(f"{prefix}_marker_records", 0) > 0,
        f"{prefix} marker records are missing",
    )
    threshold_numels = marker_values(prefix, "threshold_numels")
    x_shapes = marker_values(prefix, "x_shapes")
    inclusives = marker_values(prefix, "inclusive")
    require(threshold_numels, f"{prefix} threshold_numels are missing")
    require(x_shapes, f"{prefix} x_shapes are missing")
    require(inclusives, f"{prefix} inclusive values are missing")

    if method == "teal":
        require(
            all(int(numel) == 1 for numel in threshold_numels),
            f"{prefix} TEAL should use scalar threshold, got {threshold_numels}",
        )
        require(
            inclusives == [False],
            f"{prefix} TEAL should use exclusive compare, got {inclusives}",
        )
    elif method == "larosa":
        rows = {
            int(shape[0])
            for shape in x_shapes
            if isinstance(shape, list) and shape
        }
        require(rows, f"{prefix} La RoSA x_shapes do not expose batch rows")
        require(
            all(int(numel) in rows for numel in threshold_numels),
            (
                f"{prefix} La RoSA should use one threshold per row, got "
                f"threshold_numels={threshold_numels}, x_shapes={x_shapes}"
            ),
        )
        require(
            any(int(numel) > 1 for numel in threshold_numels)
            or any(
                isinstance(shape, list)
                and shape
                and int(shape[0]) == 1
                for shape in x_shapes
            ),
            (
                f"{prefix} La RoSA marker did not prove batched or "
                f"single-row top-k threshold, got "
                f"threshold_numels={threshold_numels}, x_shapes={x_shapes}"
            ),
        )
        require(
            inclusives == [True],
            f"{prefix} La RoSA should use inclusive top-k compare, got {inclusives}",
        )
    else:
        raise SystemExit(f"unknown method {method!r}")

ops = markers.get("ascend_sparse_linear_marker_ops", [])
require(
    any(
        op in ops
        for op in (
            "activation_sparse_linear_packed_t",
            "activation_sparse_linear_direct_t",
        )
    ),
    f"Ascend sparse custom op marker is missing, got {ops}",
)
weight_t_values = markers.get("ascend_sparse_linear_marker_weight_t_provided", [])
require(
    True in weight_t_values,
    f"Ascend marker did not prove transposed-weight path, got {weight_t_values}",
)
print(f"{json_path}: PPL marker metadata validated for {method}")
PY
}

echo "=== environment ==="
echo "VLLM_ROOT=${VLLM_ROOT}"
echo "ASCEND_ROOT=${ASCEND_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "BENCH_BATCH_SIZES=${BENCH_BATCH_SIZES}"
echo "MODEL_DTYPE=${MODEL_DTYPE}"
echo "THROUGHPUT_SPARSE_LINEAR_POLICY=${THROUGHPUT_SPARSE_LINEAR_POLICY}"
echo "PPL_SPARSE_LINEAR_POLICY=${PPL_SPARSE_LINEAR_POLICY}"
echo "VLLM_ASCEND_FORCE_CPU_SLOT_MAPPING=${VLLM_ASCEND_FORCE_CPU_SLOT_MAPPING}"
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
import torch_npu
print("torch_npu", getattr(torch_npu, "__version__", "unknown"))
print("npu_available", torch.npu.is_available())
PY

echo "=== source manifest ==="
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path


def run_git(root: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_state(root_name: str, root: Path, files: list[str]) -> dict:
    file_hashes = {}
    for file_name in files:
        path = root / file_name
        file_hashes[file_name] = {
            "exists": path.exists(),
            "sha256": sha256_file(path),
        }
    return {
        "name": root_name,
        "root": str(root),
        "branch": run_git(root, ["branch", "--show-current"]),
        "head": run_git(root, ["rev-parse", "HEAD"]),
        "status_porcelain": run_git(root, ["status", "--porcelain"]),
        "tracked_remote": run_git(
            root,
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        ),
        "files": file_hashes,
    }


vllm_root = Path(os.environ["VLLM_ROOT"]).resolve()
ascend_root = Path(os.environ["ASCEND_ROOT"]).resolve()
manifest = {
    "vllm": repo_state(
        "vllm-hust",
        vllm_root,
        [
            "scripts/bench_sparse_kernel_throughput.py",
            "scripts/eval_teal_ppl_alignment.py",
            "scripts/eval_larosa_ppl_alignment.py",
            "scripts/run_sparse_kernel_verification.sh",
            "vllm/sparsity/config.py",
            "vllm/sparsity/distribution.py",
            "vllm/sparsity/layers.py",
            "vllm/sparsity/kernels/sparse_gemv.py",
            "vllm/model_executor/models/qwen2.py",
            "vllm/model_executor/models/llama.py",
        ],
    ),
    "ascend": repo_state(
        "vllm-ascend-hust",
        ascend_root,
        [
            "csrc/kernels/activation_sparse_linear.cpp",
            "csrc/torch_binding.cpp",
            "csrc/torch_binding_meta.cpp",
            "csrc/ops.h",
            "vllm_ascend/compilation/passes/base_pattern.py",
            "vllm_ascend/ops/layernorm.py",
            "vllm_ascend/ops/sparse_linear.py",
            "vllm_ascend/worker/block_table.py",
            "tests/ut/ops/test_sparse_linear.py",
            "benchmarks/ops/bench_activation_sparse_linear.py",
            "benchmarks/ops/run_activation_sparse_linear_matrix.py",
        ],
    ),
}
manifest_path = Path(os.environ["SOURCE_MANIFEST"])
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "=== build vllm-ascend-hust editable plugin ==="
  (
    cd "${ASCEND_ROOT}"
    "${PYTHON_BIN}" -m pip install -v -e . --no-build-isolation --no-deps
  )
fi

echo "=== python syntax checks ==="
(
  cd "${VLLM_ROOT}"
  "${PYTHON_BIN}" -m py_compile \
    scripts/bench_sparse_kernel_throughput.py \
    scripts/eval_teal_ppl_alignment.py \
    scripts/eval_larosa_ppl_alignment.py \
    tests/sparsity/test_distribution.py \
    vllm/sparsity/distribution.py \
    vllm/sparsity/kernels/sparse_gemv.py
)
(
  cd "${ASCEND_ROOT}"
  "${PYTHON_BIN}" -m py_compile \
    vllm_ascend/ops/sparse_linear.py \
    benchmarks/ops/bench_activation_sparse_linear.py \
    benchmarks/ops/run_activation_sparse_linear_matrix.py \
    tests/ut/ops/test_sparse_linear.py
)

echo "=== custom op unit test ==="
if "${PYTHON_BIN}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pytest") is not None else 1)
PY
then
  (
    cd "${ASCEND_ROOT}"
    "${PYTHON_BIN}" -m pytest -q tests/ut/ops/test_sparse_linear.py
  )
else
  echo "pytest is unavailable; running built-in NPU sparse-linear smoke"
  (
    cd "${ASCEND_ROOT}"
    "${PYTHON_BIN}" - <<'PY'
import torch

from vllm_ascend.ops.sparse_linear import (
    _custom_op_enabled,
    activation_sparse_linear,
    activation_sparse_linear_direct,
    activation_sparse_linear_ref,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sparse_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-1, 5e-2
    return 5e-2, 5e-2


require(hasattr(torch, "npu") and torch.npu.is_available(), "NPU is required")
require(_custom_op_enabled(), "Ascend custom ops must be enabled")
torch.manual_seed(0)
for dtype in (torch.float16,):
    x = torch.randn(3, 32, device="npu", dtype=dtype)
    weight = torch.randn(17, 32, device="npu", dtype=dtype)
    threshold = torch.tensor([0.25, 0.5, 0.75], device="npu")
    actual = activation_sparse_linear(x, weight, threshold)
    direct = activation_sparse_linear_direct(x, weight, threshold)
    expected = activation_sparse_linear_ref(x, weight, threshold)
    torch.npu.synchronize()
    atol, rtol = sparse_tolerances(dtype)
    require(
        torch.allclose(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol),
        f"packed sparse linear mismatch for {dtype}",
    )
    require(
        torch.allclose(direct.cpu(), expected.cpu(), atol=atol, rtol=rtol),
        f"direct sparse linear mismatch for {dtype}",
    )

    batch_size = 8
    input_dim = 128
    output_dim = 1153
    keep = input_dim // 2
    x = torch.randn(batch_size, input_dim, device="npu", dtype=dtype)
    weight = torch.randn(output_dim, input_dim, device="npu", dtype=dtype)
    weight_t = weight.t().contiguous()
    topk_values, _ = torch.topk(x.abs().to(dtype=torch.float32), keep, dim=-1)
    threshold = topk_values[..., -1].contiguous()
    values, indices, counts = torch.ops._C_ascend.activation_sparse_pack(
        x.contiguous(),
        threshold,
        True,
    )
    packed_t = torch.ops._C_ascend.activation_sparse_linear_packed_t(
        values,
        indices,
        counts,
        weight_t,
    )
    wrapper = activation_sparse_linear(
        x,
        weight,
        threshold,
        inclusive=True,
        weight_t=weight_t,
    )
    expected = activation_sparse_linear_ref(
        x,
        weight,
        threshold,
        inclusive=True,
    )
    torch.npu.synchronize()
    require(counts.min().item() >= keep, "row_topk packed counts below keep")
    require(counts.max().item() <= input_dim, "row_topk packed counts above input")
    atol, rtol = sparse_tolerances(dtype)
    require(
        torch.allclose(packed_t.cpu(), expected.cpu(), atol=atol, rtol=rtol),
        f"packed_t row_topk mismatch for {dtype}",
    )
    require(
        torch.allclose(wrapper.cpu(), expected.cpu(), atol=atol, rtol=rtol),
        f"wrapper row_topk mismatch for {dtype}",
    )
print("built-in NPU sparse-linear smoke passed")
PY
  )
fi

echo "=== operator benchmark matrix ==="
read -r -a BENCH_BATCH_SIZE_ARRAY <<< "${BENCH_BATCH_SIZES}"
BENCH_BATCH_ARGS=()
for batch_size in "${BENCH_BATCH_SIZE_ARRAY[@]}"; do
  BENCH_BATCH_ARGS+=(--default-batch-size "${batch_size}")
done
BENCH_DIRECT_ARGS=(--skip-direct)
if [[ "${BENCH_SKIP_DIRECT}" != "1" ]]; then
  BENCH_DIRECT_ARGS=(--max-direct-err "${MAX_DIRECT_ERR}")
fi
(
  cd "${ASCEND_ROOT}"
  "${PYTHON_BIN}" benchmarks/ops/run_activation_sparse_linear_matrix.py \
    --dtype "${BENCH_DTYPE}" \
    "${BENCH_BATCH_ARGS[@]}" \
    --warmup "${BENCH_WARMUP}" \
    --iters "${BENCH_ITERS}" \
    --max-sparse-err "${MAX_SPARSE_ERR}" \
    --max-direct-t-err "${MAX_DIRECT_T_ERR}" \
    "${BENCH_DIRECT_ARGS[@]}" \
    --min-packed-total-speedup "${MIN_PACKED_TOTAL_SPEEDUP}" \
    --min-packed-total-with-threshold-speedup "${MIN_PACKED_TOTAL_WITH_THRESHOLD_SPEEDUP}" \
    --min-packed-compute-speedup "${MIN_PACKED_COMPUTE_SPEEDUP}" \
    --output-dir "${RESULT_DIR}/bench_activation_sparse_linear"
)

if [[ "${RUN_THROUGHPUT}" == "1" ]]; then
  echo "=== Qwen2.5 TEAL kernel-required throughput ==="
  (
    cd "${VLLM_ROOT}"
    VLLM_SPARSE_GEMV_LINEAR_POLICY="${THROUGHPUT_SPARSE_LINEAR_POLICY}" \
    "${PYTHON_BIN}" scripts/bench_sparse_kernel_throughput.py \
      --model "${MODEL}" \
      --method teal \
      --calibration-path "${TEAL_ARTIFACT_ROOT}/thresholds" \
      --sparsity 0.4 \
      --num-prompts "${THROUGHPUT_NUM_PROMPTS}" \
      --warmup-prompts "${THROUGHPUT_WARMUP_PROMPTS}" \
      --input-len "${THROUGHPUT_INPUT_LEN}" \
      --output-len "${THROUGHPUT_OUTPUT_LEN}" \
      --dtype "${MODEL_DTYPE}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.75}" \
      --vllm-prefill-sparsify none \
      --min-total-token-speedup "${MIN_TOTAL_TOKEN_SPEEDUP}" \
      --min-output-token-speedup "${MIN_OUTPUT_TOKEN_SPEEDUP}" \
      --json-output "${RESULT_DIR}/teal_qwen25_decode_only_sparse_gemv_throughput.json"
  )
  check_throughput_marker_metadata \
    "${RESULT_DIR}/teal_qwen25_decode_only_sparse_gemv_throughput.json" \
    teal

  echo "=== Qwen2.5 La RoSA kernel-required throughput ==="
  (
    cd "${VLLM_ROOT}"
    VLLM_SPARSE_GEMV_LINEAR_POLICY="${THROUGHPUT_SPARSE_LINEAR_POLICY}" \
    "${PYTHON_BIN}" scripts/bench_sparse_kernel_throughput.py \
      --model "${MODEL}" \
      --method larosa \
      --calibration-path "${LAROSA_ARTIFACT_ROOT}" \
      --sparsity "${LAROSA_SPARSITY:-0.40}" \
      --num-prompts "${THROUGHPUT_NUM_PROMPTS}" \
      --warmup-prompts "${THROUGHPUT_WARMUP_PROMPTS}" \
      --input-len "${THROUGHPUT_INPUT_LEN}" \
      --output-len "${THROUGHPUT_OUTPUT_LEN}" \
      --dtype "${MODEL_DTYPE}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.75}" \
      --min-total-token-speedup "${MIN_TOTAL_TOKEN_SPEEDUP}" \
      --min-output-token-speedup "${MIN_OUTPUT_TOKEN_SPEEDUP}" \
      --json-output "${RESULT_DIR}/larosa_qwen25_sparse_gemv_throughput.json"
  )
  check_throughput_marker_metadata \
    "${RESULT_DIR}/larosa_qwen25_sparse_gemv_throughput.json" \
    larosa
fi

if [[ "${RUN_QWEN_PPL}" == "1" ]]; then
  echo "=== Qwen2.5 TEAL kernel-required PPL ==="
  (
    cd "${VLLM_ROOT}"
    VLLM_SPARSE_GEMV_LINEAR_POLICY="${PPL_SPARSE_LINEAR_POLICY}" \
    "${PYTHON_BIN}" scripts/eval_teal_ppl_alignment.py \
      --model "${MODEL}" \
      --artifact-root "${TEAL_ARTIFACT_ROOT}" \
      --sparsity 0.4 \
      --dataset-name wikitext \
      --dataset-subset wikitext-2-raw-v1 \
      --dataset-split test \
      --dataset-text-field text \
      --dataset-size "${PPL_DATASET_SIZE:-100}" \
      --dataset-sample random \
      --dataset-seed 0 \
      --context-size "${PPL_CONTEXT_SIZE:-2048}" \
      --window-size "${PPL_WINDOW_SIZE:-512}" \
      --max-windows "${PPL_MAX_WINDOWS:-32}" \
      --backend both \
      --hf-reference-impl hooks \
      --hf-reference-mode decode_only \
      --vllm-prefill-sparsify none \
      --dtype "${MODEL_DTYPE}" \
      --device npu \
      --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.75}" \
      --vllm-use-sparse-gemv \
      --fail-on-mismatch \
      --json-output "${RESULT_DIR}/teal_qwen25_decode_only_sparse_gemv.json"
  )
  check_ppl_marker_metadata \
    "${RESULT_DIR}/teal_qwen25_decode_only_sparse_gemv.json" \
    teal

  echo "=== Qwen2.5 La RoSA kernel-required PPL ==="
  (
    cd "${VLLM_ROOT}"
    VLLM_SPARSE_GEMV_LINEAR_POLICY="${PPL_SPARSE_LINEAR_POLICY}" \
    "${PYTHON_BIN}" scripts/eval_larosa_ppl_alignment.py \
      --model "${MODEL}" \
      --calibration-path "${LAROSA_ARTIFACT_ROOT}" \
      --sparsity "${LAROSA_SPARSITY:-0.40}" \
      --dataset-name wikitext \
      --dataset-subset wikitext-2-raw-v1 \
      --dataset-split test \
      --dataset-text-field text \
      --dataset-size "${PPL_DATASET_SIZE:-100}" \
      --dataset-sample random \
      --dataset-seed 0 \
      --context-size "${PPL_CONTEXT_SIZE:-2048}" \
      --window-size "${PPL_WINDOW_SIZE:-512}" \
      --max-windows "${PPL_MAX_WINDOWS:-32}" \
      --backend both \
      --hf-reference-impl hooks \
      --dtype "${MODEL_DTYPE}" \
      --device npu \
      --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.75}" \
      --vllm-use-sparse-gemv \
      --fail-on-mismatch \
      --json-output "${RESULT_DIR}/larosa_qwen25_sparse_gemv.json"
  )
  check_ppl_marker_metadata \
    "${RESULT_DIR}/larosa_qwen25_sparse_gemv.json" \
    larosa
fi

echo "=== verification artifacts ==="
find "${RESULT_DIR}" -maxdepth 3 -type f | sort
