#!/usr/bin/env bash
set -u

cd /workspace/wanyao/seas0/vllm-hust-feat-sparse-pth

export ASCEND_RT_VISIBLE_DEVICES=6
export VLLM_TARGET_DEVICE=npu
export VLLM_VERSION=0.17.2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="/workspace/wanyao/seas0/vllm-hust-feat-sparse-pth:/workspace/wanyao/seas0/vllm-ascend-hust:${PYTHONPATH:-}"

SPARSE_GEMV_ARGS=()
if [[ "${USE_SPARSE_GEMV:-0}" == "1" ]]; then
  SPARSE_GEMV_ARGS+=(--vllm-use-sparse-gemv)
fi

MODEL=".cache/ascend_sparse_experiments/tiny_llama_model"
TEAL_ROOT=".cache/ascend_sparse_experiments/teal_tiny_s0.50"
LAROSA_ROOT=".cache/ascend_sparse_experiments/larosa_tiny_identity"
RESULT_DIR=".cache/ascend_sparse_experiments/results"
mkdir -p "${RESULT_DIR}"
: > "${RESULT_DIR}/status.tsv"

run_case() {
  local name="$1"
  shift
  local log="${RESULT_DIR}/${name}.log"
  echo "=== ${name} ==="
  echo -e "${name}\tSTART" >> "${RESULT_DIR}/status.tsv"
  timeout 900 "$@" > "${log}" 2>&1
  local status=$?
  echo -e "${name}\t${status}" >> "${RESULT_DIR}/status.tsv"
  tail -40 "${log}" || true
  echo
  return 0
}

COMMON_TEAL=(
  .venv/bin/python scripts/eval_teal_ppl_alignment.py
  --model "${MODEL}"
  --artifact-root "${TEAL_ROOT}"
  --sparsity 0.5
  --context-size 32
  --window-size 16
  --max-windows 1
  --backend both
  --hf-reference-impl hooks
  --hf-attn-implementation eager
  --dtype bfloat16
  --device npu
  --vllm-gpu-memory-utilization 0.2
  "${SPARSE_GEMV_ARGS[@]}"
)

run_case teal_official_half \
  "${COMMON_TEAL[@]}" \
  --hf-reference-mode official_half \
  --vllm-prefill-sparsify half \
  --json-output "${RESULT_DIR}/teal_official_half.json"

run_case teal_all_tokens \
  "${COMMON_TEAL[@]}" \
  --hf-reference-mode all_tokens \
  --vllm-prefill-sparsify all \
  --json-output "${RESULT_DIR}/teal_all_tokens.json"

run_case teal_decode_only \
  "${COMMON_TEAL[@]}" \
  --hf-reference-mode decode_only \
  --vllm-prefill-sparsify none \
  --json-output "${RESULT_DIR}/teal_decode_only.json"

COMMON_LAROSA=(
  .venv/bin/python scripts/eval_larosa_ppl_alignment.py
  --model "${MODEL}"
  --calibration-path "${LAROSA_ROOT}"
  --context-size 32
  --window-size 16
  --max-windows 1
  --backend both
  --hf-reference-impl hooks
  --hf-attn-implementation eager
  --dtype bfloat16
  --device npu
  --vllm-gpu-memory-utilization 0.2
  "${SPARSE_GEMV_ARGS[@]}"
)

run_case larosa_s0_40 \
  "${COMMON_LAROSA[@]}" \
  --sparsity 0.40 \
  --json-output "${RESULT_DIR}/larosa_s0_40.json"

run_case larosa_s0_25 \
  "${COMMON_LAROSA[@]}" \
  --sparsity 0.25 \
  --json-output "${RESULT_DIR}/larosa_s0_25.json"

echo "=== status.tsv ==="
cat "${RESULT_DIR}/status.tsv"
