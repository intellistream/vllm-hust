#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-.cache/larosa_qwen25_7b}"
RESULT_DIR="${RESULT_DIR:-${ARTIFACT_ROOT}/results}"
SPARSITY="${SPARSITY:-0.4}"
CONTEXT_SIZE="${CONTEXT_SIZE:-2048}"
WINDOW_SIZE="${WINDOW_SIZE:-512}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_SUBSET="${DATASET_SUBSET:-wikitext-2-raw-v1}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATASET_SIZE="${DATASET_SIZE:-250}"
DATASET_SAMPLE="${DATASET_SAMPLE:-random}"
DATASET_SEED="${DATASET_SEED:-0}"
CAPTURE_DATASET_SPLIT="${CAPTURE_DATASET_SPLIT:-train}"
CAPTURE_DATASET_SIZE="${CAPTURE_DATASET_SIZE:-500}"
CAPTURE_SEQUENCE_LENGTH="${CAPTURE_SEQUENCE_LENGTH:-2048}"
CAPTURE_BATCH_SIZE="${CAPTURE_BATCH_SIZE:-1}"
CAPTURE_MAX_SEQUENCES="${CAPTURE_MAX_SEQUENCES:-}"
CAPTURE_MLP="${CAPTURE_MLP:-0}"
DEVICE="${DEVICE:-npu}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
BACKEND="${BACKEND:-both}"

mkdir -p "${RESULT_DIR}"

CAPTURE_ARGS=(
  .venv/bin/python scripts/capture_larosa_activation_stats.py
  --model "${MODEL}"
  --output-path "${ARTIFACT_ROOT}"
  --dataset-name "${DATASET_NAME}"
  --dataset-subset "${DATASET_SUBSET}"
  --dataset-split "${CAPTURE_DATASET_SPLIT}"
  --dataset-size "${CAPTURE_DATASET_SIZE}"
  --dataset-sample "${DATASET_SAMPLE}"
  --dataset-seed "${DATASET_SEED}"
  --sequence-length "${CAPTURE_SEQUENCE_LENGTH}"
  --batch-size "${CAPTURE_BATCH_SIZE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [[ -n "${CAPTURE_MAX_SEQUENCES}" ]]; then
  CAPTURE_ARGS+=(--max-sequences "${CAPTURE_MAX_SEQUENCES}")
fi

if [[ "${CAPTURE_MLP}" == "1" ]]; then
  CAPTURE_ARGS+=(--capture-mlp)
fi

"${CAPTURE_ARGS[@]}"

EVAL_ARGS=(
  .venv/bin/python scripts/eval_larosa_ppl_alignment.py
  --model "${MODEL}"
  --calibration-path "${ARTIFACT_ROOT}"
  --sparsity "${SPARSITY}"
  --context-size "${CONTEXT_SIZE}"
  --window-size "${WINDOW_SIZE}"
  --backend "${BACKEND}"
  --hf-reference-impl hooks
  --hf-attn-implementation "${ATTN_IMPLEMENTATION}"
  --dataset-name "${DATASET_NAME}"
  --dataset-subset "${DATASET_SUBSET}"
  --dataset-split "${DATASET_SPLIT}"
  --dataset-size "${DATASET_SIZE}"
  --dataset-sample "${DATASET_SAMPLE}"
  --dataset-seed "${DATASET_SEED}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --rotation-dtype float32
  --json-output "${RESULT_DIR}/larosa_qwen25_s${SPARSITY}.json"
)

if [[ -n "${MAX_WINDOWS:-}" ]]; then
  EVAL_ARGS+=(--max-windows "${MAX_WINDOWS}")
fi

"${EVAL_ARGS[@]}"
