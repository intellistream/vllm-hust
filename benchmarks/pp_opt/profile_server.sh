#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

source "${SCRIPT_DIR}/load_config.sh"
load_pp_opt_config "${MODEL_KEY:?MODEL_KEY must be set}"

export VLLM_USE_PP_OPT_SCHEDULER=0
export VLLM_PP_LAYER_PARTITION="${PP_LAYER_PARTITION}"
export VLLM_PROFILE_PP_OPT_ENABLED="${VLLM_PROFILE_PP_OPT_ENABLED:-1}"
export VLLM_PROFILE_PP_OPT_OUTPUT_PATH="${VLLM_PROFILE_PP_OPT_OUTPUT_PATH:?profile output path is required}"

if [[ -n "${VLLM_HF_OVERRIDES:-}" ]]; then
  HF_OVERRIDES="${VLLM_HF_OVERRIDES}"
else
  HF_OVERRIDES="{\"max_position_embeddings\":${MAX_MODEL_LEN}}"
fi

exec vllm serve "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT:-8000}" \
  --served-model-name "${MODEL_NAME}" \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --pipeline-parallel-size "${PP_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.0,"fill_std":0.15}}' \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false,"fuse_qknorm_rope":false}}' \
  --safetensors-load-strategy eager \
  --hf-overrides "${HF_OVERRIDES}" \
  --load-format dummy \
  --model-loader-extra-config "{\"embedding_weight_path\":\"${EMBEDDING_SHARD}\",\"embedding_weight_name\":\"model.embed_tokens.weight\",\"share_dummy_weights\":${SHARE_DUMMY_WEIGHTS}}" \
  --skip-tokenizer-init
