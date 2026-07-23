#!/usr/bin/env bash
# shellcheck disable=SC2034

load_pp_opt_config() {
  local model_key=${1:?model key is required}
  local default_config_path
  default_config_path="${SCRIPT_DIR}/configs/${model_key}_pp4tp2_8x910c/config.json"
  CONFIG_PATH="${CONFIG_PATH:-${default_config_path}}"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Benchmark configuration is missing: ${CONFIG_PATH}" >&2
    return 2
  fi

  jq -e --arg model_key "${model_key}" '
    .model_key == $model_key
    and .pipeline_parallel_size > 1
    and .tensor_parallel_size > 0
    and .npu_count == (.pipeline_parallel_size * .tensor_parallel_size)
    and (.layer_partition | length) == .pipeline_parallel_size
    and (.layer_partition | add) > 0
    and .optimization.batch_queue_size > 0
    and .optimization.min_microbatches > 0
    and .optimization.min_microbatches <= .optimization.batch_queue_size
    and .optimization.target_microbatch_size > 0
  ' "${CONFIG_PATH}" >/dev/null || {
    echo "Invalid benchmark configuration: ${CONFIG_PATH}" >&2
    return 2
  }

  CONFIG_KEY=$(jq -er '.config_key' "${CONFIG_PATH}")
  CONFIG_SHA256=$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')
  MODEL_NAME=$(jq -er '.model_name' "${CONFIG_PATH}")
  local configured_model_dir
  configured_model_dir=$(jq -er '.model_dir' "${CONFIG_PATH}")
  MODEL_DIR="${MODEL_DIR_OVERRIDE:-${WORKSPACE_ROOT}/${configured_model_dir}}"
  MODEL_CONFIG_PATH="${MODEL_DIR}/config.json"
  MODEL_CONFIG_SHA256=$(
    if [[ -f "${MODEL_CONFIG_PATH}" ]]; then
      sha256sum "${MODEL_CONFIG_PATH}" | awk '{print $1}'
    fi
  )
  EMBEDDING_SHARD="${MODEL_DIR}/$(jq -er '.embedding_shard' "${CONFIG_PATH}")"
  DEPLOYMENT=$(jq -er '.deployment' "${CONFIG_PATH}")
  HARDWARE=$(jq -er '.hardware' "${CONFIG_PATH}")
  PP_SIZE=$(jq -er '.pipeline_parallel_size' "${CONFIG_PATH}")
  TP_SIZE=$(jq -er '.tensor_parallel_size' "${CONFIG_PATH}")
  NPU_COUNT=$(jq -er '.npu_count' "${CONFIG_PATH}")
  PP_LAYER_PARTITION=$(jq -er '.layer_partition | join(",")' "${CONFIG_PATH}")
  KV_CACHE_SIZE=$(jq -er '.kv_cache_size' "${CONFIG_PATH}")
  GPU_MEMORY_UTILIZATION=$(jq -er '.gpu_memory_utilization' "${CONFIG_PATH}")
  MAX_MODEL_LEN=$(jq -er '.max_model_len' "${CONFIG_PATH}")
  MAX_NUM_SEQS=$(jq -er '.max_num_seqs' "${CONFIG_PATH}")
  MAX_BATCHED_TOKENS=$(jq -er '.max_num_batched_tokens' "${CONFIG_PATH}")
  SHARE_DUMMY_WEIGHTS=$(jq -r '.share_dummy_weights' "${CONFIG_PATH}")
  COST_MODEL_PATH="${SCRIPT_DIR}/$(jq -er '.cost_model' "${CONFIG_PATH}")"
  PP_OPT_BATCH_QUEUE_SIZE=$(jq -er '.optimization.batch_queue_size' "${CONFIG_PATH}")
  PP_OPT_DYNAMIC_MICROBATCHES=$(jq -er '.optimization.dynamic_microbatches | if . then 1 else 0 end' "${CONFIG_PATH}")
  PP_OPT_MIN_MICROBATCHES=$(jq -er '.optimization.min_microbatches' "${CONFIG_PATH}")
  PP_OPT_TARGET_MICROBATCH_SIZE=$(jq -er '.optimization.target_microbatch_size' "${CONFIG_PATH}")
  PP_OPT_OVERLAP_SENDS=$(jq -er '.optimization.overlap_sends | if . then 1 else 0 end' "${CONFIG_PATH}")
}
