#!/usr/bin/env bash

set -eo pipefail

MODEL_KEY="${1:?model is required: qwen3_32b or qwen3_235b}"
TRACE_KEY="${2:?trace is required: conversation or burstgpt}"
MODE="${3:?mode is required: baseline or pp_opt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${WORKSPACE_ROOT}/.venv-pp-opt}"
ASCEND_REPO="${ASCEND_REPO:-${WORKSPACE_ROOT}/vllm-ascend-hust}"
ATB_ENV="${ATB_ENV:-${WORKSPACE_ROOT}/.cache/nnal/user_install2/nnal/atb/set_env.sh}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results/pp4tp2/raw}"
REQUEST_NUM="${REQUEST_NUM:-1000}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
PORT="${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 10000))}"
PIPELINE_PROFILE_SECONDS="${PIPELINE_PROFILE_SECONDS:-0}"
PIPELINE_PROFILE_DELAY="${PIPELINE_PROFILE_DELAY:-15}"
PIPELINE_PROFILE_STOP_AFTER_CAPTURE="${PIPELINE_PROFILE_STOP_AFTER_CAPTURE:-0}"
USE_DECODE_BENCH_CONNECTOR="${USE_DECODE_BENCH_CONNECTOR:-1}"
DECODE_BENCH_FILL_MEAN="${DECODE_BENCH_FILL_MEAN:-0.0}"
DECODE_BENCH_FILL_STD="${DECODE_BENCH_FILL_STD:-0.15}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-1}"
BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
STORE_OUTPUT_TOKEN_IDS="${STORE_OUTPUT_TOKEN_IDS:-0}"

for path in "${CANN_ENV}" "${VENV_DIR}/bin/activate" "${ATB_ENV}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required environment file is missing: ${path}" >&2
    exit 2
  fi
done

source "${CANN_ENV}"
source "${VENV_DIR}/bin/activate"
source "${ATB_ENV}" --cxx_abi=1
set -u
unset TASK_QUEUE_ENABLE

case "${BATCH_INVARIANT}" in
  0|1) ;;
  *)
    echo "BATCH_INVARIANT must be 0 or 1" >&2
    exit 2
    ;;
esac
export VLLM_BATCH_INVARIANT="${BATCH_INVARIANT}"

case "${STORE_OUTPUT_TOKEN_IDS}" in
  0|1) ;;
  *)
    echo "STORE_OUTPUT_TOKEN_IDS must be 0 or 1" >&2
    exit 2
    ;;
esac

source "${SCRIPT_DIR}/load_config.sh"
load_pp_opt_config "${MODEL_KEY}"
MODEL_WEIGHT_INDEX_PATH="${MODEL_DIR}/model.safetensors.index.json"
MODEL_WEIGHT_INDEX_SHA256=$(
  if [[ -f "${MODEL_WEIGHT_INDEX_PATH}" ]]; then
    sha256sum "${MODEL_WEIGHT_INDEX_PATH}" | awk '{print $1}'
  fi
)
PP_OPT_DYNAMIC_MICROBATCHES="${PP_OPT_DYNAMIC_MICROBATCHES_OVERRIDE:-${PP_OPT_DYNAMIC_MICROBATCHES}}"
PP_OPT_OVERLAP_SENDS="${PP_OPT_OVERLAP_SENDS_OVERRIDE:-${PP_OPT_OVERLAP_SENDS}}"
KV_CACHE_MEMORY=""
ENFORCE_EAGER=0
MODEL_LOAD_FORMAT="${MODEL_LOAD_FORMAT:-dummy}"

case "${TRACE_KEY}" in
  conversation)
    TRACE_FILE="${SCRIPT_DIR}/conversation_trace.csv"
    python "${SCRIPT_DIR}/prepare_conversation_trace.py" --output "${TRACE_FILE}"
    TRACE_ARGS=(--ignore-timestamps)
    ;;
  burstgpt)
    TRACE_FILE="${SCRIPT_DIR}/BurstGPT_1.csv"
    python "${SCRIPT_DIR}/prepare_burstgpt.py" --output "${TRACE_FILE}"
    TRACE_ARGS=()
    ;;
  *)
    echo "Unsupported trace: ${TRACE_KEY}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  baseline)
    export VLLM_USE_PP_OPT_SCHEDULER=0
    unset VLLM_PP_OPT_COST_MODEL_PATH
    unset VLLM_PP_OPT_BATCH_QUEUE_SIZE
    unset VLLM_PP_OPT_DYNAMIC_MICROBATCHES
    unset VLLM_PP_OPT_MIN_MICROBATCHES
    unset VLLM_PP_OPT_TARGET_MICROBATCH_SIZE
    unset VLLM_PP_OPT_OVERLAP_SENDS
    unset VLLM_PP_OPT_MONITOR_INTERVAL
    ;;
  pp_opt)
    if [[ ! -f "${COST_MODEL_PATH}" ]]; then
      echo "Calibrated cost model is required for ${CONFIG_KEY}: ${COST_MODEL_PATH}" >&2
      echo "Run benchmarks/pp_opt/run_calibration.sh ${MODEL_KEY} first." >&2
      exit 2
    fi
    jq -e \
      --arg config_key "${CONFIG_KEY}" \
      --arg config_sha256 "${CONFIG_SHA256}" \
      --arg model_config_sha256 "${MODEL_CONFIG_SHA256}" \
      --argjson pp_size "${PP_SIZE}" \
      --argjson tp_size "${TP_SIZE}" \
      '.metadata.config_key == $config_key
       and .metadata.config_sha256 == $config_sha256
       and .metadata.model_config_sha256 == $model_config_sha256
       and .metadata.pipeline_parallel_size == $pp_size
       and .metadata.tensor_parallel_size == $tp_size' \
      "${COST_MODEL_PATH}" >/dev/null || {
        echo "Cost model does not match benchmark configuration ${CONFIG_KEY}: ${COST_MODEL_PATH}" >&2
        exit 2
      }
    export VLLM_USE_PP_OPT_SCHEDULER=1
    export VLLM_PP_OPT_COST_MODEL_PATH="${COST_MODEL_PATH}"
    export VLLM_PP_OPT_BATCH_QUEUE_SIZE="${PP_OPT_BATCH_QUEUE_SIZE}"
    export VLLM_PP_OPT_DYNAMIC_MICROBATCHES="${PP_OPT_DYNAMIC_MICROBATCHES}"
    export VLLM_PP_OPT_MIN_MICROBATCHES="${PP_OPT_MIN_MICROBATCHES}"
    export VLLM_PP_OPT_TARGET_MICROBATCH_SIZE="${PP_OPT_TARGET_MICROBATCH_SIZE}"
    export VLLM_PP_OPT_OVERLAP_SENDS="${PP_OPT_OVERLAP_SENDS}"
    export VLLM_PP_OPT_MONITOR_INTERVAL=0
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

export VLLM_PP_LAYER_PARTITION="${PP_LAYER_PARTITION}"

KV_TRANSFER_ARGS=()
if [[ "${USE_DECODE_BENCH_CONNECTOR}" == "1" ]]; then
  KV_TRANSFER_CONFIG=$(jq -cn \
    --argjson fill_mean "${DECODE_BENCH_FILL_MEAN}" \
    --argjson fill_std "${DECODE_BENCH_FILL_STD}" \
    '{
      kv_connector: "DecodeBenchConnector",
      kv_role: "kv_both",
      kv_connector_extra_config: {
        fill_mean: $fill_mean,
        fill_std: $fill_std
      }
    }')
  KV_TRANSFER_ARGS+=(
    --kv-transfer-config
    "${KV_TRANSFER_CONFIG}"
  )
elif [[ "${USE_DECODE_BENCH_CONNECTOR}" != "0" ]]; then
  echo "USE_DECODE_BENCH_CONNECTOR must be 0 or 1" >&2
  exit 2
fi

MODEL_LOADING_ARGS=()
case "${MODEL_LOAD_FORMAT}" in
  dummy)
    MODEL_LOADING_ARGS+=(
      --load-format dummy
      --model-loader-extra-config
      "{\"embedding_weight_path\":\"${EMBEDDING_SHARD}\",\"embedding_weight_name\":\"model.embed_tokens.weight\",\"share_dummy_weights\":${SHARE_DUMMY_WEIGHTS}}"
    )
    ;;
  auto)
    if [[ ! -f "${MODEL_WEIGHT_INDEX_PATH}" ]]; then
      echo "Real-weight index is required: ${MODEL_WEIGHT_INDEX_PATH}" >&2
      exit 2
    fi
    MODEL_LOADING_ARGS+=(--load-format auto)
    ;;
  *)
    echo "MODEL_LOAD_FORMAT must be dummy or auto" >&2
    exit 2
    ;;
esac

if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "${NPU_COUNT}" ]]; then
  echo "Expected ${NPU_COUNT} NPUs, got SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}" >&2
  exit 2
fi

for path in "${MODEL_DIR}/config.json" "${EMBEDDING_SHARD}" "${TRACE_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file is missing: ${path}" >&2
    exit 2
  fi
done

python - "${REPO_ROOT}" "${ASCEND_REPO}" <<'PY'
import importlib.metadata
import json
import pathlib
import sys
import sysconfig
import urllib.parse

import vllm
import vllm_ascend

expected = [pathlib.Path(value).resolve() for value in sys.argv[1:]]
actual = [pathlib.Path(vllm.__file__).resolve(), pathlib.Path(vllm_ascend.__file__).resolve()]
for package_path, source_root in zip(actual, expected):
    if source_root not in package_path.parents:
        raise SystemExit(f"non-editable package path: {package_path}; expected {source_root}")

site_paths = {
    sysconfig.get_path("purelib"),
    sysconfig.get_path("platlib"),
}
distributions = {
    dist.metadata["Name"].lower(): dist
    for dist in importlib.metadata.distributions(path=sorted(site_paths))
}
for distribution_name, source_root in zip(
    ("vllm", "vllm-ascend-hust"), expected
):
    dist = distributions.get(distribution_name)
    direct_url_text = dist.read_text("direct_url.json") if dist else None
    if not direct_url_text:
        raise SystemExit(f"{distribution_name} is not installed editable")
    direct_url = json.loads(direct_url_text)
    installed_source = pathlib.Path(
        urllib.parse.unquote(urllib.parse.urlparse(direct_url["url"]).path)
    ).resolve()
    if not direct_url.get("dir_info", {}).get("editable"):
        raise SystemExit(f"{distribution_name} is not installed editable")
    if installed_source != source_root:
        raise SystemExit(
            f"{distribution_name} editable source is {installed_source}; "
            f"expected {source_root}"
        )
    print(f"editable {distribution_name}: {installed_source}")
PY

CORE_GIT_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
ASCEND_GIT_SHA=$(git -C "${ASCEND_REPO}" rev-parse HEAD)
CORE_GIT_STATUS=$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)
ASCEND_GIT_STATUS=$(
  git -C "${ASCEND_REPO}" status --porcelain --untracked-files=normal
)
CORE_GIT_CLEAN=$([[ -z "${CORE_GIT_STATUS}" ]] && echo 1 || echo 0)
ASCEND_GIT_CLEAN=$([[ -z "${ASCEND_GIT_STATUS}" ]] && echo 1 || echo 0)
case "${REQUIRE_CLEAN_GIT}" in
  0) ;;
  1)
    if [[ "${CORE_GIT_CLEAN}" != "1" || "${ASCEND_GIT_CLEAN}" != "1" ]]; then
      echo "Formal benchmark requires clean core and Ascend worktrees" >&2
      exit 2
    fi
    ;;
  *)
    echo "REQUIRE_CLEAN_GIT must be 0 or 1" >&2
    exit 2
    ;;
esac

export VLLM_PLUGINS=ascend
export VLLM_LOG_STATS_INTERVAL=1
export VLLM_ASCEND_USE_MODELSCOPE=0
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=0
export VLLM_ASCEND_ENABLE_MLA_PROLOGUE=0
export VLLM_ASCEND_FORCE_CPU_SLOT_MAPPING=1
export VLLM_ASCEND_FORCE_NATIVE_ROPE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_ENABLE_MC2=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=1
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export VLLM_HF_OVERRIDES="{\"max_position_embeddings\":${MAX_MODEL_LEN}}"

SERVER_PROFILE_ARGS=()
if (( PIPELINE_PROFILE_SECONDS > 0 )); then
  export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
  export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
  PIPELINE_PROFILE_DIR="${PIPELINE_PROFILE_DIR:-${RESULT_ROOT}/profiles/${MODEL_KEY}-${TRACE_KEY}-${MODE}}"
  mkdir -p "${PIPELINE_PROFILE_DIR}"
  SERVER_PROFILE_ARGS+=(
    --profiler-config
    "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PIPELINE_PROFILE_DIR}\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
  )
fi

EXPERIMENT_ID="${MODEL_KEY}-${TRACE_KEY}-${MODE}${RUN_TAG:+-${RUN_TAG}}"
RUN_DIR="${RESULT_ROOT}/${EXPERIMENT_ID}"
mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/server.log"
CLIENT_LOG="${RUN_DIR}/client.log"
CLIENT_STDOUT="${RUN_DIR}/client.stdout.log"
CLIENT_RESULTS="${RUN_DIR}/client_results.json"
METADATA="${RUN_DIR}/metadata.json"

export EXPERIMENT_ID MODEL_KEY TRACE_KEY MODE MODEL_NAME PP_SIZE TP_SIZE
export KV_CACHE_SIZE REQUEST_NUM PORT GPU_MEMORY_UTILIZATION MAX_MODEL_LEN
export MAX_NUM_SEQS
export KV_CACHE_MEMORY ENFORCE_EAGER MAX_BATCHED_TOKENS TRACE_FILE MODEL_DIR
export MODEL_DIR_OVERRIDE MODEL_LOAD_FORMAT
export SHARE_DUMMY_WEIGHTS CONFIG_KEY CONFIG_PATH CONFIG_SHA256
export MODEL_CONFIG_PATH MODEL_CONFIG_SHA256 DEPLOYMENT HARDWARE
export MODEL_WEIGHT_INDEX_PATH MODEL_WEIGHT_INDEX_SHA256
export CORE_GIT_SHA ASCEND_GIT_SHA CORE_GIT_CLEAN ASCEND_GIT_CLEAN
export REQUIRE_CLEAN_GIT BATCH_INVARIANT VLLM_BATCH_INVARIANT
export STORE_OUTPUT_TOKEN_IDS
export PP_LAYER_PARTITION COST_MODEL_PATH PP_OPT_BATCH_QUEUE_SIZE
export PP_OPT_DYNAMIC_MICROBATCHES PP_OPT_MIN_MICROBATCHES
export PP_OPT_TARGET_MICROBATCH_SIZE
export PP_OPT_OVERLAP_SENDS
export USE_DECODE_BENCH_CONNECTOR
export DECODE_BENCH_FILL_MEAN DECODE_BENCH_FILL_STD
python - "${METADATA}" <<'PY'
import json
import os
import platform
import sys
import time

keys = (
    "EXPERIMENT_ID", "MODEL_KEY", "TRACE_KEY", "MODE", "MODEL_NAME",
    "PP_SIZE", "TP_SIZE", "KV_CACHE_SIZE", "REQUEST_NUM", "PORT",
    "GPU_MEMORY_UTILIZATION", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
    "KV_CACHE_MEMORY", "ENFORCE_EAGER", "MODEL_DIR_OVERRIDE",
    "MODEL_LOAD_FORMAT",
    "MAX_BATCHED_TOKENS", "SHARE_DUMMY_WEIGHTS",
    "CONFIG_KEY", "CONFIG_PATH", "CONFIG_SHA256", "MODEL_CONFIG_PATH",
    "MODEL_CONFIG_SHA256", "MODEL_WEIGHT_INDEX_PATH",
    "MODEL_WEIGHT_INDEX_SHA256", "DEPLOYMENT", "HARDWARE",
    "CORE_GIT_SHA", "ASCEND_GIT_SHA", "CORE_GIT_CLEAN",
    "ASCEND_GIT_CLEAN", "REQUIRE_CLEAN_GIT", "BATCH_INVARIANT",
    "VLLM_BATCH_INVARIANT",
    "PP_LAYER_PARTITION", "COST_MODEL_PATH", "PP_OPT_BATCH_QUEUE_SIZE",
    "PP_OPT_DYNAMIC_MICROBATCHES", "PP_OPT_MIN_MICROBATCHES",
    "PP_OPT_TARGET_MICROBATCH_SIZE", "PP_OPT_OVERLAP_SENDS",
    "VLLM_USE_PP_OPT_SCHEDULER", "VLLM_PP_OPT_COST_MODEL_PATH",
    "VLLM_PP_OPT_BATCH_QUEUE_SIZE", "VLLM_PP_OPT_DYNAMIC_MICROBATCHES",
    "VLLM_PP_OPT_MIN_MICROBATCHES", "VLLM_PP_OPT_TARGET_MICROBATCH_SIZE",
    "VLLM_PP_OPT_OVERLAP_SENDS",
    "VLLM_PP_LAYER_PARTITION",
    "USE_DECODE_BENCH_CONNECTOR", "DECODE_BENCH_FILL_MEAN",
    "DECODE_BENCH_FILL_STD", "STORE_OUTPUT_TOKEN_IDS",
    "TRACE_FILE", "MODEL_DIR", "SLURM_JOB_ID", "SLURM_JOB_NODELIST",
)
payload = {key.lower(): os.environ.get(key) for key in keys}
payload["hostname"] = platform.node()
payload["start_timestamp_s"] = time.time()
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(payload, output, indent=2)
PY

server_pid=""
client_pid=""
cleanup() {
  if [[ -n "${client_pid}" ]] && kill -0 "${client_pid}" 2>/dev/null; then
    kill "${client_pid}" 2>/dev/null || true
    wait "${client_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting ${EXPERIMENT_ID} on port ${PORT}; logs: ${RUN_DIR}"
SERVER_MEMORY_ARGS=()
if [[ -n "${KV_CACHE_MEMORY}" ]]; then
  SERVER_MEMORY_ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
fi
SERVER_EXECUTION_ARGS=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  SERVER_EXECUTION_ARGS+=(--enforce-eager)
fi
vllm serve "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${MODEL_NAME}" \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --pipeline-parallel-size "${PP_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${SERVER_MEMORY_ARGS[@]}" \
  "${SERVER_EXECUTION_ARGS[@]}" \
  "${KV_TRANSFER_ARGS[@]}" \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false,"fuse_qknorm_rope":false}}' \
  --safetensors-load-strategy eager \
  --hf-overrides "${VLLM_HF_OVERRIDES}" \
  "${MODEL_LOADING_ARGS[@]}" \
  --skip-tokenizer-init \
  "${SERVER_PROFILE_ARGS[@]}" \
  >"${SERVER_LOG}" 2>&1 &
server_pid=$!

health_deadline=$((SECONDS + ${HEALTH_TIMEOUT:-1800}))
health_check() {
  python - "http://127.0.0.1:${PORT}/health" <<'PY'
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(sys.argv[1], timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
}

until health_check >/dev/null 2>&1; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}" || true
    echo "Server exited before becoming healthy; see ${SERVER_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= health_deadline )); then
    echo "Server health check timed out; see ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 5
done

client_command=(python "${SCRIPT_DIR}/client.py" \
  --workload "${TRACE_FILE}" \
  --kv-cache-size "${KV_CACHE_SIZE}" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model-name "${MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --request-num "${REQUEST_NUM}" \
  --vocab-size 151936 \
  --max-workers 1024 \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --output "${CLIENT_RESULTS}" \
  --log "${CLIENT_LOG}" \
  "${TRACE_ARGS[@]}" \
  --verbose \
)
if [[ "${STORE_OUTPUT_TOKEN_IDS}" == "1" ]]; then
  client_command+=(--store-output-token-ids)
fi

if (( PIPELINE_PROFILE_SECONDS > 0 )); then
  "${client_command[@]}" >"${CLIENT_STDOUT}" 2>&1 &
  client_pid=$!
  sleep "${PIPELINE_PROFILE_DELAY}"
  curl --silent --show-error --fail -X POST \
    "http://127.0.0.1:${PORT}/start_profile" >/dev/null
  sleep "${PIPELINE_PROFILE_SECONDS}"
  curl --silent --show-error --fail -X POST \
    "http://127.0.0.1:${PORT}/stop_profile" >/dev/null
  if [[ "${PIPELINE_PROFILE_STOP_AFTER_CAPTURE}" == "1" ]]; then
    kill "${client_pid}" 2>/dev/null || true
    wait "${client_pid}" 2>/dev/null || true
    client_pid=""
    python - "${METADATA}" <<'PY'
import json
import sys
import time

with open(sys.argv[1], encoding="utf-8") as source:
    payload = json.load(source)
payload["end_timestamp_s"] = time.time()
payload["status"] = "profiled"
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(payload, output, indent=2)
PY
    echo "Captured ${EXPERIMENT_ID}; profile: ${PIPELINE_PROFILE_DIR}"
    exit 0
  fi
  wait "${client_pid}"
  client_pid=""
else
  "${client_command[@]}" >"${CLIENT_STDOUT}" 2>&1
fi

python - "${METADATA}" "${CLIENT_RESULTS}" <<'PY'
import json
import sys
import time

with open(sys.argv[1], encoding="utf-8") as source:
    payload = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    client = json.load(source)

failed_requests = int(client["statistics"]["failed_requests"])
payload["end_timestamp_s"] = time.time()
payload["status"] = "completed" if failed_requests == 0 else "failed"
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(payload, output, indent=2)

if failed_requests:
    raise SystemExit(f"client reported {failed_requests} failed requests")
PY

echo "Completed ${EXPERIMENT_ID}; results: ${CLIENT_RESULTS}"
