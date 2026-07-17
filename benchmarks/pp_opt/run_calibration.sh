#!/usr/bin/env bash

set -eo pipefail

MODEL_KEY="${1:?model is required: qwen3_32b or qwen3_235b}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${WORKSPACE_ROOT}/.venv-pp-opt}"
ASCEND_REPO="${ASCEND_REPO:-${WORKSPACE_ROOT}/vllm-ascend-hust}"
ATB_ENV="${ATB_ENV:-${WORKSPACE_ROOT}/.cache/nnal/user_install2/nnal/atb/set_env.sh}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
RUN_ID="${CALIBRATION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 10000))}"

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

source "${SCRIPT_DIR}/load_config.sh"
load_pp_opt_config "${MODEL_KEY}"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "${NPU_COUNT}" ]]; then
  echo "Expected ${NPU_COUNT} NPUs, got SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}" >&2
  exit 2
fi

for path in "${MODEL_DIR}/config.json" "${EMBEDDING_SHARD}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file is missing: ${path}" >&2
    exit 2
  fi
done

python - "${REPO_ROOT}" "${ASCEND_REPO}" <<'PY'
import pathlib
import sys

import vllm
import vllm_ascend

expected = [pathlib.Path(value).resolve() for value in sys.argv[1:]]
actual = [pathlib.Path(vllm.__file__).resolve(), pathlib.Path(vllm_ascend.__file__).resolve()]
for package_path, source_root in zip(actual, expected):
    if source_root not in package_path.parents:
        raise SystemExit(f"non-editable package path: {package_path}; expected {source_root}")
print(f"editable vllm: {actual[0]}")
print(f"editable vllm_ascend: {actual[1]}")
PY

export MODEL_KEY PORT CONFIG_PATH
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
WORKLOAD_DIR="${SCRIPT_DIR}/calibration/workloads/${CONFIG_KEY}"
RUN_ROOT="${SCRIPT_DIR}/calibration/runs"
RUN_DIR="${RUN_ROOT}/${CONFIG_KEY}/${RUN_ID}"

if [[ "${SKIP_WORKLOAD_GENERATION:-0}" != "1" ]]; then
  python "${SCRIPT_DIR}/generate_calibration_workloads.py" \
    --output-dir "${WORKLOAD_DIR}" \
    --kv-cache-size "${KV_CACHE_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --overwrite
fi

echo "Calibrating ${CONFIG_KEY}; run directory: ${RUN_DIR}"
COLLECT_ARGS=()
if [[ "${CALIBRATION_RESUME:-0}" == "1" ]]; then
  COLLECT_ARGS+=(--resume)
fi
python "${SCRIPT_DIR}/collect_calibration.py" \
  --python "${VENV_DIR}/bin/python" \
  --vllm-bin "${VENV_DIR}/bin/vllm" \
  --workload-dir "${WORKLOAD_DIR}" \
  --run-dir "${RUN_DIR}" \
  --model-name "${MODEL_NAME}" \
  --kv-cache-size "${KV_CACHE_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-workers 1024 \
  --vocab-size 151936 \
  --server-script "${SCRIPT_DIR}/profile_server.sh" \
  --client "${SCRIPT_DIR}/client.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --health-timeout "${HEALTH_TIMEOUT:-1800}" \
  "${COLLECT_ARGS[@]}" \
  "$@"

python "${SCRIPT_DIR}/fit_calibration.py" \
  --input-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/fit_result.json" \
  --model-name "${MODEL_NAME}" \
  --deployment "${DEPLOYMENT}" \
  --hardware "${HARDWARE}"

python "${SCRIPT_DIR}/install_calibration.py" \
  --fit-result "${RUN_DIR}/fit_result.json" \
  --config "${CONFIG_PATH}" \
  --model-config "${MODEL_CONFIG_PATH}" \
  --output "${COST_MODEL_PATH}" \
  --run-id "${RUN_ID}"
