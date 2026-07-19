#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-a320m2tn910cu}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-120}"
REQUEST_NUM="${REQUEST_NUM:-200}"
RESULT_ROOT="${SCRIPT_DIR}/results/gate/raw"
SBATCH_COMMON=(
  --parsable
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=32
  --gres=npu:8
  --partition="${PARTITION}"
  --output="${RESULT_ROOT}/slurm-%j.out"
)

submit() {
  local model=$1
  local mode=$2
  sbatch "${SBATCH_COMMON[@]}" \
    --job-name="gate-${model#qwen3_}-${mode}" \
    --export="ALL,REQUEST_NUM=${REQUEST_NUM},RESULT_ROOT=${RESULT_ROOT},RUN_TAG=gate${REQUEST_NUM}" \
    --wrap="${SCRIPT_DIR}/run_experiment.sh ${model} conversation ${mode}"
}

wait_for_jobs() {
  local ids_csv
  ids_csv=$(IFS=,; echo "$*")
  while squeue --noheader --jobs="${ids_csv}" 2>/dev/null | rg -q .; do
    sleep 20
  done
}

mkdir -p "${RESULT_ROOT}"
for model in qwen3_32b qwen3_235b; do
  jobs=("$(submit "${model}" baseline)")
  sleep "${STARTUP_STAGGER_SECONDS}"
  jobs+=("$(submit "${model}" pp_opt)")
  echo "Submitted ${model} gate: ${jobs[*]}"
  wait_for_jobs "${jobs[@]}"
done

python "${SCRIPT_DIR}/analyze_results.py" \
  --raw-dir "${RESULT_ROOT}" \
  --summary "${SCRIPT_DIR}/results/gate/summary.csv" \
  --comparison "${SCRIPT_DIR}/results/gate/comparison.csv" \
  --throughput "${SCRIPT_DIR}/results/gate/throughput.csv"
python "${SCRIPT_DIR}/check_gate.py" \
  "${SCRIPT_DIR}/results/gate/comparison.csv"
