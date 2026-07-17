#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-a320m2tn910cu}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-120}"
RESULT_ROOT="${SCRIPT_DIR}/results/pp4tp2/raw"
SBATCH_COMMON=(
  --parsable
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=32
  --partition="${PARTITION}"
)

submit() {
  local npu_count=$1
  local model=$2
  local trace=$3
  local mode=$4
  local name="pp-${model#qwen3_}-${trace:0:4}-${mode}"
  sbatch "${SBATCH_COMMON[@]}" \
    --job-name="${name}" \
    --gres="npu:${npu_count}" \
    --output="${RESULT_ROOT}/slurm-%j.out" \
    --export="ALL,RESULT_ROOT=${RESULT_ROOT}" \
    --wrap="${SCRIPT_DIR}/run_experiment.sh ${model} ${trace} ${mode}"
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
  for trace in conversation burstgpt; do
    jobs=("$(submit 8 "${model}" "${trace}" baseline)")
    sleep "${STARTUP_STAGGER_SECONDS}"
    jobs+=("$(submit 8 "${model}" "${trace}" pp_opt)")
    echo "Submitted ${model} ${trace} wave: ${jobs[*]}"
    wait_for_jobs "${jobs[@]}"
  done
done

echo "All matrix jobs left the Slurm queue. Run analyze_results.py next."
