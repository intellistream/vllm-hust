# Pipeline-Parallel Optimization Benchmark

This benchmark compares the default vLLM-HUST pipeline scheduler with the
PP optimization scheduler using identical server arguments, request order,
client admission policy, and model execution. Both modes execute every
transformer layer. Dummy loading replaces checkpoint parameters, but does not
skip model operators; the real input embedding is loaded from one checkpoint
shard so prompt token lookup remains valid.

## Supported matrix

| Model | Layout | Trace | Modes | NPUs |
| --- | --- | --- | --- | ---: |
| Qwen3-32B | PP4 + TP2 | conversation | baseline, PP-opt | 8 |
| Qwen3-32B | PP4 + TP2 | BurstGPT | baseline, PP-opt | 8 |
| Qwen3-235B-A22B | PP4 + TP2 | conversation | baseline, PP-opt | 8 |
| Qwen3-235B-A22B | PP4 + TP2 | BurstGPT | baseline, PP-opt | 8 |

Each full run replays 1,000 requests with a 131,072-token model limit. The
conversation trace is submitted as fast as the client-side KV budget allows.
BurstGPT preserves the timestamps in the trace.

## Repository layout

Place the two editable repositories, virtual environment, and model metadata
under one workspace:

```text
workspace/
  .venv-pp-opt/
  models/
    Qwen3-32B/
    Qwen3-235B-A22B/
  vllm-hust/
  vllm-ascend-hust/
```

The launchers accept `VENV_DIR`, `ASCEND_REPO`, `ATB_ENV`, and `CANN_ENV` when
the workspace uses different paths.

## Environment setup

The validated environment used Python 3.11, CANN 8.5.0, the ATB runtime, and
eight Ascend 910C NPUs. Create one environment and install both repositories
in editable mode:

```bash
cd /path/to/workspace
python3.11 -m venv --system-site-packages .venv-pp-opt
source .venv-pp-opt/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /path/to/atb/set_env.sh --cxx_abi=1

VLLM_TARGET_DEVICE=empty \
  pip install --no-build-isolation -e ./vllm-hust
pip install --no-build-isolation -e ./vllm-ascend-hust
```

Do not install either package from a binary wheel afterward. Every benchmark
launch verifies that `vllm` and `vllm_ascend` resolve to these editable source
trees and fails if either installation is shadowed.

Install the benchmark analysis dependencies if they are not already present:

```bash
pip install matplotlib numpy pandas safetensors
```

## Model files

Only model metadata, tokenizer files, the weight index, and the first
safetensors shard are required. With the Hugging Face CLI:

```bash
cd /path/to/workspace
hf download Qwen/Qwen3-32B \
  --local-dir models/Qwen3-32B --exclude '*.safetensors'
hf download Qwen/Qwen3-32B \
  --local-dir models/Qwen3-32B \
  --include model-00001-of-00017.safetensors

hf download Qwen/Qwen3-235B-A22B \
  --local-dir models/Qwen3-235B-A22B --exclude '*.safetensors'
hf download Qwen/Qwen3-235B-A22B \
  --local-dir models/Qwen3-235B-A22B \
  --include model-00001-of-00118.safetensors
```

The 235B dummy configuration shares storage between corresponding parameters
in repeated transformer layers to leave enough NPU memory for MoE workspace.
This changes parameter storage only; layer count, tensor shapes, operators,
and pipeline communication are unchanged in both compared modes.

## Calibration

Calibration is mandatory for every new hardware generation, model, PP/TP
layout, layer partition, or material CANN/vLLM runtime revision. A cost model
from another deployment is not portable. The checked-in fits are valid only
for the configuration and model hashes recorded in their metadata.

The local calibration pipeline generates 48 traces, runs the full model under
rank-local timing instrumentation, fits forward and total cost models, and
installs a hash-bound result under `configs/`:

```bash
srun --job-name=cal-q32-pp4tp2 \
  -N 1 --ntasks-per-node=1 --cpus-per-task=32 \
  --gres=npu:8 --partition=a320m2tn910cu \
  benchmarks/pp_opt/run_calibration.sh qwen3_32b

srun --job-name=cal-q235-pp4tp2 \
  -N 1 --ntasks-per-node=1 --cpus-per-task=32 \
  --gres=npu:8 --partition=a320m2tn910cu \
  benchmarks/pp_opt/run_calibration.sh qwen3_235b
```

Calibration models rank-local execution cost. It does not model queueing,
communication, or the kernel-efficiency loss caused by very small
microbatches. Always run the regression gate after calibration.

## Run one comparison

From the vLLM-HUST repository root:

```bash
srun --job-name=ppopt-q32-conv-base \
  -N 1 --ntasks-per-node=1 --cpus-per-task=32 \
  --gres=npu:8 --partition=a320m2tn910cu \
  benchmarks/pp_opt/run_experiment.sh \
  qwen3_32b conversation baseline

srun --job-name=ppopt-q32-conv-opt \
  -N 1 --ntasks-per-node=1 --cpus-per-task=32 \
  --gres=npu:8 --partition=a320m2tn910cu \
  benchmarks/pp_opt/run_experiment.sh \
  qwen3_32b conversation pp_opt
```

Accepted model keys are `qwen3_32b` and `qwen3_235b`; traces are
`conversation` and `burstgpt`; modes are `baseline` and `pp_opt`.
`REQUEST_NUM=200` selects a short gate. `RESULT_ROOT`, `PORT`,
`REQUEST_TIMEOUT`, and `PARTITION` are also configurable.

The validated optimized configuration uses four static microbatches and waits
for each previous PP send before issuing the next. These settings are saved in
the model configuration and must remain identical across a reported baseline
and optimized pair except for the PP scheduler itself.

## Gate and full matrix

Run the 200-request conversation gate first:

```bash
benchmarks/pp_opt/submit_gate.sh
```

The gate exits nonzero if optimized throughput regresses for either model. Run
the eight-experiment matrix only after the gate passes:

```bash
benchmarks/pp_opt/submit_matrix.sh
```

The matrix uses two eight-NPU jobs at most and therefore reserves no more than
16 NPUs concurrently. Analyze completed runs with:

```bash
python benchmarks/pp_opt/analyze_results.py
python benchmarks/pp_opt/plot_throughput.py
```

## Expected result

The validated Qwen3-32B conversation run completed 1,000/1,000 requests in
both modes on eight 910C NPUs:

| Mode | Duration | Request throughput | Output throughput |
| --- | ---: | ---: | ---: |
| Baseline | 1,446.91 s | 0.6911 req/s | 241.45 tok/s |
| PP-opt | 1,187.73 s | 0.8419 req/s | 294.14 tok/s |
| Speedup | **1.218x** | **+21.82%** | **+21.82%** |

Treat approximately 1.1-1.3x as the expected range on the same hardware and
software stack, not as a universal target. The full two-model/two-trace matrix
has not yet been validated with the recovered static configuration; no
expected speedup is claimed for those six remaining pairs.

See [design.md](design.md) for scheduler and Ascend integration details and
[results.md](results.md) for the measured pipeline compactness evidence.
