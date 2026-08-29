# Mooncake connector equivalence real-run gate

This runbook covers the core-delivered direct-transfer and distributed-store
bridges without treating the external Mooncake system as part of vLLM core.

## Evidence labels

- `preflight-only` checks the exact Python environment, Mooncake version,
  runnable vLLM CLI, selected Bundle v1 roles, topology-specific config and
  executables, model, accelerator inventory and free memory, ports, fresh
  output path, and core revision. It starts no service and proves no connector
  behavior.
- `real-online` requires live process-owned vLLM and Mooncake services, real
  requests, raw telemetry, and cleanup of only the process groups created by
  that run.
- Unit tests, imports, generated summaries, and historical logs are neither a
  real rerun nor performance evidence.

## Topology-aware preflight

Choose one closed topology: `direct`, `store-embedded`, `store-standalone`,
`combined-embedded`, or `combined-standalone`.

```bash
python examples/disaggregated/mooncake_connector/real_run_preflight.py \
  --project-root /workspace/vllm-hust \
  --model /workspace/models/Qwen3-8B \
  --python /controlled/env/bin/python \
  --topology direct \
  --accelerator nvidia \
  --expected-devices 2 \
  --minimum-free-memory-mib 20000 \
  --ports 18000 18010 18020 18998 15051 \
  --output-dir /new/path/tied-to-the-run
```

Store topologies additionally require `--store-config`. Embedded mode requires
a nonzero `global_segment_size`; standalone-store requires zero, plus both
`mooncake_master` and `mooncake_client` on `PATH`. The output directory must not
already exist. The command writes only `preflight.json`, installs nothing, and
does not launch, discover, reuse, or terminate a service.

The free-memory threshold is an admission input, not a benchmark result. Set it
from the model and engine configuration. A device that exists but is occupied
does not count as eligible. Because device memory and ports are volatile, the
direct runner repeats both checks immediately before it creates evidence or
starts a service.

## Required matched matrix

For each chosen topology, use the same core and Mooncake revisions, model,
requests, device assignment, ports, transport, cache/store sizes, service
implementation, and request order for:

1. legacy built-in connector-name configuration;
2. typed selection of the exact direct/store scheduler, worker, and telemetry
   components from `vllm-core.mooncake-bridges`;
3. rollback on a fresh process start using built-in names and no typed
   selection or extension-manifest environment.

Retain raw requests and outputs, resolved configuration, scheduler/worker/API
telemetry, transfer and store hit/miss evidence, block hashes, failures,
timeouts, recovery, shutdown logs, service logs, dependency/device inventory,
and exact revisions. Store comparisons must also cover master loss and, for
standalone mode, client/SSD-owner loss. A successful import or typed startup is
not lifecycle, rollback, hardware, or performance equivalence.

The direct example runner enforces one mode per fresh process and output
directory. For example:

```bash
MODE=legacy \
MODEL=/workspace/models/Qwen3-8B \
PYTHON_BIN=/controlled/env/bin/python \
PREFLIGHT_RECORD=/evidence/preflight/preflight.json \
OUTPUT_DIR=/evidence/mooncake-legacy \
MIN_FREE_GPU_MEMORY_MIB=20000 \
examples/disaggregated/mooncake_connector/run_mooncake_connector.sh
```

Repeat with fresh `OUTPUT_DIR` values and `MODE=typed`, then `MODE=rollback`.
The runner rejects a different checkout, model, GPU set, port set, topology, or
weaker free-memory gate than the preflight admitted. It uses a fixed workload
seed by default, retains raw stdout/stderr and configuration, detects child
death during readiness waits, starts every prefiller, decoder, and proxy in a
separate retained process group, and performs bounded cleanup only on those
children. Store runners must preserve the same ownership invariant for
`mooncake_master` and `mooncake_client`.

An attempted A100 run showed why these gates are required: an unrelated 32B
service occupied both GPUs and the proxy port. The attempted Mooncake processes
were cleaned up without touching that service; the subsequent preflight
correctly rejected both devices for insufficient free memory. This is blocked
resource-admission evidence, not a Mooncake failure or real-online result.

A later audit observed both 80 GiB A100s and the required ports free, but the
same externally managed GLM-4 32B service automatically restarted roughly one
minute later. The final-revision preflight then observed only 9062 MiB free on
each device and port 18000 occupied, and again launched nothing. Other connected
GPU hosts do not satisfy this runner's two-device local topology: the available
A6000 hosts expose one GPU each, one host is occupied, and another has an NVML
driver/library mismatch. Do not terminate or reuse those workloads and do not
weaken the two-device gate merely to produce a run.
