# Mooncake connector equivalence real-run gate

This runbook covers the core-delivered direct-transfer and distributed-store
bridges without treating the external Mooncake system as part of vLLM core.

## Evidence labels

- `preflight-only` checks the exact Python environment, Mooncake version,
  selected Bundle v1 roles, topology-specific config and executables, model,
  accelerator inventory, ports, fresh output path, and core revision. It starts
  no service and proves no connector behavior.
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
  --output-dir /new/path/tied-to-the-run
```

Store topologies additionally require `--store-config`. Embedded mode requires
a nonzero `global_segment_size`; standalone-store requires zero, plus both
`mooncake_master` and `mooncake_client` on `PATH`. The output directory must not
already exist. The command writes only `preflight.json`, installs nothing, and
does not launch, discover, reuse, or terminate a service.

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

The example runner starts every prefiller, decoder, and proxy in a separately
retained process group and performs bounded cleanup only on those children. A
future real-online runner must preserve that ownership invariant for
`mooncake_master` and `mooncake_client` as well.
