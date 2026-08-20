# KV materialization decision plugin

This package adds a vLLM connector plugin that chooses between loading a
complete CPU-resident prefix and recomputing that prefix. It is intentionally
outside the vLLM scheduler, cache hashing, and connector implementation: the
plugin subclasses `SimpleCPUOffloadConnector` through the public connector
module-path configuration.

## Modes

Set `kv_connector_module_path` to
`kv_materialization_plugin.connector.DynamicSimpleCPUOffloadConnector` and
select a mode in `kv_connector_extra_config`:

- `load`: force the native CPU KV load path (the default and cold-start
  fallback).
- `recompute`: force prefix recomputation.
- `dynamic`: compare the mean recent decision-to-path-completion latency of
  loading and recomputing. A tie chooses recompute deterministically. Queue,
  service, and residual fields are diagnostics; the decision does not add them
  together again.

Dynamic mode is conservative about data quality: only measurements for the
exact same token/block size from a matched calibration workload are reused.
All admission positions in that workload remain in one path bucket so the
mean captures batching and queueing behavior. Missing, stale, or invalid
measurements use the configured fallback. Every decision records the selected
branch, reason, predictions, both active-path counts, observation counts/ages,
measured service/wait components, and completion cost in the connector audit
log.

The dynamic confidence gate requires both branches to have fresh samples and
requires each sample to carry an explicit phase wait measurement. A missing
old calibration file therefore uses the configured fallback with reason
`insufficient_observation_confidence`; it is never treated as a zero wait.

Each completed sample still records how many requests were already active on
its selected path when admitted, but this count is diagnostic rather than a
bucket key. Greedily comparing isolated path depths is unsafe for batched
recompute: a cheap first load can look attractive even when the all-load
workload is slower. Forced load and forced recompute lifecycles therefore
calibrate one workload-wide completion mean per exact prefix size. Existing
calibration files without an active-path field remain compatible.

The audit field `queue_wait_ms` is measured separately from service time. Its
scope is the plugin's admission-to-first-service-start interval: for load it
is scheduler decision to worker copy submission, and for recompute it is
scheduler decision to the first compute step. This is an observed runtime
admission queue, not a claim about an opaque device driver's internal queue.
`extra_wait_ms` remains a residual between scheduler-observed total time and
worker service time, and may still include dispatch, process communication,
and metadata return. The decision compares the observed end-to-end total; the
explicit phase wait is a confidence requirement and diagnostic, not a term
added to the total a second time.

Recompute timing follows a request across chunked-prefill steps and completes
only after all CPU-hit prefix tokens have actually been recomputed. Worker
service time sums model-execution steps; gaps between steps remain part of the
scheduler-observed end-to-end cost.

Example extra configuration:

```json
{
  "materialization_mode": "dynamic",
  "fallback_mode": "load",
  "min_copy_samples": 3,
  "min_recompute_samples": 3,
  "max_observation_age_ms": 5000,
  "sample_window_size": 32,
  "kv_bytes_per_block": 0
}
```

## Enable the calibrated-mean decision

Put both the plugin source tree and the matching `vllm-hust` checkout on
`PYTHONPATH`, then select the external connector in `--kv-transfer-config`.
Dynamic mode also needs a telemetry file containing fresh samples for both
forced branches at the exact workload sizes that will be served:

```bash
export PYTHONPATH=/path/to/vllm-hust/plugins/kv_materialization/src:/path/to/vllm-hust

vllm serve /path/to/model \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector":"DynamicSimpleCPUOffloadConnector",
    "kv_connector_module_path":"kv_materialization_plugin.connector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "materialization_mode":"dynamic",
      "fallback_mode":"load",
      "min_copy_samples":3,
      "min_recompute_samples":3,
      "max_observation_age_ms":7200000,
      "sample_window_size":256,
      "cpu_bytes_to_use":34359738368,
      "telemetry_input_path":"/path/to/calibration.json",
      "telemetry_output_path":"/path/to/runtime-telemetry.json",
      "audit_enabled":true,
      "audit_output_path":"/path/to/audit.jsonl"
    }
  }'
```

`materialization_mode=load` and `materialization_mode=recompute` force one
branch and collect its measurements. Dynamic mode cannot infer the missing
branch during cold start: if either exact-size bucket lacks enough fresh
samples, it uses `fallback_mode`. The parent repository runner performs the
required forced-load/forced-recompute calibration, freezes the merged
telemetry, and then starts dynamic mode with that file:

```bash
python experiments/scripts/kv_materialization/run_m1.py \
  --suite --run \
  --config experiments/configs/kv_materialization/m1_saturated.json \
  --workload experiments/workloads/kv_materialization/m1_short_single_request.json \
  --model /path/to/model \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --suite-repetitions 3 \
  --calibration-repetitions 3
```

Calibration is workload-specific. Serial and concurrent workloads must use
separate calibration files, and a concurrent workload must keep the same
request-size multiset and concurrency pattern across forced and dynamic
lifecycles. Active-path counts are audited but do not alter the prediction;
the decision compares exact-size workload means.

Set `kv_bytes_per_block` when the deployment can provide the KV footprint per
scheduler block. The value is retained in telemetry for audit and later
calibration; it is not used to replace the measured end-to-end cost.

The package has no install-time dependency on a PyPI vLLM release. Install it
in the same environment as the checked-out `vllm-hust` tree, or add its
`src/` directory to `PYTHONPATH` as the parent repository's experiment runner
does.

Use the existing vLLM-HUST runtime environment for tests; do not create a
second virtual environment inside this plugin directory. From the parent
repository root, run:

```bash
python -m pytest -q upstream/vllm-hust/plugins/kv_materialization/tests
```
