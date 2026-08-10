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
- `dynamic`: compare the median recent end-to-end cost of loading and
  recomputing. A tie chooses recompute deterministically.

Dynamic mode is conservative about data quality: only measurements for the
exact same token/block-size buckets are reused. Missing, stale, or invalid
measurements use the configured fallback. Every decision records the selected
branch, reason, predictions, observation counts/ages, measured service/wait
components, and completion cost in the connector audit log.

The dynamic confidence gate requires both branches to have fresh samples and
requires each sample to carry an explicit phase wait measurement. A missing
old calibration file therefore falls back to `load` with reason
`insufficient_observation_confidence`; it is never treated as a zero wait.

M1 dynamic estimates are valid only when no other materialization request is
active. If a second request overlaps an unfinished load or recompute attempt,
the plugin records `unsupported_concurrent_context` and uses the configured
fallback (`load` in the M1 configuration). Forced modes remain available for
matched baselines. Maintaining copy-byte or recompute-token backlog across
requests is intentionally outside the M1 scope.

The audit field `queue_wait_ms` is measured separately from service time. Its
scope is the plugin's admission-to-first-service-start interval: for load it
is scheduler decision to worker copy submission, and for recompute it is
scheduler decision to the first compute step. This is an observed runtime
admission queue, not a claim about an opaque device driver's internal queue.
`extra_wait_ms` remains a residual between scheduler-observed total time and
worker service time, and may still include dispatch, process communication,
and metadata return. The decision estimator uses the explicit phase wait
field and the service field; it does not mistake the residual for queue time.

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
