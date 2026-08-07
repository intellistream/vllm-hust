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
- `dynamic`: compare recent copy queue/service time with recent recompute
  queue/service time. A tie chooses recompute deterministically.

Dynamic mode is conservative about data quality: missing, stale, or invalid
measurements use the configured fallback. Every decision records the selected
branch, reason, predictions, and completion cost in the connector audit log.

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
scheduler block; it enables the bandwidth estimate when only bandwidth-based
copy telemetry is available. `0` keeps the direct copy-service estimate.

The package has no install-time dependency on a PyPI vLLM release. Install it
in the same environment as the checked-out `vllm-hust` tree, or add its
`src/` directory to `PYTHONPATH` as the parent repository's experiment runner
does.

CPU-only tests can be run from this directory with:

```bash
uv run --with pytest --with pytest-cov pytest -q
```
