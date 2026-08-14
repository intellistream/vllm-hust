# KV Cache Compression

KV cache compression is an experimental, opt-in interface for platform
plugins that compact a request's KV cache after prefill. vLLM core owns block
allocation and request lifecycle changes, while the selected platform provider
owns the compression algorithm and validates its supported models, devices,
and execution modes.

The feature is disabled by default. When it is disabled, vLLM does not resolve
or instantiate a provider and uses the normal scheduler and KV cache paths.

## Configuration

Pass a versioned JSON object through `--kv-cache-compression-config`:

```bash
vllm serve <model> \
  --kv-cache-compression-config '{
    "schema_version": 1,
    "provider": "<platform-provider>",
    "provider_config": {
      "provider_option": "value"
    }
  }'
```

The equivalent Python API accepts `KVCacheCompressionConfig` through
`EngineArgs` or `VllmConfig`. Provider configuration values must be JSON
scalars. Provider-specific options and compatibility requirements are
documented by the platform plugin.

## Lifecycle

Before allocating the model KV cache, every worker reports whether it supports
the requested provider and runtime configuration. A missing provider, schema
mismatch, inconsistent worker report, or unsupported configuration stops
initialization instead of silently falling back to ordinary KV cache behavior.

After a successful compressing prefill, the worker returns a versioned plan
containing the semantic and physical token counts and the source block table.
The scheduler validates the complete plan before changing block ownership,
then sends the committed block table back to the worker. Chunked prefill emits
no plan before its final chunk.

Providers can request private destination blocks when compression must preserve
immutable prefix-cache source blocks. The scheduler admits the source and
destination capacity together and swaps block tables only after the provider
finishes materializing every layer. With asynchronous scheduling, the final
compression transaction fences only that request until the plan commits;
other requests remain schedulable.

## Limitations

Core support does not imply that a provider accepts every vLLM model, cache
layout, parallel configuration, attention backend, or scheduling feature.
Consult the provider documentation before enabling compression. Runtime
failures after compaction begins are fatal because continuing with an
ambiguous physical cache layout could produce incorrect output.

To roll back, remove `--kv-cache-compression-config` or set
`kv_cache_compression_config=None` in the Python API.
