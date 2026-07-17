# PP Optimization Design

## Motivation

The default pipeline queue sends one large decode batch through the PP stages.
With variable request context lengths and stage costs, device work is often
serialized and later stages wait for earlier stages. The optimization divides
active decode requests into several cost-balanced microbatches and keeps
multiple nonblocking executions in flight so different PP stages can compute
concurrently.

## Scheduler

`VLLM_USE_PP_OPT_SCHEDULER=1` enables an isolated scheduler path. Scheduler
outputs carry a `microbatch_id` through EngineCore and the workers. The
scheduler owns:

- request admission into a bounded set of persistent microbatches;
- calibrated cost-based placement and next-microbatch selection;
- global `max_num_seqs` and KV-allocation enforcement across all microbatches;
- in-flight ownership, completion cleanup, and optional tail compaction;
- periodic request-count and estimated-cost imbalance metrics.

The rank-local cost model is:

```text
cost = p0 * requests * layers
     + p1 * requests
     + p2 * context_tokens * layers
     + p3 * context_tokens
     + p4 * layers
     + p5
```

The scheduler scores a microbatch by its maximum predicted forward cost across
PP stages. Calibration records separate forward and total models for every
rank and binds the fitted file to the model config, topology, and explicit
layer partition.

## Engine execution

EngineCore uses a queue sized independently from the backend's default
concurrent-batch count. It schedules a free microbatch, launches model
execution and sampling nonblockingly, and retires completed outputs in
pipeline order. A microbatch cannot be selected again while its prior
execution is in flight.

PP send behavior is explicit. `VLLM_PP_OPT_OVERLAP_SENDS=0` waits for the
previous send immediately before the next send. The optional overlap mode
clones and retains send buffers until their asynchronous handles complete.
The validated result uses overlap disabled because that configuration had the
best measured stability and throughput.

## Ascend integration

The companion vLLM-Ascend-HUST repository provides:

- PP send-buffer lifetime handling for queued microbatches;
- CPU slot mapping for the DecodeBench connector's externally supplied KV;
- tuple-KV-cache filling and batched fill operations;
- native rotary embedding selection for the tested Qwen models;
- current vLLM MoE runner API integration for Qwen3-235B-A22B;
- rank-local calibration timing and CANN profiler scope annotations.

The two repositories are a versioned pair and must be installed editable from
sibling source trees. The benchmark checks both import paths before starting a
server.

## Full model execution

`--load-format dummy` initializes placeholder parameters but executes the
normal model forward path on every PP stage. No transformer layer is bypassed.
The dummy loader overlays `model.embed_tokens.weight` from the first real
safetensors shard. For the 235B model only, corresponding processed parameters
in repeated layers share storage to fit the dummy model and MoE workspace on
eight NPUs; every layer still invokes its full operators.

## Measurement

Aggregate output goodput is successful generated tokens divided by client wall
time. The throughput curve uses vLLM's one-second generation-throughput log,
clipped to the interval from first request send through final completion.

For pipeline compactness, the Ascend runner emits a scope containing PP rank,
microbatch ID, request count, and context tokens. CANN device traces are
analyzed using TP rank zero for each PP stage. Reported metrics include active
stage count, stage compute duty, all-stage idle time, wave skew, and integrated
Cube utilization. Cube utilization over wall time is an MFU proxy, not formal
model FLOP utilization.

## Limitations

Cost fitting predicts rank-local operator time, not global queueing or
microbatch kernel efficiency. Splitting too aggressively can lower Cube
utilization enough to erase pipeline overlap gains. Configuration therefore
requires both calibration and an end-to-end regression gate on each target
deployment.
