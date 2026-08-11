# Prefix Routing Runtime Fault Evidence

This note records the maintenance status for PR #225, `Harden prefix routing
runtime faults`.

The PR is a correctness and recovery patch for distributed prefix-cache
routing. It does not make P99, throughput, NPU performance, or deployment
reliability claims while the Ascend benchmark gate is unstable.

## Ascend Benchmark Gate Status

Public PR and check metadata for commit
`1d4c7924fcc489d62d52299be9fead8fbe6aae4a` show:

| Check | Status |
| --- | --- |
| `pre-commit` | success |
| `linux-ascend-inference-smoke` | success |
| `linux-ascend-inference-regression` | success |
| `ascend-benchmark` | failure |

The benchmark bot comment reported:

| Field | Value |
| --- | --- |
| Workflow run | `31072433562` |
| Scenario | `random-online` |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Publish mode | `artifact-preview` |
| Perfgate mode | `report` |
| Baseline source | `unavailable` |
| Stage 1 baseline | `e4ce33646f2ef1781289e6dc651fad0d00177c55`, source `unavailable` |
| Failed requests | `0` |

Failure reason: the Ascend benchmark gate did not have an available baseline
source and stayed in an unstable artifact-preview/report state. Therefore the
benchmark output is not a stable comparison gate and must not be used to claim
performance or reliability. The runtime correctness checks below are separate
host/runtime evidence and do not close the benchmark gate.

## Runtime Mapping Under Test

The PR maps the parent fault model to runtime fields:

| Fault-model field | Runtime mapping |
| --- | --- |
| view epoch / expiry | `view_epoch`, `expires_at`, `last_receipt_at` on prefix-cache node state |
| worker incarnation | HTTP event uploader header and ZMQ `publisher_epoch` |
| cache-state receipt | `BlockStored`, snapshots, and heartbeat-only recovery receipts |
| transfer commit marker | post-commit `BlockStored`; staged transfer state is not routable |

Routing is fail-closed: expired views, changed incarnations, cache clears,
missing receipts, and uncommitted transfer state are not eligible for remote
HIT routing.

## Correctness And Recovery Evidence

Validated locally on the PR head with:

```bash
VLLM_PLUGINS= python -m ruff check \
  vllm/distributed/prefix_scheduler.py \
  vllm/entrypoints/openai/prefix_routing.py \
  vllm/distributed/kv_events.py \
  tests/distributed/test_prefix_scheduler.py \
  tests/distributed/test_prefix_routing_e2e.py

VLLM_PLUGINS= python -m pytest -q \
  tests/distributed/test_prefix_scheduler.py \
  tests/distributed/test_prefix_routing_e2e.py
```

Result on 2026-08-11 UTC:

```text
ruff: All checks passed.
pytest: 81 passed.
```

Fault matrix:

| Fault | Injection / condition | Oracle | Tests |
| --- | --- | --- | --- |
| stale view | receipt expires before routing | no remote HIT; local fail-closed path | `test_global_prefix_scheduler_expires_stale_receipt_before_routing`, `test_prefix_routing_proxy_fails_closed_when_cached_decision_is_stale` |
| worker restart | worker incarnation / publisher epoch changes | old cache view invalidated before reuse | `test_global_prefix_scheduler_clears_cache_state_on_worker_incarnation_change`, `test_prefix_routing_invalidates_stale_view_on_epoch_mismatch_without_snapshot`, `test_prefix_routing_dual_worker_restart_and_commit_recovery_network_e2e` |
| heartbeat / receipt loss | no fresh receipt until TTL expiry | no stale remote HIT after expiry | `test_prefix_cache_upload_periodically_reconciles_idle_state`, `test_prefix_cache_upload_recovers_after_disconnect`, `test_global_prefix_scheduler_expires_stale_receipt_before_routing` |
| cache loss | `AllBlocksCleared` event | old prefix hashes removed; MISS/fallback | `test_global_prefix_scheduler_cache_loss_event_fails_closed` |
| partial transfer | transfer interrupted before commit | no HIT until committed `BlockStored` | `test_prefix_routing_partial_transfer_waits_for_commit_network_e2e` |
| commit recovery | committed `BlockStored` after interruption | HIT restored only after commit publication | `test_prefix_routing_dual_worker_restart_and_commit_recovery_network_e2e` |
| duplicate retry | same request ID retried | reuse the same still-current decision; stale decisions fail closed | `test_prefix_routing_proxy_reuses_request_id_decision`, `test_prefix_routing_duplicate_request_id_stale_decision_e2e`, `test_prefix_routing_duplicate_request_id_preserves_local_fallback_network_e2e` |
| pre-response transport failure | remote connection fails before response headers | one local fallback; no reroute to another remote for the same request ID | `test_prefix_routing_falls_back_locally_when_upstream_cannot_start`, `test_prefix_routing_dual_worker_transport_failure_falls_back_once_network_e2e` |
| stream-started interruption | failure after response streaming starts | do not restart or duplicate response | `test_prefix_routing_stream_failure_does_not_restart_response`, `test_prefix_routing_stream_failure_network_e2e` |
| malformed control-plane input | bad ZMQ / HTTP ingest messages | reject or isolate invalid messages without poisoning routing state | `test_prefix_routing_zmq_subscriber_isolates_invalid_messages`, `test_prefix_routing_http_event_ingest_rejects_invalid_token`, `test_prefix_routing_http_event_ingest_rejects_oversized_batch` |

## Stop Condition

Do not use this PR to claim performance, P99, throughput, or deployment
reliability until the benchmark gate is stable and has an available baseline
source. Until then, the supported claim is limited to: the listed
host/runtime fault-injection tests pass on the PR head and exercise the
fail-closed routing semantics above.
