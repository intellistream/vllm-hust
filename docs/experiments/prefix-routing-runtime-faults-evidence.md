# Prefix Routing Runtime Fault Evidence

This note records the maintenance status for PR #225, `Harden prefix routing
runtime faults`.

The PR is a correctness and recovery patch for distributed prefix-cache
routing. It does not make P99, throughput, NPU performance, or deployment
reliability claims while the Ascend benchmark gate is unstable.

## Evidence Identity And CI Status

Runtime code head and CI status at evidence refresh:

| Field | Value |
| --- | --- |
| Runtime code head | `869999d92ea3b8f4dff31e5266898c7b27dc265c` |
| Base branch head | `e4ce33646f2ef1781289e6dc651fad0d00177c55` |
| PR head branch | `feature/prefix-cache-routing-runtime-faults` |
| PR base branch | `feature/prefix-cache-routing-reliability` |
| Runtime-code pre-commit run | `31814024275`, success |
| Runtime-code Ascend benchmark run | `31814024034`, cancelled |
| Runtime-code Ascend smoke run | `31814024058`, cancelled |
| Runtime-code Ascend regression run | `31814024066`, cancelled |

At evidence refresh time, the runtime code head had a passing pre-commit check,
while the Ascend benchmark, smoke, and regression workflows for
`869999d92ea3b8f4dff31e5266898c7b27dc265c` were cancelled. This PR therefore
does not have passing Ascend benchmark evidence for that code head and does not
claim performance, P99, throughput, NPU deployment reliability, or
benchmark-gate stability.

This evidence file may be updated by documentation-only commits on top of the
runtime code head. Such commits must not be interpreted as new runtime
correctness evidence unless they also rerun and record the local tests below.
The only runtime claim below is local host/runtime fail-closed correctness test
coverage for code through `869999d92ea3b8f4dff31e5266898c7b27dc265c`; later
evidence-only commits do not alter runtime behavior.

## Historical CI / Benchmark Context

The following rows are historical and are retained only to explain previous
review context. They are not current-head evidence.

Historical CI status for previous PR head
`cea6c2a7d6d1b504beb9b09de0144eeeaa518618`:

| Field | Value |
| --- | --- |
| Head commit | `cea6c2a7d6d1b504beb9b09de0144eeeaa518618` |
| Pre-commit run | `31490771598`, success |
| Ascend smoke run | `31490771637`, success |
| Ascend benchmark run | `31490771716`, failure |
| Ascend regression run | `31490771691`, success |

Historical benchmark bot comment for earlier PR head
`1d4c7924fcc489d62d52299be9fead8fbe6aae4a`:

| Field | Value |
| --- | --- |
| Head commit | `1d4c7924fcc489d62d52299be9fead8fbe6aae4a` |
| Workflow run | `31072433562` |
| Scenario | `random-online` |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Publish mode | `artifact-preview` |
| Perfgate mode | `report` |
| Baseline source | `unavailable` |
| Stage 1 baseline | `e4ce33646f2ef1781289e6dc651fad0d00177c55`, source `unavailable` |
| Failed requests | `0` |

Historical failure reason: the earlier Ascend benchmark gate did not have an
available baseline source and stayed in an unstable artifact-preview/report
state. Both historical benchmark outputs above are prior-head context only.
They must not be read as current-head evidence and must not be used to claim
performance or deployment reliability. The runtime correctness checks below
are separate host/runtime evidence and do not close the current benchmark gate.

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

Validated locally for runtime code through
`869999d92ea3b8f4dff31e5266898c7b27dc265c` with:

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

Result on 2026-08-15 UTC:

```text
ruff: All checks passed.
pytest: 83 passed in 7.00s.
```

Fault matrix:

| Fault | Injection / condition | Oracle | Tests |
| --- | --- | --- | --- |
| stale view | receipt expires before routing | no remote HIT; local fail-closed path | `test_global_prefix_scheduler_expires_stale_receipt_before_routing`, `test_prefix_routing_proxy_fails_closed_when_cached_decision_is_stale` |
| worker restart | worker incarnation / publisher epoch changes | old cache view invalidated before reuse | `test_global_prefix_scheduler_clears_cache_state_on_worker_incarnation_change`, `test_global_prefix_scheduler_clears_legacy_cache_on_first_incarnation_event`, `test_prefix_routing_invalidates_stale_view_on_epoch_mismatch_without_snapshot`, `test_prefix_routing_dual_worker_restart_and_commit_recovery_network_e2e` |
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
