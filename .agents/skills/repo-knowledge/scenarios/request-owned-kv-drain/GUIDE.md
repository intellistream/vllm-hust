# Change or debug request-owned KV drain

Use this scenario when changing the worker-side PREEMPT/offload lifecycle,
investigating a stuck owner command, or composing D2H drain with RELEASE or
RESTORE.

## Preserve the three distinct fences

The upstream offloading worker already submits STORE asynchronously. The
request-owned layer must preserve three separately ordered facts:

1. A PREEMPT command may advance the worker logical manager's global owner
   command fence immediately. Delaying manager application blocks or makes
   later command sequences stale.
2. Its scheduler-facing PREEMPT receipt must stay private while D2H is in
   flight. The scheduler keeps the request command-pending during this gap.
3. Its exact device source stays allocated until the adapter produces the
   owner/epoch/generation-qualified successful STORE receipt. Only then may
   `take_reclaimable`, `RequestOwnedKVStore.preempt`, and the flush fence run;
   the released PREEMPT receipt and cache-pool snapshot come afterwards.

`RequestOwnedKVDrainController` owns this gap as `DRAINING`. Do not implement
background drain by merely removing `adapter.wait()` from `worker_base.py`:
that publishes reclaim authority before host durability.

## Poll and liveness ownership

- A token-bearing step calls the controller's nonblocking poll and lets the
  transfer stream overlap unrelated model work.
- A zero-token control step waits once for all current drains. This prevents
  command-only heartbeat spinning when there is no useful work to overlap.
- `RequestOwnedBulkOffloadAdapter.poll()` destructively consumes completions
  from its shared STORE/RESTORE worker namespace. Before the synchronous
  RESTORE path can submit and poll H2D, all background drains must be quiesced.
- A same-key RELEASE must wait for its D2H before it may invalidate adapter or
  source state. The now-superseded PREEMPT receipt is discarded; only the
  current RELEASE receipt is published.
- Any failed, mismatched, stale, or owner-ambiguous completion is fail-stop.
  Never reclaim the source or manufacture a rejected/retryable PREEMPT after
  the logical manager has committed the in-flight transition.

The scheduler wire never carries block IDs or the private `DRAINING` enum. Its
observable in-flight facts are an absent PREEMPT event, `pending_dma > 0`, and
the still-allocated post-step pool snapshot.

## Focused verification

Run from a checkout with the repository's Python environment:

```bash
.venv/bin/python -m pytest tests/v1/worker/test_request_owned_drain.py -q
.venv/bin/python -m pytest \
  tests/v1/worker/test_request_owned_offload.py \
  tests/v1/worker/test_request_owned_boundary.py -q
```

The focused drain tests must cover nonblocking token work, one-wait zero-token
liveness, delayed terminal receipt/reclaim, transfer failure without reclaim,
failure before receipt capture, and same-key RELEASE supersession. Run the
request-owned worker and scheduler suites before publication because receipt
batch decoration composes with deferred sampling and owner admission.
