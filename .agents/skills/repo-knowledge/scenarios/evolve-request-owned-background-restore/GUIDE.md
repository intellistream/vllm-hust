# Evolve request-owned background restore

Use this scenario when changing or integrating request-owned KV H2D restore,
especially when it shares an offload adapter with background D2H drain or a
token-bearing step whose sampling is deferred.

## Accepted runtime boundary

- The scheduler may emit a cold request's `RESTORE` command in the same step
  that schedules unrelated requests. The RESTORE target itself must carry no
  authorization token until a terminal `pending_dma=0` receipt has applied.
- The device runner zeros the complete final destination and synchronizes that
  writer before `RequestOwnedBulkRestoreWork.execute_after_zero()` submits H2D.
  Submission does not wait. The wrapper runs unrelated forward and sampling,
  then `RequestOwnedRestoreGuard.finish()` waits for the exact job, marks the
  exact allocation generation HOT, and replaces the buffered restore intent
  with its terminal receipt.
- A zero-token restore naturally falls back to a same-call exact wait. A
  token-bearing split-sampling step transfers its restore guard into
  `_RequestOwnedDeferredStep`; the executor's real grammar payload must reach
  `sample_tokens()` unchanged before the H2D receipt is closed.
- Rollback must abort the adapter identity, wait and consume any submitted H2D
  completion, and only then recycle its destination. If exact DMA quiescence
  cannot be proven, retain the destination. Once a nonempty step was built or
  executed, any restore-path failure is fail-stop rather than retryable.
- `RequestOwnedBulkOffloadAdapter.poll_jobs()` may select exact restore jobs,
  but it must hold unrelated already-processed receipts without replaying
  their host-image side effects. This is the compatibility seam for the D2H
  drain controller, which owns STORE receipts.

## Source and verification anchors

- Runtime lifecycle:
  `vllm/v1/worker/request_owned_restore_runtime.py`
- Wrapper ordering and deferred sampling:
  `vllm/v1/worker/worker_base.py`
- Shared receipt selection:
  `vllm/v1/worker/request_owned_offload.py`
- Deterministic lifecycle tests:
  `tests/v1/worker/test_request_owned_background_restore.py`
- Real scheduler co-issue proof:
  `tests/v1/core/test_scheduler_background_restore.py`

Run the focused CPU gate from the repository root:

```bash
.venv/bin/python -m pytest -q \
  tests/v1/core/test_scheduler_background_restore.py \
  tests/v1/worker/test_request_owned_background_restore.py \
  tests/v1/worker/test_request_owned_offload.py \
  tests/v1/worker/test_request_owned_deferred_sampling.py
```

The paired Ascend runner owns only the packed-zero/device-stream seam. Its
focused wiring test is
`tests/ut/worker/test_request_owned_runner_packed_zero_wiring.py`; NPU overlap
or performance remains unproven until a separately admitted optimized probe.
