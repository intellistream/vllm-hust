# Request-owned restore value contracts

This design defines the values needed by the first request-owned restore
correctness probe. It does not integrate them with scheduler admission,
worker command processing, transport receipts, or production DMA.

## Three values, three owners

`RestoreIntent/v1` is scheduler-authored and block-ID-free. It names the
request, owner, epoch, activation generation, logical token extents, phase,
first-consume deadline, urgency, and policy reason. It contains no local block
ID, byte offset, packed stride, or geometry authority.

`RestorePlan/v1` is owner-private. It binds an intent to a live allocation
generation, plan sequence, packed-geometry fingerprint, group-qualified
destination IDs, canonical spans, valid token extents, exact scheduled bytes,
and the already-reserved final footprint. It must not cross the scheduler
wire.

`RestoreCertificate/v1` is owner-authored and block-ID-free. Its explicit
status records the outcome for the exact owner/epoch/activation. Only `HOT`
can certify that intent; byte or block counts alone never infer `HOT`.
`FAILED` and `RELEASED` values explicitly remove HOT authority.

The companion demand receipt records actual newly restored blocks and bytes,
including an ordinary zero-demand activation. Logical 128-token scale remains
a separately named diagnostic proxy and is never a replacement for observed
physical demand. Receipt aggregation preserves zeros and reports activation,
wave/rank, rank, and wave distributions by phase.

## Canonical packed geometry

`block_stride_bytes` is the address pitch between packed backing rows. It is
not the meaningful image size of an arbitrary `(group, block_id)` pair. Plan
bytes are:

```text
sum(unique group-qualified destination IDs * group canonical span bytes)
```

The owner-private geometry therefore requires canonical non-overlapping
physical descriptors, aliases named exactly once, contiguous exactly tiled
group spans, dense group indices, and group spans contained by the packed
stride. A plan additionally requires disjoint cross-group destination IDs,
an exact live final-footprint reservation, valid extents derived from the
intent prefix, and structural rebinding of every job to the current geometry.

The stale fence covers owner, epoch, activation generation, allocation
generation, plan sequence, and geometry fingerprint. Unknown actual demand is
an error; neither prefix scale nor `logical_units * stride` is accepted as a
fallback.

## First correctness scope

The v1 factory emits only `group_full_page` jobs and rejects plans containing
more than four group-qualified destination IDs. This is a bounded correctness
scope, not a workload-capacity or performance conclusion. Lifting it requires
an explicitly evidence-backed successor contract rather than bypassing the
gate.
