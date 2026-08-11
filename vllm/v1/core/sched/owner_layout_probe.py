# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded scheduler-side observation of request-owner layouts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from vllm.v1.core.sched.ownership import OwnerLeaseToken

PROBE_DIR_ENV = "VLLM_REQUEST_OWNER_LAYOUT_PROBE_DIR"
RUN_ID_ENV = "VLLM_TELEMETRY_RUN_ID"
MAX_RECORDS_ENV = "VLLM_REQUEST_OWNER_LAYOUT_PROBE_MAX_RECORDS"
MAX_BYTES_ENV = "VLLM_REQUEST_OWNER_LAYOUT_PROBE_MAX_BYTES"
SCHEMA = "g4-request-owner-layout-observation/v1"
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _positive_env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


class OwnerLayoutProbe:
    """Write bounded authoritative scheduler layouts as strict JSONL."""

    def __init__(
        self,
        *,
        path: Path,
        run_id: str,
        world_size: int,
        max_records: int,
        max_bytes: int,
    ) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("world_size must be a non-bool int")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.path = path
        self.run_id = run_id
        self.world_size = world_size
        self.max_records = max_records
        self.max_bytes = max_bytes
        self._records = 0
        self._bytes = 0
        self._stream = path.open("x", encoding="utf-8")
        self._write({
            "schema": SCHEMA,
            "kind": "header",
            "run_id": run_id,
            "world_size": world_size,
        })

    @classmethod
    def requested(cls) -> bool:
        return bool(os.getenv(PROBE_DIR_ENV, "").strip())

    @classmethod
    def from_env(cls, *, world_size: int) -> OwnerLayoutProbe | None:
        directory_value = os.getenv(PROBE_DIR_ENV, "").strip()
        if not directory_value:
            return None
        directory = Path(directory_value)
        if not directory.is_absolute() or not directory.is_dir():
            raise ValueError(
                f"{PROBE_DIR_ENV} must name an existing absolute directory, "
                f"got {directory_value!r}"
            )
        run_id = os.getenv(RUN_ID_ENV, "").strip()
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(
                f"{RUN_ID_ENV} must match {_RUN_ID_RE.pattern!r}, got {run_id!r}"
            )
        return cls(
            path=directory / f"request-owner-layout-{run_id}.jsonl",
            run_id=run_id,
            world_size=world_size,
            max_records=_positive_env_int(MAX_RECORDS_ENV, 512),
            max_bytes=_positive_env_int(MAX_BYTES_ENV, 1 << 20),
        )

    def _write(self, record: dict) -> None:
        if self._records >= self.max_records:
            raise RuntimeError(
                f"request-owner layout probe exceeded {self.max_records} records"
            )
        payload = json.dumps(
            record,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
        encoded = payload.encode("utf-8")
        if self._bytes + len(encoded) > self.max_bytes:
            raise RuntimeError(
                f"request-owner layout probe exceeded {self.max_bytes} bytes"
            )
        self._stream.write(payload)
        self._stream.flush()
        self._records += 1
        self._bytes += len(encoded)

    def record_step(
        self,
        *,
        step_seq: int,
        leases: Sequence[OwnerLeaseToken],
        num_scheduled_tokens: Mapping[str, int],
    ) -> None:
        if isinstance(step_seq, bool) or not isinstance(step_seq, int) or step_seq <= 0:
            raise ValueError(
                f"step_seq must be a positive non-bool int, got {step_seq!r}"
            )
        scheduled: dict[str, int] = {}
        for request_id, count in num_scheduled_tokens.items():
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(
                    f"scheduled request id must be nonempty, got {request_id!r}"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"scheduled token count for {request_id!r} must be positive, "
                    f"got {count!r}"
                )
            scheduled[request_id] = count

        by_request: dict[str, OwnerLeaseToken] = {}
        for lease in leases:
            if not isinstance(lease, OwnerLeaseToken):
                raise TypeError(f"leases must contain OwnerLeaseToken, got {lease!r}")
            request_id = lease.key.request_id
            if request_id in by_request:
                raise ValueError(
                    f"duplicate published lease for request {request_id!r}"
                )
            if lease.step_seq != step_seq:
                raise ValueError(
                    f"lease step {lease.step_seq} for {request_id!r} != {step_seq}"
                )
            if (
                isinstance(lease.owner_id, bool)
                or not isinstance(lease.owner_id, int)
                or not 0 <= lease.owner_id < self.world_size
            ):
                raise ValueError(
                    f"lease owner for {request_id!r} is outside world size: "
                    f"{lease.owner_id!r}"
                )
            by_request[request_id] = lease
        if by_request.keys() != scheduled.keys():
            raise ValueError(
                "published lease requests must exactly match positive "
                "scheduled requests"
            )

        owner_row_counts = [0] * self.world_size
        owner_request_counts = [0] * self.world_size
        assignments = []
        for request_id in sorted(scheduled):
            lease = by_request[request_id]
            count = scheduled[request_id]
            owner_row_counts[lease.owner_id] += count
            owner_request_counts[lease.owner_id] += 1
            assignments.append({
                "request_id": request_id,
                "owner_epoch": lease.key.owner_epoch,
                "owner_rank": lease.owner_id,
                "num_scheduled_tokens": count,
                "runnable_num_tokens": lease.runnable_num_tokens,
            })
        self._write({
            "schema": SCHEMA,
            "kind": "step",
            "run_id": self.run_id,
            "step_seq": step_seq,
            "assignments": assignments,
            "scheduled_request_count": len(assignments),
            "total_scheduled_tokens": sum(owner_row_counts),
            "owner_request_counts": owner_request_counts,
            "owner_row_counts": owner_row_counts,
            "zero_row_owner_ranks": [
                rank for rank, count in enumerate(owner_row_counts) if count == 0
            ],
        })
