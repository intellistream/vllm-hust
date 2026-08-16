#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a bounded two-rank Ascend request-owned KV lifecycle probe.

This is a correctness probe, not a throughput benchmark.  It does not load
model weights.  Each torchrun rank owns one NPU and independently exercises
the production packed-page zeroer, request-owned bulk ledger/adapter, native
``swap_blocks_batch`` D2H/H2D worker, and exact host-key cleanup.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import gc
import hashlib
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch_npu  # noqa: F401
import vllm_ascend

import vllm
from vllm.v1.core.sched.ownership import OwnerLeaseKey
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.npu_worker import AscendCPUOffloadingWorker
from vllm.v1.kv_offload.cpu.worker_factory import create_cpu_offloading_worker
from vllm.v1.worker.request_owned_kv import RequestOwnedKVSnapshot
from vllm.v1.worker.request_owned_offload import (
    OwnerOffloadPlan,
    RequestOwnedBulkOffloadAdapter,
    RequestOwnedBulkRestoreWork,
    RequestOwnedOffloadError,
    make_request_owned_offload_keys,
)

NUM_DEVICE_BLOCKS = 32
HOST_BLOCKS = 16
PAGE_BYTES = 4096
PACKED_PREFIX_BYTES = 64
PACKED_SUFFIX_BYTES = 64
GROUP_BLOCK_SIZES = (4, 8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_error(
    fn: Callable[[], Any],
    needle: str,
    *,
    error_type: type[BaseException] = RequestOwnedOffloadError,
) -> str:
    try:
        fn()
    except error_type as exc:
        text = str(exc)
        _require(needle in text, f"expected {needle!r} in {text!r}")
        return text
    raise AssertionError(f"expected {error_type.__name__} containing {needle!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_import_root(module_file: str, expected_root: Path) -> Path:
    path = Path(module_file).resolve()
    root = expected_root.resolve()
    _require(root == path or root in path.parents, f"{path} is not under {root}")
    return path


def _native_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], type, type]:
    core = _assert_import_root(vllm.__file__, args.expected_core_root)
    ascend = _assert_import_root(vllm_ascend.__file__, args.expected_ascend_root)
    package_dir = ascend.parent
    kernels = (package_dir / "libvllm_ascend_kernels.so").resolve()
    _require(kernels.exists(), f"missing kernels library: {kernels}")
    ctypes.CDLL(str(kernels), mode=os.RTLD_GLOBAL)
    extension_module = importlib.import_module("vllm_ascend.vllm_ascend_C")
    extension = Path(extension_module.__file__).resolve()
    _require(extension.exists(), f"missing extension: {extension}")

    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.utils import AscendPackedKVBlockZeroer

    _require(enable_custom_op(), "vllm-ascend custom-op registration failed")
    _require(
        hasattr(torch.ops._C_ascend, "swap_blocks_batch"),
        "torch.ops._C_ascend.swap_blocks_batch is unavailable",
    )
    receipt = {
        "core": str(core),
        "ascend": str(ascend),
        "extension": str(extension),
        "extension_sha256": _sha256(extension),
        "kernels": str(kernels),
        "kernels_sha256": _sha256(kernels),
        "custom_opp": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "core_head": os.environ.get("PROBE_CORE_HEAD"),
        "ascend_head": os.environ.get("PROBE_ASCEND_HEAD"),
        "required_ops": ["swap_blocks_batch"],
    }
    return receipt, NPUModelRunner, AscendPackedKVBlockZeroer


def _make_probe_runner(*, rank: int, zeroer: Any, model_runner_cls: type) -> Any:
    """Build a minimal receiver for the exact production O1 restore seam."""

    probe_runner_cls = type(
        "_ProbeRunner",
        (),
        {
            "execute_request_owned_bulk_restore": (
                model_runner_cls.execute_request_owned_bulk_restore
            ),
            "_synchronize_request_owned_restore_zero": staticmethod(
                model_runner_cls._synchronize_request_owned_restore_zero
            ),
        },
    )
    runner = probe_runner_cls()
    runner.scheduler_config = SimpleNamespace(enable_request_owned_kv_offload=True)
    runner.parallel_config = SimpleNamespace(rank=rank)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=(object(), object()))
    runner._request_owned_packed_zeroer = zeroer
    return runner


def _snapshot(
    *,
    key: OwnerLeaseKey,
    rank: int,
    generation: int,
    computed: int,
    tables: tuple[tuple[int, ...], ...],
) -> RequestOwnedKVSnapshot:
    return RequestOwnedKVSnapshot(
        key=key,
        owner_rank=rank,
        allocation_generation=generation,
        num_computed_tokens=computed,
        reserved_num_tokens=12,
        pending_free=False,
        tables=tables,
    )


def _pattern(rank: int, block_id: int, *, revision: int) -> torch.Tensor:
    values = torch.arange(PAGE_BYTES, dtype=torch.int32)
    values = (values * 17 + rank * 29 + block_id * 13 + revision * 37) % 251
    return (values - 125).to(torch.int8)


def _put_pattern(
    backing: torch.Tensor,
    rank: int,
    block_id: int,
    *,
    revision: int,
) -> torch.Tensor:
    expected = _pattern(rank, block_id, revision=revision)
    backing[block_id].copy_(expected.to(backing.device))
    return expected


def _assert_block(backing: torch.Tensor, block_id: int, expected: torch.Tensor) -> None:
    torch.testing.assert_close(backing[block_id].cpu(), expected)


def _assert_receipt(receipt: Any, plan: OwnerOffloadPlan) -> None:
    _require(receipt.success, receipt.error or "transfer receipt failed")
    _require(receipt.identity == plan.identity, "receipt identity mismatch")
    _require(
        receipt.device_block_ids == plan.device_block_ids,
        "receipt device block IDs mismatch",
    )
    _require(receipt.offload_keys == plan.offload_keys, "receipt host keys mismatch")


def _store(
    adapter: RequestOwnedBulkOffloadAdapter,
    snapshot: RequestOwnedKVSnapshot,
) -> tuple[OwnerOffloadPlan, Any]:
    keys = make_request_owned_offload_keys(snapshot, GROUP_BLOCK_SIZES)
    plan = OwnerOffloadPlan.from_snapshot(snapshot, keys)
    identity = adapter.bind(snapshot, active=True)
    _expect_error(lambda: adapter.submit_store(plan), "ACTIVE")
    adapter.retire(identity)
    job = adapter.submit_store(plan)
    _require(not adapter.ledger.is_host_durable(plan), "store became durable early")
    _expect_error(lambda: adapter.take_reclaimable(identity), "durable store")
    adapter.wait((job,))
    receipts = adapter.poll()
    _require(len(receipts) == 1, f"store produced {len(receipts)} receipts")
    receipt = receipts[0]
    _assert_receipt(receipt, plan)
    _require(adapter.ledger.is_host_durable(plan), "store receipt is not durable")
    _expect_error(lambda: adapter.ledger.complete(receipt), "duplicate")
    _require(
        adapter.take_reclaimable(identity) == plan.device_block_ids,
        "reclaim receipt does not name the exact source blocks",
    )
    return plan, receipt


def _restore(
    *,
    adapter: RequestOwnedBulkOffloadAdapter,
    runner: Any,
    backing: torch.Tensor,
    snapshot: RequestOwnedKVSnapshot,
    keys: tuple[tuple[bytes, ...], ...],
    expected_pages: tuple[tuple[torch.Tensor, ...], ...],
    step_seq: int,
) -> tuple[OwnerOffloadPlan, Any]:
    plan = OwnerOffloadPlan.from_snapshot(snapshot, keys)
    identity = adapter.bind(snapshot, active=False)
    _expect_error(lambda: adapter.activate(identity), "restore completion")

    zero_ids = snapshot.tables
    for group in zero_ids:
        for block_id in group:
            backing[block_id].fill_(77)
    torch.npu.synchronize()

    bad_zero_ids = (
        zero_ids[0],
        (zero_ids[0][0], *zero_ids[1]),
    )
    bad_work = RequestOwnedBulkRestoreWork(
        step_seq=step_seq,
        adapter=adapter,
        plan=plan,
        zero_block_ids=bad_zero_ids,
    )
    before_bad_zero = backing.cpu().clone()
    _expect_error(
        lambda: runner.execute_request_owned_bulk_restore((bad_work,)),
        "appears in kv cache groups",
        error_type=ValueError,
    )
    _require(not bad_work.executed, "H2D began after a rejected zero plan")
    torch.testing.assert_close(backing.cpu(), before_bad_zero)
    _require(adapter.ledger.pending_jobs == (), "rejected zero left pending DMA")

    work = RequestOwnedBulkRestoreWork(
        step_seq=step_seq,
        adapter=adapter,
        plan=plan,
        zero_block_ids=zero_ids,
    )
    runner.execute_request_owned_bulk_restore((work,))
    _require(work.executed, "restore work was not executed")
    _expect_error(work.execute_after_zero, "already executed")
    _require(adapter.ledger.is_hot(identity), "restore receipt did not mark HOT")

    for group_index, block_ids in enumerate(plan.device_block_ids):
        for position, block_id in enumerate(block_ids):
            _assert_block(backing, block_id, expected_pages[group_index][position])
    for group_index, group in enumerate(zero_ids):
        restored = set(plan.device_block_ids[group_index])
        for block_id in group:
            if block_id not in restored:
                _assert_block(
                    backing,
                    block_id,
                    torch.zeros(PAGE_BYTES, dtype=torch.int8),
                )

    adapter.activate(identity)
    _require(adapter.ledger.pending_jobs == (), "activation left pending DMA")
    return plan, identity


def _run_rank(rank: int, args: argparse.Namespace) -> dict[str, Any]:
    _require(
        args.world_size == 2,
        f"probe requires exactly 2 ranks, got {args.world_size}",
    )
    native, model_runner_cls, packed_zeroer_cls = _native_preflight(args)
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    free_before, total_memory = torch.npu.mem_get_info(rank)
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton,
    )

    init_device_properties_triton()

    raw = torch.full(
        (PACKED_PREFIX_BYTES + NUM_DEVICE_BLOCKS * PAGE_BYTES + PACKED_SUFFIX_BYTES,),
        -101,
        dtype=torch.int8,
        device=device,
    )
    backing = raw[
        PACKED_PREFIX_BYTES : PACKED_PREFIX_BYTES + NUM_DEVICE_BLOCKS * PAGE_BYTES
    ].view(NUM_DEVICE_BLOCKS, PAGE_BYTES)
    _require(backing.is_contiguous(), "offset packed backing is not contiguous")
    _require(
        backing.data_ptr() - raw.data_ptr() == PACKED_PREFIX_BYTES,
        "packed backing base offset was lost",
    )

    kv_caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=backing, page_size_bytes=PAGE_BYTES)],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE_BYTES)],
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE_BYTES)],
        ],
    )
    worker = create_cpu_offloading_worker(
        kv_caches=kv_caches,
        block_size_factor=1,
        num_cpu_blocks=HOST_BLOCKS,
    )
    _require(
        isinstance(worker, AscendCPUOffloadingWorker),
        f"unexpected worker type: {type(worker).__name__}",
    )
    manager = CPUOffloadingManager(num_blocks=HOST_BLOCKS)
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=rank,
        manager=manager,
        worker=worker,
    )
    atexit.register(adapter.shutdown)
    zeroer = packed_zeroer_cls(device, pin_memory=True)
    zeroer.init_meta(
        backing,
        (
            (0, PAGE_BYTES, PAGE_BYTES),
            (0, PAGE_BYTES, PAGE_BYTES),
        ),
        num_blocks=NUM_DEVICE_BLOCKS,
    )
    runner = _make_probe_runner(
        rank=rank,
        zeroer=zeroer,
        model_runner_cls=model_runner_cls,
    )
    key = OwnerLeaseKey(f"o2-rank-{rank}", 0)

    source1_tables = ((1, 3), (5,))
    source1 = _snapshot(
        key=key,
        rank=rank,
        generation=1,
        computed=6,
        tables=source1_tables,
    )
    expected1 = (
        (
            _put_pattern(backing, rank, 1, revision=1),
            _put_pattern(backing, rank, 3, revision=1),
        ),
        (_put_pattern(backing, rank, 5, revision=1),),
    )
    plan1, store_receipt1 = _store(adapter, source1)
    keys1 = plan1.offload_keys
    for group in plan1.device_block_ids:
        for block_id in group:
            backing[block_id].fill_(91)

    destination1 = _snapshot(
        key=key,
        rank=rank,
        generation=2,
        computed=6,
        tables=((8, 10, 11), (12, 14)),
    )
    _, destination1_id = _restore(
        adapter=adapter,
        runner=runner,
        backing=backing,
        snapshot=destination1,
        keys=keys1,
        expected_pages=expected1,
        step_seq=1,
    )
    try:
        adapter.retire(plan1.identity)
    except RequestOwnedOffloadError as exc:
        _require("stale" in str(exc), f"unexpected stale-fence error: {exc}")
    else:
        raise AssertionError("stale generation was accepted")

    # Close the first restored generation but deliberately keep its host image.
    # Extending the computed prefix changes only each partial-tail key; full
    # logical blocks remain reusable while the changed tails must be recopied.
    adapter.release(destination1)
    source2 = _snapshot(
        key=key,
        rank=rank,
        generation=3,
        computed=7,
        tables=((8, 10), (12,)),
    )
    expected2 = (
        (
            expected1[0][0],
            _put_pattern(backing, rank, 10, revision=2),
        ),
        (_put_pattern(backing, rank, 12, revision=2),),
    )
    _assert_block(backing, 8, expected1[0][0])
    plan2, store_receipt2 = _store(adapter, source2)
    keys2 = plan2.offload_keys
    _require(keys2[0][0] == keys1[0][0], "full-block host key was not reused")
    _require(keys2[0][1] != keys1[0][1], "group-0 partial-tail key aliased")
    _require(keys2[1][0] != keys1[1][0], "group-1 partial-tail key aliased")
    for group in plan2.device_block_ids:
        for block_id in group:
            backing[block_id].fill_(92)

    destination2 = _snapshot(
        key=key,
        rank=rank,
        generation=4,
        computed=7,
        tables=((16, 18, 19), (20, 22)),
    )
    plan4, destination2_id = _restore(
        adapter=adapter,
        runner=runner,
        backing=backing,
        snapshot=destination2,
        keys=keys2,
        expected_pages=expected2,
        step_seq=2,
    )
    _require(destination1_id != destination2_id, "generation fence did not advance")
    _require(plan4.identity == destination2_id, "final destination identity mismatch")

    adapter.release(source1)  # stale release is an inert no-op.
    adapter.release(destination2)
    adapter.evict_owned_host_keys(key)
    _require(
        not adapter.ledger.is_host_durable(plan4),
        "RELEASE left the final host image durable",
    )
    _require(manager.resident_blocks == 0, "RELEASE leaked durable host blocks")
    _require(adapter.ledger.pending_jobs == (), "RELEASE leaked pending jobs")
    _require(adapter.poll() == (), "RELEASE left an unconsumed receipt")

    torch.npu.synchronize()
    prefix = raw[:PACKED_PREFIX_BYTES].cpu()
    suffix = raw[-PACKED_SUFFIX_BYTES:].cpu()
    _require(bool(torch.all(prefix == -101)), "packed prefix sentinel was corrupted")
    _require(bool(torch.all(suffix == -101)), "packed suffix sentinel was corrupted")

    result = {
        "status": "pass",
        "rank": rank,
        "world_size": args.world_size,
        "device": str(device),
        "free_before_bytes": free_before,
        "total_memory_bytes": total_memory,
        "packed_backing_offset_bytes": PACKED_PREFIX_BYTES,
        "packed_page_bytes": PAGE_BYTES,
        "device_blocks": NUM_DEVICE_BLOCKS,
        "host_blocks": HOST_BLOCKS,
        "cycles": [
            {
                "computed_tokens": 6,
                "store_job": store_receipt1.job_id,
                "stored_blocks": plan1.device_block_ids,
                "partial_tail": True,
            },
            {
                "computed_tokens": 7,
                "store_job": store_receipt2.job_id,
                "stored_blocks": plan2.device_block_ids,
                "partial_tail_keys_changed": True,
            },
        ],
        "oracles": {
            "active_never_stored": True,
            "durable_before_reclaim": True,
            "receipt_replay_fenced": True,
            "stale_generation_fenced": True,
            "zero_failure_issued_no_h2d": True,
            "zero_before_h2d": True,
            "exact_destination_bytes": True,
            "extra_destination_pages_zero": True,
            "partial_tail_keys_extent_qualified": True,
            "release_evicts_exact_host_image": True,
            "clean_terminal_state": True,
        },
        "native": native,
    }

    adapter.shutdown()
    atexit.unregister(adapter.shutdown)
    del runner, zeroer, adapter, manager, worker, kv_caches, backing, raw
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    free_after, _ = torch.npu.mem_get_info(rank)
    result["free_after_bytes"] = free_after
    result["allocator_delta_bytes"] = free_before - free_after
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-core-root", type=Path, required=True)
    parser.add_argument("--expected-ascend-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.rank = int(os.environ.get("RANK", "-1"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    _require(args.rank >= 0, "RANK is required; launch this probe with torchrun")
    _require(args.local_rank == args.rank, "single-node rank/local-rank mismatch")
    return args


def main() -> None:
    args = _parse_args()
    result = _run_rank(args.rank, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rank-{args.rank}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
