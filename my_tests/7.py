# 功能验证说明：
# - 验证 TieringOffloadingManager 分层 manager 状态机。
# - 覆盖 CPU primary HIT 直接 load、complete_store() 触发 primary -> secondary cascade。
# - 覆盖 primary eviction 后 secondary HIT 触发 promotion，lookup 返回 RETRY/HIT_PENDING。
# - 覆盖 promotion 完成后 primary lookup 恢复 HIT，并可 prepare_load/complete_load。
# - 追加 FileSystemTierManager 验证，覆盖 CPU primary -> SSD(fs) -> CPU primary 的真实字节 I/O。
#
PYTHONPATH="$PWD" python3 - <<'PY'
import mmap
import os
import tempfile
import time
from unittest.mock import MagicMock

import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

LINE = "=" * 70


def banner(step: str) -> None:
    print(LINE)
    print(step)
    print(LINE)


def pass_msg(msg: str) -> None:
    print(f"PASS {msg}")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def require_is(got, expected, msg: str) -> None:
    if got is not expected:
        raise AssertionError(f"{msg}: expected {expected}, got {got}")


def key(i: int):
    return make_offload_key(str(i).encode(), 0)


def make_mock_mmap_region(num_blocks: int, row_bytes: int = 4096):
    region = MagicMock()
    backing = torch.zeros((num_blocks, row_bytes), dtype=torch.int8)
    region.create_kv_memoryview.return_value = memoryview(backing.numpy())
    region.cleanup.return_value = None
    region._test_backing = backing
    return region


def make_page_aligned_mock_mmap_region(num_blocks: int, row_bytes: int = 4096):
    assert row_bytes % mmap.PAGESIZE == 0
    region = MagicMock()
    total_bytes = num_blocks * row_bytes
    raw = torch.zeros(total_bytes + mmap.PAGESIZE, dtype=torch.uint8)
    shift = (mmap.PAGESIZE - (raw.data_ptr() % mmap.PAGESIZE)) % mmap.PAGESIZE
    aligned = raw[shift : shift + total_bytes].view(torch.int8)
    backing = aligned.view(num_blocks, row_bytes)
    region.create_kv_memoryview.return_value = memoryview(backing.numpy())
    region.cleanup.return_value = None
    region._test_raw = raw
    region._test_backing = backing
    return region


def make_fs_offloading_spec(block_size_factor: int = 1):
    vllm_config = MagicMock()
    vllm_config.model_config.model = "tiering-validation-model"
    vllm_config.cache_config.block_size = 16
    vllm_config.cache_config.cache_dtype = "torch.int8"
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.parallel_config.rank = 0
    vllm_config.use_v2_model_runner = False

    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = []

    offloading_spec = MagicMock()
    offloading_spec.vllm_config = vllm_config
    offloading_spec.kv_cache_config = kv_cache_config
    offloading_spec.block_size_factor = block_size_factor
    return offloading_spec


def schedule_end(manager: TieringOffloadingManager, rounds: int = 1) -> None:
    ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
    for _ in range(rounds):
        manager.on_schedule_end(ctx)
        list(manager.take_events())


def primary_ref_cnt(primary: CPUPrimaryTierOffloadingManager, k) -> int:
    block = primary._policy.get(k)
    require(block is not None, f"primary block missing for key={k!r}")
    return block.ref_cnt


def drive_until(description: str, predicate, manager: TieringOffloadingManager):
    for _ in range(200):
        if predicate():
            return
        schedule_end(manager, rounds=1)
        time.sleep(0.01)
    raise AssertionError(f"timeout while waiting for {description}")


banner("Step1")
req_ctx = ReqContext(req_id="tiering_step7_validation")
mock_region = make_mock_mmap_region(num_blocks=2)

primary = CPUPrimaryTierOffloadingManager(
    num_blocks=2,
    mmap_region=mock_region,
    cache_policy="lru",
)

secondary = ExampleSecondaryTierManager(
    offloading_spec=MagicMock(),
    primary_kv_view=primary.get_kv_memoryview(),
    tier_type="example",
)

manager = TieringOffloadingManager(
    primary_tier=primary,
    secondary_tiers=[secondary],
    enable_events=True,
)

schedule_ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
manager.on_new_request(req_ctx)

require(req_ctx.req_id in manager._req_state, "request state was not initialized")
require(schedule_ctx.new_req_ids == [], "ScheduleEndContext initialization failed")

pass_msg(
    "created TieringOffloadingManager, CPU primary, Example secondary, "
    "ReqContext, ScheduleEndContext"
)

k0, k1, k2 = key(0), key(1), key(2)

banner("Step2")
secondary.submit_store = MagicMock(wraps=secondary.submit_store)

store_out = manager.prepare_store([k0, k1], req_ctx)
require(store_out is not None, "prepare_store returned None")
require(store_out.keys_to_store == [k0, k1], "prepare_store keys_to_store mismatch")

manager.complete_store([k0, k1], req_ctx, success=True)

secondary.submit_store.assert_called_once()
require(secondary.get_num_blocks() == 2, "secondary did not receive cascaded blocks")

require_is(
    secondary.lookup(k0, req_ctx),
    LookupResult.HIT,
    "secondary lookup k0 after cascade",
)
require_is(
    secondary.lookup(k1, req_ctx),
    LookupResult.HIT,
    "secondary lookup k1 after cascade",
)

require_is(manager.lookup(k0, req_ctx), LookupResult.HIT, "CPU primary lookup k0")
require_is(manager.lookup(k1, req_ctx), LookupResult.HIT, "CPU primary lookup k1")

load_spec = manager.prepare_load([k0, k1], req_ctx)
require(
    list(load_spec.block_ids) == [0, 1],
    f"primary load block ids mismatch: {load_spec.block_ids}",
)
manager.complete_load([k0, k1], req_ctx)

pass_msg(
    "prepare_store -> complete_store triggered primary -> secondary cascade "
    "and primary direct load works"
)

banner("Step3")
schedule_end(manager, rounds=2)

require(
    primary_ref_cnt(primary, k0) == 0,
    "k0 ref_cnt was not released after cascade completion",
)
require(
    primary_ref_cnt(primary, k1) == 0,
    "k1 ref_cnt was not released after cascade completion",
)

evict_store = manager.prepare_store([k2], req_ctx)
require(evict_store is not None, "prepare_store for eviction returned None")
require(evict_store.keys_to_store == [k2], "eviction store keys_to_store mismatch")
require(
    len(evict_store.evicted_keys) == 1,
    f"expected one evicted key, got {evict_store.evicted_keys}",
)

evicted_key = evict_store.evicted_keys[0]
manager.complete_store([k2], req_ctx, success=True)

require_is(
    primary.lookup(evicted_key, req_ctx),
    LookupResult.MISS,
    "evicted key primary lookup",
)
require_is(
    secondary.lookup(evicted_key, req_ctx),
    LookupResult.HIT,
    "evicted key secondary lookup",
)

pass_msg("primary eviction produced primary MISS while secondary still has HIT")

banner("Step4")
secondary.submit_load = MagicMock(wraps=secondary.submit_load)

lookup_result = manager.lookup(evicted_key, req_ctx)
require_is(
    lookup_result,
    LookupResult.RETRY,
    "secondary-hit lookup should start promotion",
)
require(
    len(manager._pending_load_submissions) == 1,
    "promotion was not queued in pending submissions",
)

repeat_lookup = manager.lookup(evicted_key, req_ctx)
require_is(
    repeat_lookup,
    LookupResult.HIT_PENDING,
    "duplicate lookup during promotion should be HIT_PENDING",
)

secondary.submit_load.assert_not_called()

schedule_end(manager, rounds=1)

secondary.submit_load.assert_called_once()
require(
    any(j.is_promotion for j in manager._transfer_jobs.values()),
    "promotion transfer job was not tracked",
)

pass_msg(
    "lookup returned RETRY, duplicate lookup returned HIT_PENDING, "
    "promotion was submitted on schedule end"
)

banner("Step5")
schedule_end(manager, rounds=1)

require(
    not any(j.is_promotion for j in manager._transfer_jobs.values()),
    "promotion job still tracked after completion",
)
require_is(
    manager.lookup(evicted_key, req_ctx),
    LookupResult.HIT,
    "promoted key lookup after completion",
)

pass_msg("promotion completion processed and lookup returns HIT")

banner("Step6")
final_load_spec = manager.prepare_load([evicted_key], req_ctx)
require(
    len(final_load_spec.block_ids) == 1,
    "prepare_load for promoted key did not return one block id",
)

manager.complete_load([evicted_key], req_ctx)

require_is(
    primary.lookup(evicted_key, req_ctx),
    LookupResult.HIT,
    "primary lookup after final complete_load",
)

pass_msg("prepare_load -> complete_load completed and primary cache is usable after promotion")

banner("Step7")
manager.on_request_finished(req_ctx)
manager.shutdown()

pass_msg("tiering manager validation completed")

banner("Step8")
fs_tmpdir = tempfile.TemporaryDirectory(prefix="vllm-tiering-fs-")
fs_req_ctx = ReqContext(req_id="tiering_fs_validation")
fs_region = make_page_aligned_mock_mmap_region(num_blocks=1, row_bytes=mmap.PAGESIZE)
fs_primary = CPUPrimaryTierOffloadingManager(
    num_blocks=1,
    mmap_region=fs_region,
    cache_policy="lru",
)
fs_secondary = FileSystemTierManager(
    offloading_spec=make_fs_offloading_spec(block_size_factor=1),
    primary_kv_view=fs_primary.get_kv_memoryview(),
    tier_type="fs",
    root_dir=fs_tmpdir.name,
    n_read_threads=1,
    n_write_threads=1,
)
fs_manager = TieringOffloadingManager(
    primary_tier=fs_primary,
    secondary_tiers=[fs_secondary],
    enable_events=True,
)
fs_manager.on_new_request(fs_req_ctx)
fs_key = key(100)
fs_backing = fs_region._test_backing
fs_pattern = ((torch.arange(mmap.PAGESIZE, dtype=torch.int16) % 127) - 63).to(
    torch.int8
)
fs_backing[0].copy_(fs_pattern)
pass_msg("created TieringOffloadingManager with FileSystemTierManager and page-aligned primary memory")

banner("Step9")
fs_store_out = fs_manager.prepare_store([fs_key], fs_req_ctx)
require(fs_store_out is not None, "fs prepare_store returned None")
require(fs_store_out.keys_to_store == [fs_key], "fs prepare_store keys_to_store mismatch")
require(list(fs_store_out.store_spec.block_ids) == [0], "fs primary block id mismatch")
fs_manager.complete_store([fs_key], fs_req_ctx, success=True)
drive_until(
    "fs cascade completion",
    lambda: not fs_manager._transfer_jobs,
    fs_manager,
)
fs_path = fs_secondary.file_mapper.get_file_name(fs_key)
require(os.path.exists(fs_path), f"fs secondary file was not created: {fs_path}")
require_is(fs_manager.lookup(fs_key, fs_req_ctx), LookupResult.HIT, "fs primary lookup after cascade")
pass_msg("complete_store cascaded real primary bytes to fs secondary and created SSD file")

banner("Step10")
fs_manager.reset_cache()
fs_backing[0].fill_(-9)
require_is(
    fs_primary.lookup(fs_key, fs_req_ctx),
    LookupResult.MISS,
    "fs primary lookup after reset_cache",
)

promotion_started = False
for _ in range(200):
    result = fs_manager.lookup(fs_key, fs_req_ctx)
    if result is LookupResult.RETRY and fs_manager._pending_load_submissions:
        promotion_started = True
        break
    if result is LookupResult.HIT_PENDING:
        promotion_started = True
        break
    schedule_end(fs_manager, rounds=1)
    time.sleep(0.01)

require(promotion_started, "fs secondary hit did not start promotion")
schedule_end(fs_manager, rounds=1)
require(
    any(job.is_promotion for job in fs_manager._transfer_jobs.values()),
    "fs promotion transfer job was not submitted",
)
pass_msg("primary MISS plus fs secondary HIT started promotion through real fs tier")

banner("Step11")
drive_until(
    "fs promotion completion",
    lambda: not any(job.is_promotion for job in fs_manager._transfer_jobs.values()),
    fs_manager,
)
require_is(
    fs_manager.lookup(fs_key, fs_req_ctx),
    LookupResult.HIT,
    "fs promoted key lookup after completion",
)
fs_load_spec = fs_manager.prepare_load([fs_key], fs_req_ctx)
require(list(fs_load_spec.block_ids) == [0], "fs promoted block id mismatch")
torch.testing.assert_close(fs_backing[0], fs_pattern)
fs_manager.complete_load([fs_key], fs_req_ctx)
pass_msg("fs promotion restored original primary bytes exactly")

banner("Step12")
fs_manager.on_request_finished(fs_req_ctx)
fs_manager.shutdown()
fs_tmpdir.cleanup()
pass_msg("filesystem secondary byte-level validation completed")

print(LINE)
print("FINAL RESULT : PASS")
print(LINE)
PY
