# 功能验证说明：
# - 验证 CPU offload worker 的异步完成语义和多 job load/store 数据闭环。
# - 对多个 STORE/LOAD job 检查 job_id、success、transfer_size、transfer_time。
# - 验证 get_finished() drain 行为，并确认 LOAD 后 NPU KV tensor 完整恢复。
#
python - <<'PY'
import torch

from vllm.platforms import current_platform
from vllm_ascend.utils import enable_custom_op

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.worker_factory import (
    create_cpu_offloading_worker,
    is_ascend_platform,
)

print("=" * 70)
print("Test7: Async Completion Semantics")
print("=" * 70)

assert current_platform.device_type == "npu"
assert is_ascend_platform()
assert enable_custom_op()
assert hasattr(torch.ops._C_ascend, "swap_blocks_batch")

page_size = 512
block_size_factor = 1
num_gpu_blocks = 16
num_cpu_blocks = 16
device = "npu:0"

########################################################################
# Build KV Cache
########################################################################

kv_tensor = torch.zeros(
    (num_gpu_blocks, page_size),
    dtype=torch.int8,
    device=device,
)

for block in range(num_gpu_blocks):
    kv_tensor[block].fill_(block + 1)

expected = kv_tensor.detach().cpu().clone()

kv_caches = CanonicalKVCaches(
    tensors=[
        CanonicalKVCacheTensor(
            tensor=kv_tensor,
            page_size_bytes=page_size,
        )
    ],
    group_data_refs=[
        [
            CanonicalKVCacheRef(
                tensor_idx=0,
                page_size_bytes=page_size,
            )
        ]
    ],
)

worker = create_cpu_offloading_worker(
    kv_caches=kv_caches,
    block_size_factor=block_size_factor,
    num_cpu_blocks=num_cpu_blocks,
)

########################################################################
# STORE JOBS
########################################################################

print("\n" + "=" * 70)
print("Step1. Submit STORE jobs")
print("=" * 70)

store_jobs = [
    (701, [0, 1], [0, 1]),
    (702, [4, 5], [4, 5]),
    (703, [8, 9], [8, 9]),
]

for job_id, gpu_blocks, cpu_blocks in store_jobs:

    ok = worker.submit_store(
        job_id,
        GPULoadStoreSpec(
            block_ids=gpu_blocks,
            group_sizes=[len(gpu_blocks)],
            block_indices=[0],
        ),
        CPULoadStoreSpec(cpu_blocks),
    )

    print(
        f"submit_store job={job_id} "
        f"gpu={gpu_blocks} cpu={cpu_blocks} ok={ok}"
    )

    assert ok

print("\nWaiting STORE jobs ...")

worker.wait({701, 702, 703})

finished = worker.get_finished()

print("\nSTORE get_finished():")
for r in finished:
    print(r)

assert len(finished) == 3

finished_by_id = {r.job_id: r for r in finished}

assert set(finished_by_id.keys()) == {701, 702, 703}

for job_id in [701, 702, 703]:
    r = finished_by_id[job_id]

    assert r.success is True
    assert r.transfer_size == 2 * page_size
    assert r.transfer_time >= 0

print("\nPASS: STORE completion semantics correct.")

########################################################################
# get_finished drain
########################################################################

print("\nChecking drain semantics...")

drained = worker.get_finished()

print("second get_finished():", drained)

assert drained == []

print("PASS: get_finished() drains completed STORE jobs.")

########################################################################
# Clear GPU
########################################################################

print("\n" + "=" * 70)
print("Step2. Clear NPU")
print("=" * 70)

for block in [0, 1, 4, 5, 8, 9]:
    kv_tensor[block].zero_()

torch.npu.synchronize()

cleared = kv_tensor[[0,1,4,5,8,9]].detach().cpu()

torch.testing.assert_close(
    cleared,
    torch.zeros_like(cleared),
)

print("PASS: NPU cleared.")

########################################################################
# LOAD JOBS
########################################################################

print("\n" + "=" * 70)
print("Step3. Submit LOAD jobs")
print("=" * 70)

load_jobs = [
    (801, [0, 1], [0, 1]),
    (802, [4, 5], [4, 5]),
    (803, [8, 9], [8, 9]),
]

for job_id, cpu_blocks, gpu_blocks in load_jobs:

    ok = worker.submit_load(
        job_id,
        CPULoadStoreSpec(cpu_blocks),
        GPULoadStoreSpec(
            block_ids=gpu_blocks,
            group_sizes=[len(gpu_blocks)],
            block_indices=[0],
        ),
    )

    print(
        f"submit_load job={job_id} "
        f"cpu={cpu_blocks} gpu={gpu_blocks} ok={ok}"
    )

    assert ok

print("\nWaiting LOAD jobs ...")

worker.wait({801, 802, 803})

finished = worker.get_finished()

print("\nLOAD get_finished():")
for r in finished:
    print(r)

assert len(finished) == 3

finished_by_id = {r.job_id: r for r in finished}

assert set(finished_by_id.keys()) == {801, 802, 803}

for job_id in [801, 802, 803]:

    r = finished_by_id[job_id]

    assert r.success is True
    assert r.transfer_size == 2 * page_size
    assert r.transfer_time >= 0

print("\nPASS: LOAD completion semantics correct.")

########################################################################
# get_finished drain
########################################################################

print("\nChecking drain semantics...")

drained = worker.get_finished()

print("second get_finished():", drained)

assert drained == []

print("PASS: get_finished() drains completed LOAD jobs.")

########################################################################
# Verify restored GPU
########################################################################

print("\n" + "=" * 70)
print("Step4. Verify restored data")
print("=" * 70)

restored = kv_tensor.detach().cpu()

torch.testing.assert_close(
    restored,
    expected,
)

print("PASS: NPU restored correctly.")

########################################################################

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT: PASS")
print("=" * 70)
PY
