# 功能验证说明：
# - 验证连续提交多个 STORE job 时的异步队列和完成语义。
# - 检查 get_finished() 返回的 job_id 集合和 success 状态。
# - 验证每个 CPU block 收到对应 NPU block 数据，且未触碰 CPU blocks 保持 sentinel。
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
print("Test5: Multiple Store Jobs Queue")
print("=" * 70)

assert current_platform.device_type == "npu"
assert is_ascend_platform()
assert enable_custom_op()
assert hasattr(torch.ops._C_ascend, "swap_blocks_batch")

page_size = 512
num_gpu_blocks = 12
num_cpu_blocks = 12
device = "npu:0"

########################################################################
# Build KV cache
########################################################################

kv_tensor = torch.zeros(
    (num_gpu_blocks, page_size),
    dtype=torch.int8,
    device=device,
)

# 每个 block 使用不同 pattern
for block in range(num_gpu_blocks):
    kv_tensor[block].fill_(block + 1)

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
    block_size_factor=1,
    num_cpu_blocks=num_cpu_blocks,
)

cpu_tensor = worker._store_handler.dst_tensors[0]

cpu_tensor.fill_(-9)

########################################################################
# Build three jobs
########################################################################

jobs = [
    (
        501,
        [0, 1],
        [0, 1],
    ),
    (
        502,
        [4, 5],
        [4, 5],
    ),
    (
        503,
        [8, 9],
        [8, 9],
    ),
]

print("\nSubmitting jobs ...")

for job_id, gpu_blocks, cpu_blocks in jobs:

    src = GPULoadStoreSpec(
        block_ids=gpu_blocks,
        group_sizes=[len(gpu_blocks)],
        block_indices=[0],
    )

    dst = CPULoadStoreSpec(cpu_blocks)

    ok = worker.submit_store(job_id, src, dst)

    print(
        f"job={job_id}",
        "gpu=", gpu_blocks,
        "cpu=", cpu_blocks,
        "submit=", ok,
    )

    assert ok

########################################################################
# Wait all together
########################################################################

print("\nWaiting all jobs...")

worker.wait({501, 502, 503})

finished = worker.get_finished()

print("\nFinished jobs:")

for r in finished:
    print(r)

finished_by_id = {r.job_id: r for r in finished}

assert set(finished_by_id) == {501, 502, 503}

assert all(r.success for r in finished)

print("\nPASS: all jobs completed.")

########################################################################
# Verify CPU data
########################################################################

print("\nVerifying CPU blocks...")

for _, gpu_blocks, cpu_blocks in jobs:

    for g, c in zip(gpu_blocks, cpu_blocks):

        actual = cpu_tensor[c].cpu()

        expected = kv_tensor[g].cpu()

        torch.testing.assert_close(
            actual,
            expected,
        )

        print(
            f"CPU block {c} <- GPU block {g} PASS"
        )

########################################################################
# Verify untouched blocks remain sentinel
########################################################################

print("\nChecking untouched CPU blocks...")

for block in [2,3,6,7,10,11]:

    torch.testing.assert_close(
        cpu_tensor[block].cpu(),
        torch.full_like(cpu_tensor[block].cpu(), -9),
    )

print("PASS: untouched CPU blocks unchanged.")

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT: PASS")
print("=" * 70)

PY
