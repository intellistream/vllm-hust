# 功能验证说明：
# - 验证多 KV tensor / 多 group 场景下的真实数据搬运。
# - 构造 4 个 KV tensor，2 个 group，每个 group 引用多个 tensor。
# - 覆盖 block_size_factor=2、group_sizes、block_indices 的组合映射，并验证 store/load 恢复。
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
print("Step1 Platform")
print("=" * 70)

assert current_platform.device_type == "npu"
assert is_ascend_platform()
assert enable_custom_op()
assert hasattr(torch.ops._C_ascend, "swap_blocks_batch")

page_size = 512
block_size_factor = 2

num_gpu_blocks = 8
num_cpu_blocks = 8

device = "npu:0"

###########################################################
# build four tensors
###########################################################

kv_tensors = []

for tid in range(4):

    t = torch.zeros(
        (num_gpu_blocks, page_size),
        dtype=torch.int8,
        device=device,
    )

    for block in range(num_gpu_blocks):
        t[block].fill_((tid + 1) * 10 + block)

    kv_tensors.append(t)

kv_caches = CanonicalKVCaches(
    tensors=[
        CanonicalKVCacheTensor(
            tensor=t,
            page_size_bytes=page_size,
        )
        for t in kv_tensors
    ],
    group_data_refs=[
        [
            CanonicalKVCacheRef(
                tensor_idx=0,
                page_size_bytes=page_size,
            ),
            CanonicalKVCacheRef(
                tensor_idx=1,
                page_size_bytes=page_size,
            ),
        ],
        [
            CanonicalKVCacheRef(
                tensor_idx=2,
                page_size_bytes=page_size,
            ),
            CanonicalKVCacheRef(
                tensor_idx=3,
                page_size_bytes=page_size,
            ),
        ],
    ],
)

worker = create_cpu_offloading_worker(
    kv_caches=kv_caches,
    block_size_factor=block_size_factor,
    num_cpu_blocks=num_cpu_blocks,
)

cpu_tensors = worker._store_handler.dst_tensors

cpu_views = [
    t.view(num_cpu_blocks, block_size_factor, page_size)
    for t in cpu_tensors
]

print("worker:", type(worker))
print("cpu tensors:", len(cpu_tensors))

###########################################################
# STORE
###########################################################

print("\n" + "=" * 70)
print("STORE")
print("=" * 70)

for cpu in cpu_tensors:
    cpu.fill_(-9)

gpu_blocks = [0,1,1,2]

group_sizes = [2,2]

block_indices = [0,1]

cpu_blocks = [0,1,2]

store_src = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=group_sizes,
    block_indices=block_indices,
)

store_dst = CPULoadStoreSpec(cpu_blocks)

assert worker.submit_store(
    100,
    store_src,
    store_dst,
)

worker.wait({100})

print(worker.get_finished())

###########################################################
# verify group0
###########################################################

print("\nVerify Group0")

for tid in [0,1]:

    torch.testing.assert_close(
        cpu_views[tid][0,0],
        kv_tensors[tid][0].cpu(),
    )

    torch.testing.assert_close(
        cpu_views[tid][0,1],
        kv_tensors[tid][1].cpu(),
    )

print("PASS")

###########################################################
# verify group1
###########################################################

print("\nVerify Group1")

for tid in [2,3]:

    torch.testing.assert_close(
        cpu_views[tid][1,0],
        torch.full_like(cpu_views[tid][1,0],-9),
    )

    torch.testing.assert_close(
        cpu_views[tid][1,1],
        kv_tensors[tid][1].cpu(),
    )

    torch.testing.assert_close(
        cpu_views[tid][2,0],
        kv_tensors[tid][2].cpu(),
    )

    torch.testing.assert_close(
        cpu_views[tid][2,1],
        torch.full_like(cpu_views[tid][2,1],-9),
    )

print("PASS")

###########################################################
# clear gpu
###########################################################

for tid in [0,1]:
    kv_tensors[tid][0].zero_()
    kv_tensors[tid][1].zero_()

for tid in [2,3]:
    kv_tensors[tid][1].zero_()
    kv_tensors[tid][2].zero_()

torch.npu.synchronize()

###########################################################
# LOAD
###########################################################

print("\n" + "=" * 70)
print("LOAD")
print("=" * 70)

load_src = CPULoadStoreSpec(cpu_blocks)

load_dst = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=group_sizes,
    block_indices=block_indices,
)

assert worker.submit_load(
    101,
    load_src,
    load_dst,
)

worker.wait({101})

print(worker.get_finished())

###########################################################
# verify restored
###########################################################

print("\nVerify Restore")

for tid in [0,1]:

    torch.testing.assert_close(
        kv_tensors[tid][0].cpu(),
        torch.full((page_size,),10*(tid+1),dtype=torch.int8),
    )

    torch.testing.assert_close(
        kv_tensors[tid][1].cpu(),
        torch.full((page_size,),10*(tid+1)+1,dtype=torch.int8),
    )

for tid in [2,3]:

    torch.testing.assert_close(
        kv_tensors[tid][1].cpu(),
        torch.full((page_size,),10*(tid+1)+1,dtype=torch.int8),
    )

    torch.testing.assert_close(
        kv_tensors[tid][2].cpu(),
        torch.full((page_size,),10*(tid+1)+2,dtype=torch.int8),
    )

print("PASS")

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT : PASS")
print("=" * 70)

PY
