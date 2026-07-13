# 功能验证说明：
# - 验证 block_size_factor=4 时的边界 sub-block 映射。
# - 覆盖只写入 CPU offload block 最后一个 sub-block 的 store/load 场景。
# - 检查目标 sub-block 数据正确，其他 sub-block 保持 sentinel，不被误写。
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
print("Test6: block_size_factor=4 Boundary Mapping")
print("=" * 70)

###########################################################
# Platform
###########################################################

assert current_platform.device_type == "npu"
assert is_ascend_platform()
assert enable_custom_op()
assert hasattr(torch.ops._C_ascend, "swap_blocks_batch")

print("Platform OK.")

###########################################################
# Parameters
###########################################################

page_size = 512
block_size_factor = 4

num_gpu_blocks = 8
num_cpu_blocks = 4

device = "npu:0"

###########################################################
# Build KV Cache
###########################################################

kv_tensor = torch.zeros(
    (num_gpu_blocks, page_size),
    dtype=torch.int8,
    device=device,
)

# block0=1 block1=2 block2=3 block3=4 ...
for block in range(num_gpu_blocks):
    kv_tensor[block].fill_(block + 1)

expected = kv_tensor[3].detach().cpu().clone()

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

###########################################################
# Worker
###########################################################

worker = create_cpu_offloading_worker(
    kv_caches=kv_caches,
    block_size_factor=block_size_factor,
    num_cpu_blocks=num_cpu_blocks,
)

cpu_tensor = worker._store_handler.dst_tensors[0]

cpu_view = cpu_tensor.view(
    num_cpu_blocks,
    block_size_factor,
    page_size,
)

print("CPU tensor shape :", cpu_tensor.shape)
print("CPU view shape   :", cpu_view.shape)

###########################################################
# Fill sentinel
###########################################################

cpu_tensor.fill_(-9)

###########################################################
# STORE
###########################################################

print("\n" + "=" * 70)
print("Step1 STORE")
print("=" * 70)

gpu_blocks = [3]

cpu_blocks = [0]

store_src = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=[1],
    block_indices=[3],
)

store_dst = CPULoadStoreSpec(cpu_blocks)

assert worker.submit_store(
    601,
    store_src,
    store_dst,
)

worker.wait({601})

print(worker.get_finished())

###########################################################
# Verify mapping
###########################################################

print("\nVerify CPU mapping")

print("sub0:")
print(cpu_view[0,0,:16])

print("sub1:")
print(cpu_view[0,1,:16])

print("sub2:")
print(cpu_view[0,2,:16])

print("sub3:")
print(cpu_view[0,3,:16])

sentinel = torch.full(
    (page_size,),
    -9,
    dtype=torch.int8,
)

torch.testing.assert_close(cpu_view[0,0], sentinel)
torch.testing.assert_close(cpu_view[0,1], sentinel)
torch.testing.assert_close(cpu_view[0,2], sentinel)

torch.testing.assert_close(
    cpu_view[0,3],
    expected,
)

print("PASS: Store boundary mapping correct.")

###########################################################
# Clear GPU block3
###########################################################

print("\n" + "=" * 70)
print("Step2 Clear GPU")
print("=" * 70)

kv_tensor[3].zero_()

torch.npu.synchronize()

cleared = kv_tensor[3].detach().cpu()

print(cleared[:16])

torch.testing.assert_close(
    cleared,
    torch.zeros_like(cleared),
)

print("PASS: GPU block cleared.")

###########################################################
# LOAD
###########################################################

print("\n" + "=" * 70)
print("Step3 LOAD")
print("=" * 70)

load_src = CPULoadStoreSpec(cpu_blocks)

load_dst = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=[1],
    block_indices=[3],
)

assert worker.submit_load(
    602,
    load_src,
    load_dst,
)

worker.wait({602})

print(worker.get_finished())

###########################################################
# Verify restored
###########################################################

print("\nVerify restored GPU")

restored = kv_tensor[3].detach().cpu()

print(restored[:16])

torch.testing.assert_close(
    restored,
    expected,
)

print("PASS: Load boundary mapping correct.")

###########################################################
# Verify other sub-blocks unchanged
###########################################################

print("\nVerify untouched sub-blocks")

torch.testing.assert_close(cpu_view[0,0], sentinel)
torch.testing.assert_close(cpu_view[0,1], sentinel)
torch.testing.assert_close(cpu_view[0,2], sentinel)

print("PASS: Other sub-blocks remain untouched.")

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT: PASS")
print("=" * 70)

PY
