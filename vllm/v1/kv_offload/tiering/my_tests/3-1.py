# 功能验证说明：
# - 验证 block_size_factor=2 时的对齐分层 block 映射。
# - CPU 一个 offload block 对应 2 个 NPU/GPU KV blocks。
# - 覆盖 submit_store() 和 submit_load()，确认 CPU sub-block 映射和 NPU 回填都正确。
#
python - <<'PY'
import torch

from vllm.platforms import current_platform
from vllm_ascend.utils import enable_custom_op

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)

from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from vllm.v1.kv_offload.cpu.worker_factory import (
    create_cpu_offloading_worker,
    is_ascend_platform,
)

print("=" * 70)
print("Step 1. Platform")
print("=" * 70)

print("platform:", current_platform.device_type)
print("is_ascend:", is_ascend_platform())

assert current_platform.device_type == "npu"

print("\n" + "=" * 70)
print("Step 2. Enable Custom Op")
print("=" * 70)

assert enable_custom_op()

assert hasattr(torch.ops._C_ascend, "swap_blocks_batch")

print("Custom op loaded.")

print("\n" + "=" * 70)
print("Step 3. Build KV Cache")
print("=" * 70)

num_gpu_blocks = 8
num_cpu_blocks = 8

page_size_bytes = 512
block_size_factor = 2

device = "npu:0"

kv_tensor = torch.zeros(
    (num_gpu_blocks, page_size_bytes),
    dtype=torch.int8,
    device=device,
)

#
# block0 -> 全1
# block1 -> 全2
# block2 -> 全3
# block3 -> 全4
#
for block in range(4):
    kv_tensor[block].fill_(block + 1)

expected = kv_tensor[:4].detach().cpu().clone()

print("Expected:")
print(expected[:, :16])

kv_caches = CanonicalKVCaches(
    tensors=[
        CanonicalKVCacheTensor(
            tensor=kv_tensor,
            page_size_bytes=page_size_bytes,
        )
    ],
    group_data_refs=[
        [
            CanonicalKVCacheRef(
                tensor_idx=0,
                page_size_bytes=page_size_bytes,
            )
        ]
    ],
)

print("\n" + "=" * 70)
print("Step 4. Create Worker")
print("=" * 70)

worker = create_cpu_offloading_worker(
    kv_caches=kv_caches,
    block_size_factor=block_size_factor,
    num_cpu_blocks=num_cpu_blocks,
)

print(type(worker))

cpu_tensor = worker._store_handler.dst_tensors[0]

cpu_view = cpu_tensor.view(
    num_cpu_blocks,
    block_size_factor,
    page_size_bytes,
)

print("CPU tensor shape:", cpu_tensor.shape)
print("CPU view shape :", cpu_view.shape)

print("\n" + "=" * 70)
print("Step 5. submit_store()")
print("=" * 70)

gpu_blocks = [0,1,2,3]
cpu_blocks = [0,1]

src = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=[4],
    block_indices=[0],
)

dst = CPULoadStoreSpec(cpu_blocks)

assert worker.submit_store(100, src, dst)

worker.wait({100})

print(worker.get_finished())

print("\nCPU View:")

print(cpu_view[0,0,:16])
print(cpu_view[0,1,:16])
print(cpu_view[1,0,:16])
print(cpu_view[1,1,:16])

torch.testing.assert_close(cpu_view[0,0], expected[0])
torch.testing.assert_close(cpu_view[0,1], expected[1])
torch.testing.assert_close(cpu_view[1,0], expected[2])
torch.testing.assert_close(cpu_view[1,1], expected[3])

print("\nPASS: Store mapping correct.")

print("\n" + "=" * 70)
print("Step 6. Clear GPU")
print("=" * 70)

for b in gpu_blocks:
    kv_tensor[b].zero_()

torch.npu.synchronize()

cleared = kv_tensor[gpu_blocks].detach().cpu()

print(cleared[:, :16])

torch.testing.assert_close(
    cleared,
    torch.zeros_like(expected),
)

print("PASS: NPU cleared.")

print("\n" + "=" * 70)
print("Step 7. submit_load()")
print("=" * 70)

src = CPULoadStoreSpec(cpu_blocks)

dst = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=[4],
    block_indices=[0],
)

assert worker.submit_load(101, src, dst)

worker.wait({101})

print(worker.get_finished())

print("\n" + "=" * 70)
print("Step 8. Verify")
print("=" * 70)

torch.npu.synchronize()

restored = kv_tensor[gpu_blocks].detach().cpu()

print(restored[:, :16])

torch.testing.assert_close(
    restored,
    expected,
)

print("\nPASS: Load mapping correct.")

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT: PASS")
print("=" * 70)

PY
