# 功能验证说明：
# - 验证 block_size_factor=2 时的非对齐分层 block 映射。
# - 从 block_indices=[1] 开始 store/load，覆盖跨 CPU offload block 的 sub-block 映射。
# - 同时检查未被目标映射覆盖的 CPU sub-block 保持 sentinel，不被误写。
#
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

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

############################################################
# Configuration
############################################################

page_size_bytes = 512
block_size_factor = 2

num_gpu_blocks = 8
num_cpu_blocks = 8

device = "npu:0"

############################################################
# Build KV cache
############################################################

print("\n" + "=" * 70)
print("Step 3. Build KV Cache")
print("=" * 70)

kv_tensor = torch.zeros(
    (num_gpu_blocks, page_size_bytes),
    dtype=torch.int8,
    device=device,
)

for b in range(num_gpu_blocks):
    kv_tensor[b].fill_(b + 1)

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

worker = create_cpu_offloading_worker(
    kv_caches=kv_caches,
    block_size_factor=block_size_factor,
    num_cpu_blocks=num_cpu_blocks,
)

cpu_tensor = worker._store_handler.dst_tensors[0]
cpu_view = cpu_tensor.view(
    num_cpu_blocks,
    block_size_factor,
    page_size_bytes,
)

############################################################
# Case A
############################################################

print("\n" + "=" * 70)
print("Case A. Unaligned STORE")
print("=" * 70)

cpu_tensor.fill_(-9)

gpu_blocks = [1, 2]
cpu_blocks = [0, 1]

expected1 = kv_tensor[1].detach().cpu().clone()
expected2 = kv_tensor[2].detach().cpu().clone()

store_src = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=[len(gpu_blocks)],
    block_indices=[1],     # <<< 非对齐
)

store_dst = CPULoadStoreSpec(cpu_blocks)

assert worker.submit_store(
    1001,
    store_src,
    store_dst,
)

worker.wait({1001})

print(worker.get_finished())

print()

print("CPU block0 sub0 (should remain -9)")
print(cpu_view[0,0][:16])

print("CPU block0 sub1 (should become block1)")
print(cpu_view[0,1][:16])

print("CPU block1 sub0 (should become block2)")
print(cpu_view[1,0][:16])

print("CPU block1 sub1 (should remain -9)")
print(cpu_view[1,1][:16])

torch.testing.assert_close(
    cpu_view[0,1],
    expected1,
)

torch.testing.assert_close(
    cpu_view[1,0],
    expected2,
)

torch.testing.assert_close(
    cpu_view[0,0],
    torch.full_like(expected1,-9),
)

torch.testing.assert_close(
    cpu_view[1,1],
    torch.full_like(expected1,-9),
)

print("\nPASS: Unaligned STORE mapping correct.")

############################################################
# Case B
############################################################

print("\n" + "=" * 70)
print("Case B. Unaligned LOAD")
print("=" * 70)

for b in gpu_blocks:
    kv_tensor[b].zero_()

torch.npu.synchronize()

print(kv_tensor[gpu_blocks].cpu()[:,:16])

load_src = CPULoadStoreSpec(cpu_blocks)

load_dst = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=[len(gpu_blocks)],
    block_indices=[1],     # <<< 非对齐
)

assert worker.submit_load(
    1002,
    load_src,
    load_dst,
)

worker.wait({1002})

print(worker.get_finished())

restored = kv_tensor[gpu_blocks].detach().cpu()

print()

print(restored[:,:16])

expected = torch.stack([
    expected1,
    expected2,
])

torch.testing.assert_close(
    restored,
    expected,
)

print("\nPASS: Unaligned LOAD mapping correct.")

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT: PASS")
print("=" * 70)

PY
