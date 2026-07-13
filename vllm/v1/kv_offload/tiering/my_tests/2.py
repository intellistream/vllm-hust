# 功能验证说明：
# - 验证真实 KV 数据 NPU -> CPU -> NPU 的完整闭环。
# - 先 submit_store() 验证 CPU buffer 内容正确，再清零 NPU blocks。
# - 调用 submit_load() 后验证 NPU KV tensor 恢复为原始 pattern。
#
python - <<'PY'
import traceback
import torch

from vllm.platforms import current_platform
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
from vllm_ascend.utils import enable_custom_op

print("=" * 70)
print("Step 1. Platform Check")
print("=" * 70)

print("platform:", current_platform.device_type)
print("is_ascend:", is_ascend_platform())

assert current_platform.device_type == "npu"

print("\n" + "=" * 70)
print("Step 2. Enable Custom Ops")
print("=" * 70)

ok = enable_custom_op()
print("enable_custom_op:", ok)
assert ok

print("swap_blocks_batch packet:")
print(torch.ops._C_ascend.swap_blocks_batch)

print("\nDispatcher contains:")
for name in sorted(torch._C._dispatch_get_all_op_names()):
    if "swap_blocks_batch" in name:
        print(" ", name)

print("\n" + "=" * 70)
print("Step 3. Build KV Cache")
print("=" * 70)

num_gpu_blocks = 16
num_cpu_blocks = 32
page_size_bytes = 512
block_size_factor = 1

device = "npu:0"

kv_tensor = torch.zeros(
    (num_gpu_blocks, page_size_bytes),
    dtype=torch.int8,
    device=device,
)

store_gpu_blocks = [2, 3, 4]
cpu_blocks = [0, 1, 2]

for value, block in enumerate(store_gpu_blocks, start=1):
    row = torch.full(
        (page_size_bytes,),
        value,
        dtype=torch.int8,
    )
    kv_tensor[block].copy_(row.to(device))

expected = kv_tensor[store_gpu_blocks].detach().cpu().clone()

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

#
# ---------------------------------------------------------------------
# NPU -> CPU
# ---------------------------------------------------------------------
#

print("\n" + "=" * 70)
print("Step 5. submit_store()")
print("=" * 70)

store_src = GPULoadStoreSpec(
    block_ids=store_gpu_blocks,
    group_sizes=[len(store_gpu_blocks)],
    block_indices=[0],
)

store_dst = CPULoadStoreSpec(cpu_blocks)

job_id = 123

print("submit_store:", store_gpu_blocks, "->", cpu_blocks)

try:
    ret = worker.submit_store(
        job_id,
        store_src,
        store_dst,
    )
    print("submit_store returned:", ret)

except Exception:
    traceback.print_exc()
    worker.shutdown()
    raise

worker.wait({job_id})

print("Finished:")
print(worker.get_finished())

cpu_tensor = worker._store_handler.dst_tensors[0]

actual_cpu = cpu_tensor[cpu_blocks].clone()

print("\nVerify CPU Buffer")

torch.testing.assert_close(
    actual_cpu,
    expected,
)

print("PASS: CPU buffer is correct.")

#
# ---------------------------------------------------------------------
# Clear NPU
# ---------------------------------------------------------------------
#

print("\n" + "=" * 70)
print("Step 6. Clear NPU")
print("=" * 70)

for block in store_gpu_blocks:
    kv_tensor[block].zero_()

torch.npu.synchronize()

cleared = kv_tensor[store_gpu_blocks].detach().cpu().clone()

print("Cleared NPU:")
print(cleared[:, :16])

torch.testing.assert_close(
    cleared,
    torch.zeros_like(expected),
)

print("\nPASS: NPU blocks cleared.")

#
# ---------------------------------------------------------------------
# CPU -> NPU
# ---------------------------------------------------------------------
#

print("\n" + "=" * 70)
print("Step 7. submit_load()")
print("=" * 70)

load_job = 456

load_src = CPULoadStoreSpec(cpu_blocks)

load_dst = GPULoadStoreSpec(
    block_ids=store_gpu_blocks,
    group_sizes=[len(store_gpu_blocks)],
    block_indices=[0],
)

print("submit_load:", cpu_blocks, "->", store_gpu_blocks)

try:
    ret = worker.submit_load(
        load_job,
        load_src,
        load_dst,
    )

    print("submit_load returned:", ret)

except Exception:
    traceback.print_exc()
    worker.shutdown()
    raise

worker.wait({load_job})

print("Finished:")
print(worker.get_finished())

#
# ---------------------------------------------------------------------
# Verify NPU
# ---------------------------------------------------------------------
#

print("\n" + "=" * 70)
print("Step 8. Verify Restored NPU")
print("=" * 70)

restored = kv_tensor[store_gpu_blocks].detach().cpu().clone()

print(restored[:, :16])

torch.testing.assert_close(
    restored,
    expected,
)

print("PASS: NPU tensor restored correctly.")

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT: PASS")
print("=" * 70)
PY
