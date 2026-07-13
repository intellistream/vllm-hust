# 功能验证说明：
# - 验证真实 KV 数据从 NPU KV tensor 搬运到 CPU offload buffer 的正确性。
# - 给 NPU blocks 写入可识别 pattern，调用 worker.submit_store()。
# - worker.wait() 完成后，逐 block 比较 CPU buffer 与原始 NPU 数据完全一致。
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

#
# 给三个 block 写入容易识别的数据
#
store_gpu_blocks = [2, 3, 4]

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

print("\n" + "=" * 70)
print("Step 5. submit_store()")
print("=" * 70)

cpu_blocks = [0, 1, 2]

src_spec = GPULoadStoreSpec(
    block_ids=store_gpu_blocks,
    group_sizes=[len(store_gpu_blocks)],
    block_indices=[0],
)

dst_spec = CPULoadStoreSpec(cpu_blocks)

job_id = 123

print("submit_store:", store_gpu_blocks, "->", cpu_blocks)

try:
    ret = worker.submit_store(
        job_id,
        src_spec,
        dst_spec,
    )

    print("submit_store returned:", ret)

except Exception:
    print("\nsubmit_store FAILED\n")
    traceback.print_exc()
    worker.shutdown()
    raise

print("\nWaiting...")

worker.wait({job_id})

print("Finished:")
print(worker.get_finished())

print("\n" + "=" * 70)
print("Step 6. Verify CPU Buffer")
print("=" * 70)

cpu_tensor = worker._store_handler.dst_tensors[0]

actual = cpu_tensor[cpu_blocks].clone()

print("Actual:")
print(actual[:, :16])

print("Checking...")

torch.testing.assert_close(actual, expected)

print("\nPASS: CPU data is identical to NPU data.")

worker.shutdown()

print("\n" + "=" * 70)
print("FINAL RESULT: PASS")
print("=" * 70)

PY
