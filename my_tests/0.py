# 功能验证说明：
# - 验证 KV offload CPU worker 在 Ascend/NPU 平台上的基础初始化路径。
# - 确认 create_cpu_offloading_worker() 能选择 Ascend 相关 worker 实现。
# - 确认 CPU offload buffer 被正确分配，并可通过 worker._store_handler.dst_tensors 访问。
#
python - <<'PY'
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.cpu.worker_factory import (
    create_cpu_offloading_worker,
    is_ascend_platform,
)

print("platform:", current_platform.device_type)
print("is_ascend:", is_ascend_platform())

num_gpu_blocks = 16
num_cpu_blocks = 32
page_size_bytes = 512
block_size_factor = 1

device = (
    "npu:0"
    if current_platform.device_type == "npu"
    else f"{current_platform.device_type}:0"
)

kv_tensor = torch.zeros(
    (num_gpu_blocks, page_size_bytes),
    dtype=torch.int8,
    device=device,
)

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

print("worker class:", type(worker))

print("\nstore handler attrs:")
print(dir(worker._store_handler))

print("\ndst tensors:")
print(worker._store_handler.dst_tensors)

cpu_tensor = worker._store_handler.dst_tensors[0]

print("\ncpu tensor:")
print("shape =", cpu_tensor.shape)
print("device =", cpu_tensor.device)
print("dtype =", cpu_tensor.dtype)

worker.shutdown()

print("\nPASS")
PY
