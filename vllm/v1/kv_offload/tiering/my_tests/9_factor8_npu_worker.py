# 功能验证说明：
# - 验证更接近在线 Qwen 配置的 NPU <-> CPU KV 搬运映射。
# - 覆盖 block_size_factor=8、多 KV tensor、多 group、非 0 block_indices。
# - 该测试用于排查重复请求命中 offload 后输出异常是否来自 worker sub-block 映射。
#
python - <<'PY'
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

LINE = "=" * 70


def banner(name: str) -> None:
    print(LINE)
    print(name)
    print(LINE)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


banner("Step1. Platform")
print("platform:", current_platform.device_type)
print("is_ascend:", is_ascend_platform())
require(current_platform.device_type == "npu", "this test requires Ascend NPU")
require(is_ascend_platform(), "Ascend platform was not detected")
require(enable_custom_op(), "failed to enable vllm-ascend custom ops")
require(hasattr(torch.ops._C_ascend, "swap_blocks_batch"), "missing swap_blocks_batch")

banner("Step2. Build multi-tensor KV cache")
device = "npu:0"
num_gpu_blocks = 32
num_cpu_blocks = 16
page_size = 1024
block_size_factor = 8
num_tensors = 6

kv_tensors = []
originals = []
for tid in range(num_tensors):
    tensor = torch.empty(
        (num_gpu_blocks, page_size),
        dtype=torch.int8,
        device=device,
    )
    for bid in range(num_gpu_blocks):
        # Use a non-constant byte pattern so shifted or repeated copies fail.
        row = ((torch.arange(page_size, dtype=torch.int16) + tid * 17 + bid * 3) % 127)
        row = row.to(torch.int8).to(device)
        tensor[bid].copy_(row)
    kv_tensors.append(tensor)
    originals.append(tensor.detach().cpu().clone())

kv_caches = CanonicalKVCaches(
    tensors=[
        CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_size)
        for tensor in kv_tensors
    ],
    group_data_refs=[
        [
            CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size),
            CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=page_size),
            CanonicalKVCacheRef(tensor_idx=2, page_size_bytes=page_size),
        ],
        [
            CanonicalKVCacheRef(tensor_idx=3, page_size_bytes=page_size),
            CanonicalKVCacheRef(tensor_idx=4, page_size_bytes=page_size),
            CanonicalKVCacheRef(tensor_idx=5, page_size_bytes=page_size),
        ],
    ],
)

worker = create_cpu_offloading_worker(
    kv_caches=kv_caches,
    block_size_factor=block_size_factor,
    num_cpu_blocks=num_cpu_blocks,
)

cpu_tensors = worker._store_handler.dst_tensors
for cpu_tensor in cpu_tensors:
    cpu_tensor.fill_(-9)

banner("Step3. Store factor=8 aligned and unaligned groups")
# Group 0 stores logical GPU blocks 0..15 into CPU blocks 0..1.
# Group 1 stores logical GPU blocks 5..17 into CPU blocks 2..4, which is
# intentionally unaligned and crosses three CPU offload blocks.
gpu_blocks = list(range(0, 16)) + list(range(5, 18))
group_sizes = [16, 13]
block_indices = [0, 5]
cpu_blocks = [0, 1, 2, 3, 4]

store_src = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=group_sizes,
    block_indices=block_indices,
)
store_dst = CPULoadStoreSpec(cpu_blocks)

require(worker.submit_store(9001, store_src, store_dst), "submit_store failed")
worker.wait({9001})
finished = worker.get_finished()
print("finished:", finished)
require(len(finished) == 1 and finished[0].success, "store did not finish cleanly")

banner("Step4. Verify CPU sub-block layout")
cpu_views = [
    cpu.view(num_cpu_blocks, block_size_factor, page_size)
    for cpu in cpu_tensors
]

# Group 0: tensors 0..2, CPU blocks 0..1 sub-blocks 0..7.
for tid in range(3):
    for logical_block in range(16):
        cpu_block = logical_block // block_size_factor
        sub_block = logical_block % block_size_factor
        torch.testing.assert_close(
            cpu_views[tid][cpu_block, sub_block],
            originals[tid][logical_block],
        )

# Group 1: tensors 3..5, logical blocks 5..17 begin at CPU block 2 sub-block 5.
for tid in range(3, 6):
    for logical_block in range(5, 18):
        rel = logical_block
        cpu_block = 2 + rel // block_size_factor
        sub_block = rel % block_size_factor
        torch.testing.assert_close(
            cpu_views[tid][cpu_block, sub_block],
            originals[tid][logical_block],
        )
    # untouched sub-blocks before logical block 5 remain sentinel.
    for sub_block in range(5):
        torch.testing.assert_close(
            cpu_views[tid][2, sub_block],
            torch.full((page_size,), -9, dtype=torch.int8),
        )

print("PASS: CPU layout is correct")

banner("Step5. Clear target NPU blocks")
for tid in range(3):
    kv_tensors[tid][0:16].zero_()
for tid in range(3, 6):
    kv_tensors[tid][5:18].zero_()
torch.npu.synchronize()

banner("Step6. Load back to NPU")
load_src = CPULoadStoreSpec(cpu_blocks)
load_dst = GPULoadStoreSpec(
    block_ids=gpu_blocks,
    group_sizes=group_sizes,
    block_indices=block_indices,
)

require(worker.submit_load(9002, load_src, load_dst), "submit_load failed")
worker.wait({9002})
finished = worker.get_finished()
print("finished:", finished)
require(len(finished) == 1 and finished[0].success, "load did not finish cleanly")
torch.npu.synchronize()

banner("Step7. Verify restored NPU data")
for tid in range(3):
    torch.testing.assert_close(
        kv_tensors[tid][0:16].detach().cpu(),
        originals[tid][0:16],
    )
for tid in range(3, 6):
    torch.testing.assert_close(
        kv_tensors[tid][5:18].detach().cpu(),
        originals[tid][5:18],
    )

worker.shutdown()
print("FINAL RESULT: PASS")
PY
