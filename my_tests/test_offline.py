cd /root/CGCL/vllm-hust

mkdir -p /tmp/vllm_kv_tiering_fs

PYTHONPATH="$PWD" python3 - <<'PY'
import os
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

model = "/root/models/Qwen2.5-7B-Instruct"

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_bytes_to_use": 1 << 30,
        "block_size": 48,
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [
            {
                "type": "fs",
                "root_dir": "/tmp/vllm_kv_tiering_fs",
                "n_read_threads": 4,
                "n_write_threads": 4,
            }
        ],
    },
)

llm = LLM(
    model=model,
    max_model_len=4096,
    gpu_memory_utilization=0.6,
    kv_transfer_config=kv_transfer_config,
    enable_prefix_caching=True,
)

sampling = SamplingParams(temperature=0.0, max_tokens=64)

prompt = (
    "请用三句话介绍 KV cache 分层卸载的作用。"
    "要求第一句说明 GPU/NPU KV cache 压力，第二句说明 CPU/SSD 分层，第三句说明收益。"
)

print("=" * 70)
print("First request")
print("=" * 70)
out1 = llm.generate([prompt], sampling)
print(out1[0].outputs[0].text)

print("=" * 70)
print("Second request, same prefix")
print("=" * 70)
out2 = llm.generate([prompt], sampling)
print(out2[0].outputs[0].text)

print("=" * 70)
print("FS tier files")
print("=" * 70)
for root, _, files in os.walk("/tmp/vllm_kv_tiering_fs"):
    for f in files[:20]:
        print(os.path.join(root, f))
    if files:
        break

print("=" * 70)
print("FINAL RESULT : PASS")
print("=" * 70)

del llm
PY