#sever
##################################
rm -rf /tmp/vllm_kv_tiering_fs_verify
mkdir -p /tmp/vllm_kv_tiering_fs_verify

ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP=1 \
vllm-hust serve /root/models/Qwen2.5-7B-Instruct \
--host 0.0.0.0 \
--port 8081 \
--tensor-parallel-size 1 \
--generation-config vllm \
--kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
    "cpu_bytes_to_use": 1073741824,
    "block_size": 128,
    "spec_name": "TieringOffloadingSpec",
    "secondary_tiers": [
        {
        "type": "fs",
        "root_dir": "/tmp/vllm_kv_tiering_fs_verify",
        "n_read_threads": 4,
        "n_write_threads": 4
        }
    ]
    }
}'

############
ASCEND_RT_VISIBLE_DEVICES=1 \
VLLM_ASCEND_TORCH_PREFLIGHT=0 \
VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP=1 \
vllm-hust serve /root/models/Qwen2.5-7B-Instruct \
--host 0.0.0.0 \
--port 8081 \
--tensor-parallel-size 1 \
--generation-config vllm \
--kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
    "cpu_bytes_to_use": 1073741824,
    "block_size": 128,
    "spec_name": "TieringOffloadingSpec",
    "secondary_tiers": [
        {
        "type": "fs",
        "root_dir": "/tmp/vllm_kv_tiering_fs_verify",
        "n_read_threads": 4,
        "n_write_threads": 4
        }
    ]
    }
}'

##########


#client
python3 - <<'PY'
import json
import urllib.request

url = "http://127.0.0.1:8081/v1/chat/completions"

payload = {
    "model": "/root/models/Qwen2.5-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "介绍一下KV cache，并说明它为什么会影响大模型推理吞吐。"
        }
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=300) as resp:
    print(resp.read().decode("utf-8"))
PY