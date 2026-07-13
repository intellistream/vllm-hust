rm -rf /tmp/vllm_kv_tiering_lifecycle_verify
mkdir -p /tmp/vllm_kv_tiering_lifecycle_verify

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
    "lifecycle_idle_ttl_sec": 30,
    "lifecycle_delete_expired_secondary": false,
    "secondary_tiers": [
        {
        "type": "fs",
        "root_dir": "/tmp/vllm_kv_tiering_lifecycle_verify",
        "n_read_threads": 4,
        "n_write_threads": 4
        }
    ]
    }
}'



####2
rm -rf /tmp/vllm_kv_tiering_lifecycle_verify
mkdir -p /tmp/vllm_kv_tiering_lifecycle_verify

ASCEND_RT_VISIBLE_DEVICES=2 \
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
    "lifecycle_idle_ttl_sec": 5,
    "lifecycle_delete_expired_secondary": false,
    "secondary_tiers": [
        {
        "type": "fs",
        "root_dir": "/tmp/vllm_kv_tiering_lifecycle_verify",
        "n_read_threads": 4,
        "n_write_threads": 4
        }
    ]
    }
}'

#####'
rm -rf /tmp/vllm_kv_tiering_lifecycle_verify
mkdir -p /tmp/vllm_kv_tiering_lifecycle_verify

ASCEND_RT_VISIBLE_DEVICES=2 \
VLLM_ASCEND_TORCH_PREFLIGHT=0 \
VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP=1 \
vllm-hust serve /root/models/Qwen2.5-7B-Instruct \
--host 0.0.0.0 \
--port 8081 \
--tensor-parallel-size 1 \
--generation-config vllm \
--no-enable-prefix-caching \
--no-async-scheduling \
--no-enable-chunked-prefill \
--max-num-batched-tokens 32768 \
--kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
    "cpu_bytes_to_use": 1073741824,
    "block_size": 128,
    "spec_name": "TieringOffloadingSpec",
    "debug_log_offload_decisions": true,
    "lifecycle_idle_ttl_sec": 5,
    "lifecycle_delete_expired_secondary": false,
    "secondary_tiers": [
        {
        "type": "fs",
        "root_dir": "/tmp/vllm_kv_tiering_lifecycle_verify",
        "n_read_threads": 4,
        "n_write_threads": 4
        }
    ]
    }
}'