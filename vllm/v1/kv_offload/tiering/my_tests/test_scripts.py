python3 - <<'PY'
from vllm.config import KVTransferConfig
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

class DummyVllmConfig:
    pass

cfg = DummyVllmConfig()
cfg.kv_transfer_config = KVTransferConfig(
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

spec_cls = OffloadingSpecFactory.get_spec_cls(cfg)

print("factory spec:", spec_cls)
assert spec_cls is TieringOffloadingSpec

print("kv_connector:", cfg.kv_transfer_config.kv_connector)
print("kv_role:", cfg.kv_transfer_config.kv_role)
print("extra_config:", cfg.kv_transfer_config.kv_connector_extra_config)

print("=" * 70)
print("PASS: TieringOffloadingSpec is reachable through vLLM kv_transfer_config")
print("FINAL RESULT : PASS")
print("=" * 70)
PY