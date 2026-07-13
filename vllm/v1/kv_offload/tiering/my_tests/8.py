# 功能验证说明：
# - 验证 TieringOffloadingManager 的会话级 idle 生命周期元数据。
# - TTL 默认关闭时，idle retained 会话不会过期。
# - 显式配置 TTL 后，会话从 active -> idle_retained -> expired/deleted。
# - 仅在 lifecycle_delete_expired_secondary=True 时删除 FS 二级块文件。
#
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
PYTHONPATH="$REPO_ROOT" python3 - <<'PY'
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.lifecycle import (
    LifecycleConfig,
    LifecycleStatus,
    SessionLifecycleManager,
    get_session_id,
)
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

LINE = "=" * 70


def banner(step: str) -> None:
    print(LINE)
    print(step)
    print(LINE)


def pass_msg(msg: str) -> None:
    print(f"PASS {msg}")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def key(i: int):
    return make_offload_key(str(i).encode(), 0)


def wait_until_expired(manager: SessionLifecycleManager, tiers) -> int:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        expired = manager.expire_idle_sessions(tiers)
        if expired:
            return expired
        time.sleep(0.02)
    return 0


class FakeFileMapper:
    def __init__(self, root: Path):
        self.root = root

    def get_file_name(self, block_key) -> str:
        return str(self.root / f"{bytes(block_key).hex()}.bin")


class FakeFSTier:
    def __init__(self, root: Path):
        self.file_mapper = FakeFileMapper(root)


banner("Step1. Session id selection")
ctx = ReqContext(
    req_id="req_fallback",
    kv_transfer_params={"session_id": "session_a"},
)
require(get_session_id(ctx) == "session_a", "session_id was not selected")
require(
    get_session_id(ReqContext(req_id="req_only", kv_transfer_params=None))
    == "req_only",
    "req_id fallback failed",
)
pass_msg("get_session_id selects explicit session id and falls back to req_id")


banner("Step2. Default TTL disabled")
manager = SessionLifecycleManager(LifecycleConfig())
req = ReqContext(req_id="req_1", kv_transfer_params={"session_id": "chat_1"})
manager.on_new_request(req)
manager.record_request_keys(req, [key(1), key(2)])
manager.on_request_finished(req)

snapshot = manager.snapshot()
require(snapshot["sessions"] == 1, f"unexpected session count: {snapshot}")
require(snapshot["idle_sessions"] == 1, f"session did not become idle: {snapshot}")
require(snapshot["retained_blocks"] == 2, f"block keys not retained: {snapshot}")
time.sleep(0.05)
require(
    manager.expire_idle_sessions([]) == 0,
    "default lifecycle TTL should not expire idle sessions",
)
require(
    manager.snapshot()["sessions"] == 1,
    "default lifecycle TTL removed a retained session",
)
pass_msg("idle sessions are retained indefinitely when TTL is disabled")


banner("Step3. Explicit TTL expiration")
manager = SessionLifecycleManager(LifecycleConfig(idle_ttl_sec=0.05))
req = ReqContext(req_id="req_2", kv_transfer_params={"conversation_id": "chat_2"})
manager.on_new_request(req)
manager.record_request_keys(req, [key(3)])
manager.on_request_finished(req)

expired = wait_until_expired(manager, [])
require(expired == 1, f"expected exactly one expired session, got {expired}")
require(
    manager.snapshot()["sessions"] == 0,
    "expired lifecycle session metadata was not removed",
)
pass_msg("explicit TTL expires idle retained session metadata")


banner("Step4. Reusing one session across request ids")
manager = SessionLifecycleManager(LifecycleConfig(idle_ttl_sec=0.05))
req_a = ReqContext(req_id="turn_1", kv_transfer_params={"kv_session_id": "chat_3"})
req_b = ReqContext(req_id="turn_2", kv_transfer_params={"kv_session_id": "chat_3"})
manager.on_new_request(req_a)
manager.record_request_keys(req_a, [key(4)])
manager.on_request_finished(req_a)
manager.on_new_request(req_b)

snapshot = manager.snapshot()
require(snapshot["sessions"] == 1, f"session was not reused: {snapshot}")
require(snapshot["active_sessions"] == 1, f"session was not reactivated: {snapshot}")
manager.record_request_keys(req_b, [key(5)])
manager.on_request_finished(req_b)
require(
    manager.snapshot()["retained_blocks"] == 2,
    "retained block keys from multiple turns were not merged",
)
pass_msg("multiple request ids can share one lifecycle session")


banner("Step5. Optional secondary file deletion")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    tier = FakeFSTier(root)
    retained_key = key(6)
    path = Path(tier.file_mapper.get_file_name(retained_key))
    path.write_bytes(b"kv-block")

    manager = SessionLifecycleManager(
        LifecycleConfig(
            idle_ttl_sec=0.05,
            delete_expired_secondary=True,
        )
    )
    req = ReqContext(req_id="req_3", kv_transfer_params={"session_id": "chat_4"})
    manager.on_new_request(req)
    manager.record_request_keys(req, [retained_key])
    manager.on_request_finished(req)

    expired = wait_until_expired(manager, [tier])
    require(expired == 1, f"expected one deleted session, got {expired}")
    require(not path.exists(), "expired secondary FS block file was not deleted")

pass_msg("secondary files are deleted only when deletion is explicitly enabled")


banner("Step6. Shared secondary block deletion protection")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    tier = FakeFSTier(root)
    shared_key = key(8)
    unique_key = key(9)
    shared_path = Path(tier.file_mapper.get_file_name(shared_key))
    unique_path = Path(tier.file_mapper.get_file_name(unique_key))
    shared_path.write_bytes(b"shared-kv-block")
    unique_path.write_bytes(b"unique-kv-block")

    manager = SessionLifecycleManager(
        LifecycleConfig(
            idle_ttl_sec=0.05,
            delete_expired_secondary=True,
        )
    )
    req_a = ReqContext(req_id="req_shared_a", kv_transfer_params={"session_id": "a"})
    req_b = ReqContext(req_id="req_shared_b", kv_transfer_params={"session_id": "b"})
    manager.on_new_request(req_a)
    manager.record_request_keys(req_a, [shared_key, unique_key])
    manager.on_request_finished(req_a)
    manager.on_new_request(req_b)
    manager.record_request_keys(req_b, [shared_key])

    expired = wait_until_expired(manager, [tier])
    require(expired == 1, f"expected only idle session a to expire, got {expired}")
    require(shared_path.exists(), "shared block was deleted while session b used it")
    require(not unique_path.exists(), "unshared expired block was not deleted")

pass_msg("shared secondary blocks are retained while another session references them")


banner("Step7. Pending expiration keeps scheduler work alive")
manager = SessionLifecycleManager(LifecycleConfig(idle_ttl_sec=1.0))
req = ReqContext(req_id="req_pending_ttl", kv_transfer_params={"session_id": "chat_6"})
manager.on_new_request(req)
manager.on_request_finished(req)
require(manager.has_pending_expiration(), "idle TTL deadline was not reported pending")
manager = SessionLifecycleManager(LifecycleConfig())
req = ReqContext(req_id="req_no_ttl", kv_transfer_params={"session_id": "chat_7"})
manager.on_new_request(req)
manager.on_request_finished(req)
require(
    not manager.has_pending_expiration(),
    "disabled TTL should not report pending lifecycle work",
)
pass_msg("lifecycle TTL exposes pending work only when expiration is enabled")


banner("Step8. Tiering manager reports lifecycle pending work")
tiering_manager = TieringOffloadingManager(
    primary_tier=MagicMock(),
    secondary_tiers=[],
    lifecycle_config=LifecycleConfig(idle_ttl_sec=1.0),
)
req = ReqContext(req_id="req_tiering_pending", kv_transfer_params=None)
tiering_manager.on_new_request(req)
tiering_manager.on_request_finished(req)
require(
    tiering_manager.has_pending_work(),
    "TieringOffloadingManager did not propagate lifecycle pending work",
)
pass_msg("tiering manager keeps empty scheduler ticks alive for lifecycle expiry")


banner("Step9. Reset active primary state")
manager = SessionLifecycleManager(LifecycleConfig(idle_ttl_sec=1.0))
req = ReqContext(req_id="req_4", kv_transfer_params={"session_id": "chat_5"})
manager.on_new_request(req)
manager.record_request_keys(req, [key(7)])
manager.reset_active_primary_state()
snapshot = manager.snapshot()
require(snapshot["idle_sessions"] == 1, f"active session was not retained: {snapshot}")
require(snapshot["retained_blocks"] == 1, f"retained keys lost on reset: {snapshot}")
pass_msg("reset keeps retained metadata while clearing active request mappings")

banner("FINAL RESULT : PASS")
PY
