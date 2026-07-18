# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    tiered_kv_cache_connector as tiered_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.tiered_kv_cache_connector import (
    TieredKVCacheConnector,
)

pytestmark = pytest.mark.cpu_test


def _connector(*, epoch: str | None = "7") -> TieredKVCacheConnector:
    connector = object.__new__(TieredKVCacheConnector)
    connector.scheduler_manager = SimpleNamespace(hash_block_size=16)
    connector._expected_restore_epoch = epoch
    connector._restore_fallback_reasons = {}
    connector._restore_started_at = {}
    connector._restore_transfer_evidence = {}
    return connector


def _request(candidate: dict | None = None):
    params = None if candidate is None else {"tiered_restore": candidate}
    return SimpleNamespace(
        request_id="req-restore",
        num_tokens=65,
        block_hashes=[b"a", b"b", b"c", b"d"],
        kv_transfer_params=params,
    )


def _candidate(**overrides):
    candidate = {
        "request_id": "req-restore",
        "chain_id": "chain-a",
        "start_offset": 0,
        "block_count": 2,
        "block_size_tokens": 16,
        "epoch": "7",
        "block_hashes": [b"a".hex(), b"b".hex()],
        "ready": True,
        "complete": True,
    }
    candidate.update(overrides)
    return candidate


def test_factory_registers_tiered_connector():
    assert (
        KVConnectorFactory.get_connector_class_by_name("TieredKVCacheConnector")
        is TieredKVCacheConnector
    )


def test_candidate_success_caps_registration_to_declared_span(monkeypatch):
    connector = _connector()
    request = _request(_candidate())
    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "get_num_new_matched_tokens",
        lambda self, request, num_computed_tokens: (48, True),
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (32, True)
    assert connector.get_restore_fallback_reason(request) is None


@pytest.mark.parametrize(
    ("overrides", "num_computed_tokens", "reason"),
    [
        ({"ready": False}, 0, "restore_event_not_ready"),
        ({"complete": False}, 0, "restore_span_partial"),
        ({"chain_id": ""}, 0, "hash_chain_identity_missing"),
        ({"epoch": "8"}, 0, "hash_chain_epoch_mismatch"),
        ({"block_size_tokens": 8}, 0, "restore_block_size_mismatch"),
        ({"start_offset": 1}, 0, "restore_span_not_prefill_aligned"),
        ({"start_offset": 0}, 8, "restore_prefix_unaligned"),
        (
            {
                "block_count": 5,
                "block_hashes": [
                    b"a".hex(),
                    b"b".hex(),
                    b"c".hex(),
                    b"d".hex(),
                    b"e".hex(),
                ],
            },
            0,
            "restore_span_exceeds_request_hash_chain",
        ),
        ({"block_hashes": []}, 0, "hash_chain_identity_missing"),
        (
            {"block_hashes": [b"a".hex(), b"wrong".hex()]},
            0,
            "hash_chain_identity_mismatch",
        ),
    ],
)
def test_candidate_validation_fails_closed(
    monkeypatch, overrides, num_computed_tokens, reason
):
    connector = _connector()
    request = _request(_candidate(**overrides))
    parent_called = False

    def parent_match(self, request, num_computed_tokens):
        nonlocal parent_called
        parent_called = True
        return 32, True

    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "get_num_new_matched_tokens",
        parent_match,
    )

    assert connector.get_num_new_matched_tokens(request, num_computed_tokens) == (
        0,
        False,
    )
    assert connector.get_restore_fallback_reason(request) == reason
    assert not parent_called


@pytest.mark.parametrize(
    ("matched_tokens", "reason"),
    [(0, "restore_payload_unavailable"), (16, "restore_span_partial")],
)
def test_candidate_requires_complete_host_payload(monkeypatch, matched_tokens, reason):
    connector = _connector()
    request = _request(_candidate())
    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "get_num_new_matched_tokens",
        lambda self, request, num_computed_tokens: (matched_tokens, True),
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.get_restore_fallback_reason(request) == reason


def test_request_without_candidate_keeps_native_offload_behavior(monkeypatch):
    connector = _connector()
    request = _request()
    connector._restore_fallback_reasons[request.request_id] = "stale"
    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "get_num_new_matched_tokens",
        lambda self, request, num_computed_tokens: (16, True),
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (16, True)
    assert connector.get_restore_fallback_reason(request) is None


def test_server_provider_builds_candidate_from_native_host_hit(monkeypatch):
    connector = _connector()
    request = _request()
    provider_calls = []

    def provider(request, computed, matched, block_size, epoch):
        provider_calls.append(
            (request.request_id, computed, matched, block_size, epoch)
        )
        return _candidate()

    monkeypatch.setattr(tiered_module, "_restore_candidate_provider", provider)
    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "get_num_new_matched_tokens",
        lambda self, request, num_computed_tokens: (32, True),
    )

    assert connector.has_restore_candidate(request) is True
    assert connector.get_num_new_matched_tokens(request, 0) == (32, True)
    assert provider_calls == [("req-restore", 0, 32, 16, "7")]


def test_restore_completion_emits_request_correlated_telemetry(monkeypatch):
    connector = _connector()
    request = _request(_candidate())
    events = []
    connector.scheduler_manager = SimpleNamespace(
        hash_block_size=16,
        cp_world_size=1,
        cpu_kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(
                        block_size=16,
                        page_size_bytes=4096,
                    )
                )
            ]
        ),
        _reqs_to_load={
            request.request_id: SimpleNamespace(
                transfer_meta=SimpleNamespace(gpu_block_ids=[1, 2])
            )
        },
        _expected_worker_count=1,
        _store_event_pending_counts={},
        _store_event_to_blocks={
            3: SimpleNamespace(gpu_block_ids=[4, 5, 6])
        },
        _store_event_to_reqs={3: ["req-producer"]},
    )
    monkeypatch.setattr(tiered_module, "_restore_evidence_sink", events.append)
    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "get_num_new_matched_tokens",
        lambda self, request, num_computed_tokens: (32, True),
    )
    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "update_state_after_alloc",
        lambda self, request, blocks, num_external_tokens: None,
    )
    monkeypatch.setattr(
        SimpleCPUOffloadConnector,
        "update_connector_output",
        lambda self, output: None,
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (32, True)
    connector.update_state_after_alloc(request, SimpleNamespace(), 32)
    connector.update_connector_output(
        SimpleNamespace(
            finished_recving={request.request_id},
            kv_connector_worker_meta=SimpleNamespace(
                completed_store_events={3: 1}
            ),
        )
    )

    assert [event["event"] for event in events] == [
        "kv_restore_connector_scheduled",
        "kv_store_connector_complete",
        "kv_restore_connector_complete",
    ]
    assert events[-1]["request_id"] == request.request_id
    assert events[-1]["restored_tokens"] == 32
    assert events[-1]["transfer_block_pairs"] == 2
    assert events[-1]["estimated_hbm_host_bytes"] == 8192
    assert events[-1]["executed_hbm_host_bytes"] == 8192
    assert events[-1]["executed_transfer_count"] == 2
    assert events[-1]["traffic_direction"] == "host_to_hbm"
    assert events[-1]["restore_latency_ms"] >= 0
    assert events[-1]["completion_source"] == "worker_finished_recving"
    assert events[1]["request_ids"] == ["req-producer"]
    assert events[1]["executed_hbm_host_bytes"] == 12288
    assert events[1]["executed_transfer_count"] == 3
    assert events[1]["traffic_direction"] == "hbm_to_host"
