# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.v1.core.kv_cache_manager as manager_module
from vllm.v1.core.kv_cache_manager import KVCacheManager

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_request_kv_lease_is_exact_once_and_fatal_shutdown_invalidates(monkeypatch):
    events: list[tuple] = []
    monkeypatch.setattr(
        manager_module,
        "observe_resource_transition",
        lambda *args: events.append(args),
    )
    manager = KVCacheManager.__new__(KVCacheManager)
    manager._observed_kv_leases = {}
    manager._observed_kv_lease_sequence = 0
    monkeypatch.setattr(manager, "_observed_request_block_count", lambda _request: 7)

    manager._observe_kv_lease_acquired("request-0")
    manager._observe_kv_lease_acquired("request-0")
    manager._observe_kv_lease_released("request-0", 7)
    manager._observe_kv_lease_acquired("request-0")
    manager.invalidate_observed_kv_leases()

    assert [event[3] for event in events] == [
        "acquire",
        "release",
        "acquire",
        "invalidate",
    ]
    assert events[0][2] != events[2][2]
    assert {event[5] for event in events} == {7}
    assert manager._observed_kv_leases == {}


def test_disabled_request_kv_lease_observation_is_noop(monkeypatch):
    events: list[tuple] = []
    monkeypatch.setattr(
        manager_module,
        "observe_resource_transition",
        lambda *args: events.append(args),
    )
    manager = KVCacheManager.__new__(KVCacheManager)
    manager._observed_kv_leases = None
    manager._observed_kv_lease_sequence = None

    manager._observe_kv_lease_acquired("request-0")
    manager._observe_kv_lease_released("request-0", 1)
    manager.invalidate_observed_kv_leases()

    assert events == []
