# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker import kv_connector_model_runner_mixin as mixin_module
from vllm.v1.worker.gpu import kv_connector as v2_connector_module


class RecordingConnector:
    def __init__(self) -> None:
        self.observations: list[tuple[frozenset[str], int]] = []

    def observe_kv_recovery_first_compute(
        self,
        scheduled_request_ids: frozenset[str],
        timestamp_ns: int,
    ) -> None:
        self.observations.append((scheduled_request_ids, timestamp_ns))


def test_mrv1_forward_observer_passes_exact_scheduler_roster(monkeypatch):
    connector = RecordingConnector()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-0": 2, "request-1": 1}
    )
    monkeypatch.setattr(mixin_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        mixin_module,
        "get_kv_transfer_group",
        lambda: connector,
    )
    monkeypatch.setattr(mixin_module.time, "monotonic_ns", lambda: 123)

    mixin_module.KVConnectorModelRunnerMixin.observe_kv_recovery_first_compute(
        scheduler_output
    )

    assert connector.observations == [
        (frozenset({"request-0", "request-1"}), 123)
    ]


def test_mrv1_forward_observer_is_noop_without_connector(monkeypatch):
    monkeypatch.setattr(mixin_module, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mixin_module, "get_kv_transfer_group", lambda: 1 / 0)

    mixin_module.KVConnectorModelRunnerMixin.observe_kv_recovery_first_compute(
        SimpleNamespace(num_scheduled_tokens={"request-0": 1})
    )


def test_mrv2_forward_observer_passes_exact_scheduler_roster(monkeypatch):
    connector = RecordingConnector()
    active = v2_connector_module.ActiveKVConnector.__new__(
        v2_connector_module.ActiveKVConnector
    )
    active._disabled = False
    active.kv_connector = connector
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request-0": 1})
    monkeypatch.setattr(v2_connector_module.time, "monotonic_ns", lambda: 456)

    active.observe_kv_recovery_first_compute(scheduler_output)

    assert connector.observations == [(frozenset({"request-0"}), 456)]


def test_mrv2_forward_observer_is_noop_when_disabled(monkeypatch):
    connector = RecordingConnector()
    active = v2_connector_module.ActiveKVConnector.__new__(
        v2_connector_module.ActiveKVConnector
    )
    active._disabled = True
    active.kv_connector = connector
    monkeypatch.setattr(v2_connector_module.time, "monotonic_ns", lambda: 456)

    active.observe_kv_recovery_first_compute(
        SimpleNamespace(num_scheduled_tokens={"request-0": 1})
    )

    assert connector.observations == []
