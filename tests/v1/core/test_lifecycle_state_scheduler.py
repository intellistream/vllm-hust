# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.core.sched.lifecycle_state_scheduler import (
    LIFECYCLE_RECEIPT_KEY,
    LIFECYCLE_RECEIPT_SCHEMA,
    LifecycleStateScheduler,
    RequestLifecycleLedger,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def make_request(
    *, internal_id: str = "cmpl-req-1-0", external_id: str = "req-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=internal_id,
        client_index=0,
        status=RequestStatus.WAITING,
        sampling_params=SimpleNamespace(
            extra_args={"state_scheduler_lifecycle_request_id": external_id}
        ),
        is_finished=lambda: RequestStatus.is_finished(
            make_request_status[internal_id]
        ),
    )


make_request_status: dict[str, RequestStatus] = {}


def scheduler_fixture(monkeypatch: pytest.MonkeyPatch) -> LifecycleStateScheduler:
    scheduler = LifecycleStateScheduler.__new__(LifecycleStateScheduler)
    scheduler.requests = {}
    scheduler.lifecycle_ledger = RequestLifecycleLedger()

    def native_add(self: Scheduler, request: Any) -> None:
        self.requests[request.request_id] = request
        make_request_status[request.request_id] = request.status

    def native_free(
        self: Scheduler, request: Any, delay_free_blocks: bool = False
    ) -> dict[str, Any]:
        del delay_free_blocks
        self.requests.pop(request.request_id)
        make_request_status[request.request_id] = request.status
        return {"native": "preserved"}

    def native_finish(
        self: Scheduler,
        request_ids: Any,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        released = []
        for request_id in request_ids:
            request = self.requests.get(request_id)
            if request is None or RequestStatus.is_finished(request.status):
                continue
            request.status = finished_status
            make_request_status[request_id] = finished_status
            self._free_request(request)
            released.append((request_id, request.client_index))
        return released

    monkeypatch.setattr(Scheduler, "add_request", native_add)
    monkeypatch.setattr(Scheduler, "_free_request", native_free)
    monkeypatch.setattr(Scheduler, "finish_requests", native_finish)
    return scheduler


def test_natural_completion_releases_once_and_returns_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = scheduler_fixture(monkeypatch)
    request = make_request()
    make_request_status[request.request_id] = request.status
    scheduler.add_request(request)

    request.status = RequestStatus.FINISHED_STOPPED
    make_request_status[request.request_id] = request.status
    response = scheduler._free_request(request)

    assert response is not None
    assert response["native"] == "preserved"
    receipt = response[LIFECYCLE_RECEIPT_KEY]
    assert receipt["schema"] == LIFECYCLE_RECEIPT_SCHEMA
    assert receipt["event"] == "complete"
    assert receipt["applied"] is True
    assert receipt["active_count"] == 0
    assert receipt["capacity_conserved"] is True


def test_cancel_complete_race_has_one_applied_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = scheduler_fixture(monkeypatch)
    request = make_request()
    make_request_status[request.request_id] = request.status
    scheduler.add_request(request)

    released = scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_ABORTED
    )
    duplicate = scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_STOPPED
    )
    receipts = scheduler.drain_lifecycle_receipts()

    assert released == [(request.request_id, 0)]
    assert duplicate == []
    terminals = [row for row in receipts if row["event"] != "acquire"]
    assert [row["event"] for row in terminals] == ["cancel", "duplicate_terminal"]
    assert [row["applied"] for row in terminals] == [True, False]
    assert all(row["active_count"] == 0 for row in terminals)


def test_duplicate_ids_in_one_cancel_batch_release_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = scheduler_fixture(monkeypatch)
    request = make_request()
    make_request_status[request.request_id] = request.status
    scheduler.add_request(request)

    scheduler.finish_requests(
        [request.request_id, request.request_id],
        RequestStatus.FINISHED_ABORTED,
    )
    receipts = scheduler.drain_lifecycle_receipts()

    assert [row["event"] for row in receipts] == [
        "acquire",
        "cancel",
        "duplicate_terminal",
    ]


def test_unknown_cancel_is_observed_without_capacity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = scheduler_fixture(monkeypatch)

    assert scheduler.finish_requests(
        "cmpl-missing-0", RequestStatus.FINISHED_ABORTED
    ) == []
    receipt = scheduler.drain_lifecycle_receipts()[0]
    assert receipt["event"] == "unknown_terminal"
    assert receipt["applied"] is False
    assert receipt["active_count"] == 0


def test_identity_mismatch_fails_before_native_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = scheduler_fixture(monkeypatch)
    request = make_request(external_id="other")
    make_request_status[request.request_id] = request.status

    with pytest.raises(ValueError, match="identity"):
        scheduler.add_request(request)
    assert scheduler.requests == {}
    assert scheduler.drain_lifecycle_receipts() == []


def test_native_admission_failure_does_not_acquire_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = scheduler_fixture(monkeypatch)
    request = make_request()
    make_request_status[request.request_id] = request.status

    def fail_native_add(self: Scheduler, native_request: Any) -> None:
        raise RuntimeError("native admission failed")

    monkeypatch.setattr(Scheduler, "add_request", fail_native_add)
    with pytest.raises(RuntimeError, match="native admission failed"):
        scheduler.add_request(request)
    assert scheduler.lifecycle_ledger.active_count == 0
    assert scheduler.drain_lifecycle_receipts() == []
