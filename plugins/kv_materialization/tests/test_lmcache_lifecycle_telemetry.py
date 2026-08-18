"""
本模块的作用：验证 KV 写回—物化生产路径审计的计时、核算和并发语义。
输入：构造的请求元数据与可控主机时钟。
输出：对 JSONL 事件字段和异常保持行为的断言结果。
"""

from dataclasses import dataclass

import pytest

from kv_materialization_plugin.lmcache_lifecycle_telemetry import (
    LifecycleTelemetry,
)


class FakeClock:
    def __init__(self, *values: int) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


@dataclass
class Spec:
    lmcache_cached_tokens: int = 0
    vllm_cached_tokens: int = 0
    skip_leading_tokens: int = 0
    can_save: bool = False


@dataclass
class Request:
    req_id: str
    token_ids: list[int]
    load_spec: Spec | None = None
    save_spec: Spec | None = None


def test_transfer_records_service_bytes_and_overlap() -> None:
    telemetry = LifecycleTelemetry(
        None,
        bytes_per_token=4096,
        clock_ns=FakeClock(
            10,
            20,
            5_000_020,
            5_000_021,
            6_000_010,
            6_000_011,
        ),
    )
    store = telemetry.begin_transfer("store", 8)
    load = telemetry.begin_transfer("load", 4)
    telemetry.end_transfer(load)
    telemetry.end_transfer(store)

    transfers = [
        event for event in telemetry.drain_events() if event["event"] == "transfer"
    ]
    assert transfers[0]["logical_bytes"] == 4 * 4096
    assert transfers[0]["opposite_active_at_submit"] == 1
    assert transfers[0]["service_ms"] == pytest.approx(5.0)


def test_scheduler_step_records_planned_request_tokens() -> None:
    telemetry = LifecycleTelemetry(None, bytes_per_token=1024)
    telemetry.record_step(
        [
            Request(
                "r0",
                list(range(256)),
                load_spec=Spec(
                    lmcache_cached_tokens=192,
                    vllm_cached_tokens=64,
                ),
                save_spec=Spec(skip_leading_tokens=128, can_save=True),
            )
        ]
    )
    event = telemetry.drain_events()[0]
    assert event["requests"] == [
        {"request_id": "r0", "load_tokens": 128, "store_tokens": 128}
    ]


def test_measure_preserves_exception_and_marks_error() -> None:
    telemetry = LifecycleTelemetry(None, bytes_per_token=1)

    def fail() -> None:
        raise RuntimeError("copy failed")

    with pytest.raises(RuntimeError, match="copy failed"):
        telemetry.measure("load", 1, fail)
    transfer = [
        event for event in telemetry.drain_events() if event["event"] == "transfer"
    ][0]
    assert transfer["status"] == "error"


def test_invalid_direction_is_rejected() -> None:
    telemetry = LifecycleTelemetry(None, bytes_per_token=1)
    with pytest.raises(ValueError, match="未知传输方向"):
        telemetry.begin_transfer("sideways", 1)


def test_layerwise_generator_closes_at_expected_advance_count() -> None:
    telemetry = LifecycleTelemetry(None, bytes_per_token=8)

    def native():
        first = yield "prepared"
        yield f"copied:{first}"
        yield "synchronized"

    measured = telemetry.measure_generator(
        "load",
        16,
        native(),
        expected_advances=3,
        fields={"request_ids": ["r0"]},
    )
    assert next(measured) == "prepared"
    assert measured.send("layer0") == "copied:layer0"
    assert next(measured) == "synchronized"

    transfer = [
        event for event in telemetry.drain_events() if event["event"] == "transfer"
    ][0]
    assert transfer["timing_scope"] == "layerwise_generator_lifecycle"
    assert transfer["advance_calls"] == 3
    assert transfer["expected_advances"] == 3
    assert transfer["completed_by"] == "expected_advances"
    assert transfer["logical_bytes"] == 128
    assert "service_ms" not in transfer
    assert transfer["lifecycle_ms"] >= transfer["host_active_ms"]


def test_layerwise_generator_preserves_error() -> None:
    telemetry = LifecycleTelemetry(None, bytes_per_token=1)

    def native():
        yield None
        raise RuntimeError("layer copy failed")

    measured = telemetry.measure_generator(
        "store",
        1,
        native(),
        expected_advances=3,
    )
    next(measured)
    with pytest.raises(RuntimeError, match="layer copy failed"):
        next(measured)
    transfer = [
        event for event in telemetry.drain_events() if event["event"] == "transfer"
    ][0]
    assert transfer["status"] == "error"
    assert transfer["completed_by"] == "exception"
