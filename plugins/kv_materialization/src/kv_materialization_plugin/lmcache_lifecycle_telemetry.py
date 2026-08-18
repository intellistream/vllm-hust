"""
本模块的作用：记录 LMCache 生产路径中 KV 写回与物化的请求和传输生命周期。
输入：请求元数据、传输方向、token 数以及被测原生调用。
输出：不引入设备同步的 JSONL 审计事件。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TransferTicket:
    """描述一次传输开始时的并发状态。"""

    direction: str
    tokens: int
    started_ns: int
    opposite_active: int


class LifecycleTelemetry:
    """以主机单调时钟记录传输，不触碰设备 tensor。"""

    def __init__(
        self,
        output_path: str | Path | None,
        bytes_per_token: int,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token 必须为正数")
        self.bytes_per_token = bytes_per_token
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._active = {"load": 0, "store": 0}
        self._sequence = 0
        self._events: list[dict[str, Any]] = []
        self._output = (
            Path(output_path).open("a", encoding="utf-8", buffering=1)
            if output_path
            else None
        )

    def record_step(self, requests: Iterable[Any]) -> None:
        """记录调度步内计划的 load/store token 数。"""
        request_events = []
        for request in requests:
            load_spec = getattr(request, "load_spec", None)
            save_spec = getattr(request, "save_spec", None)
            token_count = len(getattr(request, "token_ids", ()))
            load_tokens = 0
            if load_spec is not None:
                load_tokens = max(
                    0,
                    int(load_spec.lmcache_cached_tokens)
                    - int(load_spec.vllm_cached_tokens),
                )
            store_tokens = 0
            if save_spec is not None and bool(save_spec.can_save):
                store_tokens = max(
                    0,
                    token_count - int(save_spec.skip_leading_tokens),
                )
            if load_tokens or store_tokens:
                request_events.append(
                    {
                        "request_id": str(request.req_id),
                        "load_tokens": load_tokens,
                        "store_tokens": store_tokens,
                    }
                )
        self._emit("scheduler_step", requests=request_events)

    def begin_transfer(self, direction: str, tokens: int) -> TransferTicket:
        """登记传输开始，并快照反方向活跃批次数。"""
        if direction not in self._active:
            raise ValueError(f"未知传输方向：{direction}")
        opposite = "store" if direction == "load" else "load"
        with self._lock:
            opposite_active = self._active[opposite]
            self._active[direction] += 1
        return TransferTicket(
            direction=direction,
            tokens=max(0, int(tokens)),
            started_ns=self._clock_ns(),
            opposite_active=opposite_active,
        )

    def end_transfer(
        self,
        ticket: TransferTicket,
        *,
        status: str = "ok",
        fields: dict[str, Any] | None = None,
        duration_field: str = "service_ms",
    ) -> None:
        """登记传输结束并输出服务时间、逻辑字节和重叠状态。"""
        ended_ns = self._clock_ns()
        with self._lock:
            self._active[ticket.direction] -= 1
            if self._active[ticket.direction] < 0:
                raise RuntimeError("传输生命周期计数失配")
        payload = dict(fields or {})
        payload.update({
            "direction": ticket.direction,
            "tokens": ticket.tokens,
            "logical_bytes": ticket.tokens * self.bytes_per_token,
            "elapsed_ms": (ended_ns - ticket.started_ns) / 1_000_000,
            "opposite_active_at_submit": ticket.opposite_active,
            "status": status,
        })
        payload[duration_field] = payload["elapsed_ms"]
        self._emit("transfer", **payload)

    def measure(
        self,
        direction: str,
        tokens: int,
        call: Callable[[], T],
        *,
        fields: dict[str, Any] | None = None,
    ) -> T:
        """保持原调用异常语义不变，并在 finally 中闭合计时。"""
        ticket = self.begin_transfer(direction, tokens)
        status = "ok"
        try:
            return call()
        except BaseException:
            status = "error"
            raise
        finally:
            self.end_transfer(ticket, status=status, fields=fields)

    def measure_generator(
        self,
        direction: str,
        tokens: int,
        generator: Generator[Any, Any, Any],
        *,
        expected_advances: int,
        fields: dict[str, Any] | None = None,
    ) -> Generator[Any, Any, Any]:
        """包装 layerwise generator，并在调用方的最后一次推进后闭合计时。"""
        return _MeasuredGenerator(
            telemetry=self,
            direction=direction,
            tokens=tokens,
            generator=generator,
            expected_advances=expected_advances,
            fields=fields,
        )

    def close(self) -> None:
        """关闭审计文件。"""
        if self._output is not None:
            self._output.close()
            self._output = None

    def drain_events(self) -> list[dict[str, Any]]:
        """返回无文件模式下的事件，供单测使用。"""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def _emit(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._sequence += 1
            payload = {
                "event": event,
                "sequence": self._sequence,
                "monotonic_ns": self._clock_ns(),
                **fields,
            }
            if self._output is None:
                self._events.append(payload)
                return
            self._output.write(json.dumps(payload, sort_keys=True) + "\n")


class _MeasuredGenerator:
    """透明代理 LMCache layerwise generator 的推进操作。"""

    def __init__(
        self,
        *,
        telemetry: LifecycleTelemetry,
        direction: str,
        tokens: int,
        generator: Generator[Any, Any, Any],
        expected_advances: int,
        fields: dict[str, Any] | None,
    ) -> None:
        self._telemetry = telemetry
        self._direction = direction
        self._tokens = tokens
        self._generator = generator
        self._expected_advances = max(0, int(expected_advances))
        self._fields = dict(fields or {})
        self._ticket: TransferTicket | None = None
        self._advance_calls = 0
        self._host_active_ns = 0
        self._finished = False

    def __iter__(self) -> "_MeasuredGenerator":
        return self

    def __next__(self) -> Any:
        return self._resume(self._generator.__next__)

    def send(self, value: Any) -> Any:
        return self._resume(lambda: self._generator.send(value))

    def throw(self, *args: Any) -> Any:
        return self._resume(lambda: self._generator.throw(*args))

    def close(self) -> None:
        try:
            self._generator.close()
        finally:
            self._finish("closed", "close")

    def _resume(self, operation: Callable[[], Any]) -> Any:
        if self._ticket is None:
            self._ticket = self._telemetry.begin_transfer(
                self._direction,
                self._tokens,
            )
        active_started_ns = self._telemetry._clock_ns()
        try:
            result = operation()
        except StopIteration:
            self._host_active_ns += (
                self._telemetry._clock_ns() - active_started_ns
            )
            self._finish("ok", "stop_iteration")
            raise
        except BaseException:
            self._host_active_ns += (
                self._telemetry._clock_ns() - active_started_ns
            )
            self._finish("error", "exception")
            raise

        self._host_active_ns += self._telemetry._clock_ns() - active_started_ns
        self._advance_calls += 1
        if (
            self._expected_advances > 0
            and self._advance_calls >= self._expected_advances
        ):
            self._finish("ok", "expected_advances")
        return result

    def _finish(self, status: str, completed_by: str) -> None:
        if self._finished or self._ticket is None:
            return
        self._finished = True
        fields = {
            **self._fields,
            "timing_scope": "layerwise_generator_lifecycle",
            "host_active_ms": self._host_active_ns / 1_000_000,
            "advance_calls": self._advance_calls,
            "expected_advances": self._expected_advances,
            "completed_by": completed_by,
        }
        self._telemetry.end_transfer(
            self._ticket,
            status=status,
            fields=fields,
            duration_field="lifecycle_ms",
        )
