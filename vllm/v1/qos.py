# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-level QoS metadata and monotonic deadline tracking."""

import math
import time
from dataclasses import dataclass

import msgspec

MAX_EXPECTED_OUTPUT_TOKENS = 2**31 - 1


class QoSParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    frozen=True,
):  # type: ignore[call-arg]
    """Optional service-level objectives attached to one request.

    The API uses milliseconds because that is the common unit for serving
    SLOs. Deadlines are converted to the engine-core monotonic clock as soon
    as a :class:`Request` is created.
    """

    ttft_slo_ms: float | None = None
    tbt_slo_ms: float | None = None
    ttlt_slo_ms: float | None = None
    expected_output_tokens: int | None = None
    service_class: str | None = None

    def __post_init__(self) -> None:
        slos = (self.ttft_slo_ms, self.tbt_slo_ms, self.ttlt_slo_ms)
        if all(value is None for value in slos):
            raise ValueError("At least one QoS SLO must be specified.")
        for name, value in zip(
            ("ttft_slo_ms", "tbt_slo_ms", "ttlt_slo_ms"), slos, strict=True
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number.")
        if self.expected_output_tokens is not None and not (
            0 < self.expected_output_tokens <= MAX_EXPECTED_OUTPUT_TOKENS
        ):
            raise ValueError(
                f"expected_output_tokens must be in [1, {MAX_EXPECTED_OUTPUT_TOKENS}]."
            )
        if self.service_class is not None and (
            not self.service_class or len(self.service_class) > 64
        ):
            raise ValueError("service_class must contain 1 to 64 characters.")


@dataclass
class QoSRuntimeState:
    """Mutable QoS state kept exclusively inside the engine-core process."""

    ttft_deadline: float | None
    tbt_slo_s: float | None
    ttlt_deadline: float | None
    expected_output_tokens: int
    service_class: str | None
    first_token_time: float | None = None
    last_token_time: float | None = None
    ttft_observed: bool = False
    ttlt_observed: bool = False

    @classmethod
    def from_params(
        cls,
        params: QoSParams,
        *,
        arrival_time: float,
        default_expected_output_tokens: int,
        wall_now: float | None = None,
        monotonic_now: float | None = None,
    ) -> "QoSRuntimeState":
        """Convert frontend wall-clock age into local monotonic deadlines."""

        wall_now = time.time() if wall_now is None else wall_now
        monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
        if not all(
            math.isfinite(value) for value in (arrival_time, wall_now, monotonic_now)
        ):
            raise ValueError("QoS timestamps must be finite.")
        frontend_age_s = max(0.0, wall_now - arrival_time)

        def deadline(slo_ms: float | None) -> float | None:
            if slo_ms is None:
                return None
            return monotonic_now + slo_ms / 1000.0 - frontend_age_s

        expected_output_tokens = (
            params.expected_output_tokens
            if params.expected_output_tokens is not None
            else default_expected_output_tokens
        )
        return cls(
            ttft_deadline=deadline(params.ttft_slo_ms),
            tbt_slo_s=(
                None if params.tbt_slo_ms is None else params.tbt_slo_ms / 1000.0
            ),
            ttlt_deadline=deadline(params.ttlt_slo_ms),
            expected_output_tokens=expected_output_tokens,
            service_class=params.service_class,
        )

    def next_token_deadline(self) -> float:
        """Return the earliest active deadline for the next generated token."""

        deadlines: list[float] = []
        if self.first_token_time is None:
            if self.ttft_deadline is not None:
                deadlines.append(self.ttft_deadline)
        elif self.tbt_slo_s is not None and self.last_token_time is not None:
            deadlines.append(self.last_token_time + self.tbt_slo_s)
        if self.ttlt_deadline is not None:
            deadlines.append(self.ttlt_deadline)
        return min(deadlines, default=math.inf)

    def has_active_deadline(self) -> bool:
        """Whether an SLO still constrains the request's current phase."""

        return math.isfinite(self.next_token_deadline())

    def waiting_deadline(self) -> float:
        """Return the phase-correct deadline while the request is queued.

        A decode request may return to the waiting queue after preemption. In
        that case its TBT deadline, rather than its already-observed TTFT
        deadline, must determine its queue order.
        """

        return self.next_token_deadline()

    def observe_tokens(self, num_new_tokens: int, now: float) -> tuple[int, int]:
        """Record generated tokens and return TTFT/TBT violation deltas."""

        if num_new_tokens <= 0:
            return 0, 0

        ttft_violations = 0
        tbt_violations = 0
        if not self.ttft_observed:
            if self.ttft_deadline is not None and now > self.ttft_deadline:
                ttft_violations = 1
            self.ttft_observed = True
            self.first_token_time = now
        elif (
            self.tbt_slo_s is not None
            and self.last_token_time is not None
            and now - self.last_token_time > self.tbt_slo_s
        ):
            tbt_violations = 1

        self.last_token_time = now
        return ttft_violations, tbt_violations

    def observe_finished(self, now: float) -> int:
        """Record completion and return the TTLT violation delta."""

        if self.ttlt_observed:
            return 0
        self.ttlt_observed = True
        return int(self.ttlt_deadline is not None and now > self.ttlt_deadline)
