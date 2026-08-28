# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime-owned health observations for the control bridge boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from vllm.v1.engine.exceptions import EngineDeadError


class RuntimeHealthState(str, Enum):
    """Closed outcomes from the authoritative engine health operation."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RuntimeHealthObservation:
    """A bounded host observation safe to copy into the bridge process."""

    state: RuntimeHealthState
    observed_at: datetime
    source: str = "engine_client.check_health"

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("health observation timestamp must be timezone-aware")
        if self.source != "engine_client.check_health":
            raise ValueError("health observation source is not runtime-owned")


class EngineHealthClient(Protocol):
    """Narrow subset of EngineClient required by the bridge adapter."""

    async def check_health(self) -> None: ...


async def observe_engine_client_health(
    client: EngineHealthClient,
    *,
    observed_at: datetime,
) -> RuntimeHealthObservation:
    """Call the same runtime operation used by the canonical /health route."""
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    try:
        await client.check_health()
    except EngineDeadError:
        state = RuntimeHealthState.UNHEALTHY
    except Exception:
        # Exception details can contain internal topology or implementation data.
        state = RuntimeHealthState.UNAVAILABLE
    else:
        state = RuntimeHealthState.HEALTHY
    return RuntimeHealthObservation(state=state, observed_at=observed_at)


def health_observation_to_dict(
    observation: RuntimeHealthObservation,
) -> dict[str, Any]:
    return {
        "state": observation.state.value,
        "observed_at": observation.observed_at.isoformat(),
        "source": observation.source,
    }


def parse_health_observation(payload: Any) -> RuntimeHealthObservation:
    if not isinstance(payload, dict):
        raise ValueError("health observation must be an object")
    expected = {"state", "observed_at", "source"}
    if set(payload) != expected:
        raise ValueError("health observation fields do not match v1")
    try:
        state = RuntimeHealthState(payload["state"])
    except (TypeError, ValueError) as error:
        raise ValueError("health observation state is invalid") from error
    if not isinstance(payload["observed_at"], str):
        raise ValueError("health observation timestamp must be a string")
    try:
        observed_at = datetime.fromisoformat(payload["observed_at"])
    except ValueError as error:
        raise ValueError("health observation timestamp is invalid") from error
    if not isinstance(payload["source"], str):
        raise ValueError("health observation source must be a string")
    return RuntimeHealthObservation(
        state=state,
        observed_at=observed_at,
        source=payload["source"],
    )
