# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from vllm.control_bridge.runtime_health import (
    RuntimeHealthState,
    observe_engine_client_health,
)
from vllm.v1.engine.exceptions import EngineDeadError


@pytest.mark.asyncio
async def test_authoritative_engine_health_success() -> None:
    client = AsyncMock()
    client.check_health.return_value = None
    now = datetime(2026, 8, 29, tzinfo=timezone.utc)

    observation = await observe_engine_client_health(client, observed_at=now)

    assert observation.state is RuntimeHealthState.HEALTHY
    assert observation.observed_at == now
    client.check_health.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_engine_dead_is_distinct_from_unavailable_check() -> None:
    now = datetime(2026, 8, 29, tzinfo=timezone.utc)
    dead = AsyncMock()
    dead.check_health.side_effect = EngineDeadError()
    unavailable = AsyncMock()
    unavailable.check_health.side_effect = RuntimeError("secret topology")

    dead_observation = await observe_engine_client_health(dead, observed_at=now)
    unavailable_observation = await observe_engine_client_health(
        unavailable, observed_at=now
    )

    assert dead_observation.state is RuntimeHealthState.UNHEALTHY
    assert unavailable_observation.state is RuntimeHealthState.UNAVAILABLE
