# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused tests for the scheduler's request-owner execution fence."""

import pytest

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_schedule_stamps_every_call_with_current_step() -> None:
    scheduler = create_scheduler()

    # A zero-token schedule call is still an execution/control-plane step.
    empty = scheduler.schedule()
    assert empty.total_num_scheduled_tokens == 0
    assert empty.step_seq == 1
    assert empty.step_seq == scheduler.current_step

    (request,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    assert scheduled.total_num_scheduled_tokens == 8
    assert scheduled.step_seq == 2
    assert scheduled.step_seq == scheduler.current_step

    # The owner fence is not the deferred-free KV fence.
    assert scheduler.sched_step_seq == 0
