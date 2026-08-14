# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.core.sched.owner_window_policy import (
    OwnerPrefillCandidate,
    OwnerWindowPolicy,
    OwnerWindowPolicyConfig,
    OwnerWindowPolicyPhase,
    OwnerWindowReadiness,
)
from vllm.v1.core.sched.ownership import OwnerLeaseKey


def _key(request_id: str) -> OwnerLeaseKey:
    return OwnerLeaseKey(request_id=request_id, owner_epoch=0)


def _controller(
    *,
    observation_steps: int = 2,
    prefill_budget: int = 1,
    max_wait: int = 8,
) -> OwnerWindowPolicy:
    return OwnerWindowPolicy(
        OwnerWindowPolicyConfig(
            world_size=2,
            decode_observation_steps=observation_steps,
            hot_low_watermark=1,
            hot_high_watermark=2,
            prefill_invocation_budget=prefill_budget,
            prefill_max_wait_steps=max_wait,
        )
    )


def _readiness(
    *,
    hot: tuple[int, int] = (2, 2),
    restoring: tuple[int, int] = (0, 0),
    restorable: tuple[int, int] = (0, 0),
    candidates: tuple[OwnerPrefillCandidate, ...] = (),
) -> OwnerWindowReadiness:
    return OwnerWindowReadiness(hot, restoring, restorable, candidates)


def test_q_is_an_observation_boundary_not_a_service_ratio() -> None:
    controller = _controller(observation_steps=1, max_wait=100)
    candidates = (
        OwnerPrefillCandidate(_key("p0"), owner_id=0, wait_steps=1),
        OwnerPrefillCandidate(_key("p1"), owner_id=1, wait_steps=1),
    )

    for _ in range(12):
        decision = controller.ack_step(
            OwnerWindowPolicyPhase.DECODE,
            _readiness(candidates=candidates),
            positive_tokens=True,
        )
        assert decision.phase is OwnerWindowPolicyPhase.DECODE
        assert decision.prefill_wave == ()


def test_low_watermark_freezes_wave_and_budget_returns_to_decode() -> None:
    controller = _controller(observation_steps=1, prefill_budget=2)
    initial = (
        OwnerPrefillCandidate(_key("old-0"), owner_id=0, wait_steps=2),
        OwnerPrefillCandidate(_key("old-1"), owner_id=1, wait_steps=2),
    )
    opened = controller.ack_step(
        OwnerWindowPolicyPhase.DECODE,
        _readiness(hot=(1, 1), candidates=initial),
        positive_tokens=True,
    )
    assert opened.phase is OwnerWindowPolicyPhase.PREFILL
    assert opened.prefill_wave == (_key("old-0"), _key("old-1"))

    with_new_arrival = initial + (
        OwnerPrefillCandidate(_key("new"), owner_id=0, wait_steps=0),
    )
    first = controller.ack_step(
        OwnerWindowPolicyPhase.PREFILL,
        _readiness(hot=(1, 1), candidates=with_new_arrival),
        positive_tokens=True,
    )
    assert first.prefill_wave == opened.prefill_wave
    assert _key("new") not in first.prefill_wave

    closed = controller.ack_step(
        OwnerWindowPolicyPhase.PREFILL,
        _readiness(hot=(1, 1), candidates=with_new_arrival),
        positive_tokens=True,
    )
    assert closed.phase is OwnerWindowPolicyPhase.DECODE
    assert closed.reason == "prefill-invocation-budget"


def test_host_restorable_candidate_precedes_new_prefill() -> None:
    controller = _controller(observation_steps=1)
    candidates = (
        OwnerPrefillCandidate(_key("p0"), owner_id=0, wait_steps=1),
        OwnerPrefillCandidate(_key("p1"), owner_id=1, wait_steps=1),
    )
    decision = controller.ack_step(
        OwnerWindowPolicyPhase.DECODE,
        _readiness(hot=(1, 1), restorable=(1, 0), candidates=candidates),
        positive_tokens=True,
    )
    assert decision.prefill_wave == (_key("p1"),)


def test_max_wait_guardrail_is_independent_of_watermarks() -> None:
    controller = _controller(observation_steps=1, max_wait=4)
    due = OwnerPrefillCandidate(_key("old"), owner_id=0, wait_steps=4)
    decision = controller.ack_step(
        OwnerWindowPolicyPhase.DECODE,
        _readiness(hot=(3, 3), candidates=(due,)),
        positive_tokens=True,
    )
    assert decision.phase is OwnerWindowPolicyPhase.PREFILL
    assert decision.prefill_wave == (_key("old"),)
    assert decision.reason == "prefill-max-wait"


def test_command_only_step_does_not_advance_phase_budget() -> None:
    controller = _controller(observation_steps=2)
    low = _readiness(
        hot=(1, 1),
        candidates=(OwnerPrefillCandidate(_key("p0"), 0, 1),),
    )
    controller.ack_step(OwnerWindowPolicyPhase.DECODE, low, positive_tokens=False)
    assert controller.decode_steps == 0
    controller.ack_step(OwnerWindowPolicyPhase.DECODE, low, positive_tokens=True)
    controller.ack_step(OwnerWindowPolicyPhase.DECODE, low, positive_tokens=False)
    assert controller.phase is OwnerWindowPolicyPhase.DECODE
    assert controller.decode_steps == 1


@dataclass
class _QueuedPrefill:
    key: OwnerLeaseKey
    owner_id: int
    arrival_step: int


def test_continuous_arrivals_give_bounded_decode_and_prefill_progress() -> None:
    controller = _controller(observation_steps=2, prefill_budget=1, max_wait=4)
    initial = [
        _QueuedPrefill(_key(f"initial-{index}"), index % 2, 0) for index in range(12)
    ]
    queued = list(initial)
    served: dict[OwnerLeaseKey, int] = {}
    decode_invocations = 0
    prefill_invocations = 0
    longest_prefill_run = 0
    current_prefill_run = 0
    queue_never_empty = True

    for step in range(60):
        if step and step % 2 == 0:
            queued.append(
                _QueuedPrefill(_key(f"arrival-{step}"), (step // 2) % 2, step)
            )
        candidates = tuple(
            OwnerPrefillCandidate(item.key, item.owner_id, step - item.arrival_step)
            for item in queued
        )
        readiness = _readiness(hot=(2, 2), candidates=candidates)
        phase = controller.phase
        if phase is OwnerWindowPolicyPhase.DECODE:
            decode_invocations += 1
            current_prefill_run = 0
        else:
            prefill_invocations += 1
            current_prefill_run += 1
            longest_prefill_run = max(longest_prefill_run, current_prefill_run)
            members = set(controller.prefill_wave)
            completed = [item for item in queued if item.key in members]
            for item in completed:
                served[item.key] = step
            queued = [item for item in queued if item.key not in members]
            readiness = _readiness(
                hot=(2, 2),
                candidates=tuple(
                    OwnerPrefillCandidate(
                        item.key, item.owner_id, step - item.arrival_step
                    )
                    for item in queued
                ),
            )
        controller.ack_step(phase, readiness, positive_tokens=True)
        queue_never_empty = queue_never_empty and bool(queued)

    assert decode_invocations > 0
    assert prefill_invocations > 0
    assert longest_prefill_run <= 1
    assert queue_never_empty
    assert all(item.key in served for item in initial)
    assert max(served[item.key] - item.arrival_step for item in initial) <= 24
