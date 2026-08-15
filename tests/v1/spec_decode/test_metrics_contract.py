# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.speculative_capability import SpeculativeCapability
from vllm.v1.spec_decode.metrics import (
    SpecDecodingLogging,
    SpecDecodingProm,
    SpecDecodingStats,
)


class _FakeMetric:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.label_values = ()
        self.increments: list[float] = []
        self.sets: list[float] = []
        self.observations: list[float] = []

    def labels(self, *values):
        self.label_values = values
        return self

    def inc(self, value=1.0):
        self.increments.append(value)

    def set(self, value):
        self.sets.append(value)

    def observe(self, value):
        self.observations.append(value)


class _TestSpecDecodingProm(SpecDecodingProm):
    _counter_cls = _FakeMetric
    _gauge_cls = _FakeMetric
    _histogram_cls = _FakeMetric


def _capability(status: str = "enabled") -> SpeculativeCapability:
    return SpeculativeCapability(
        requested_method="mtp" if status == "enabled" else "none",
        detected_checkpoint_method="mtp",
        resolved_method="mtp" if status == "enabled" else "none",
        proposer="builtin:mtp" if status == "enabled" else "none",
        platform="cuda",
        status=status,  # type: ignore[arg-type]
    )


def test_disabled_capability_gauge_is_exposed_without_runtime_metrics() -> None:
    metrics = _TestSpecDecodingProm(
        speculative_config=None,
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
        capability=_capability("disabled"),
    )

    gauge = metrics.gauge_spec_decode_capability[0]
    assert gauge.sets == [1]
    assert gauge.label_values[-6:] == (
        "none",
        "mtp",
        "none",
        "none",
        "cuda",
        "disabled",
    )
    assert metrics.spec_decoding_enabled is False


def test_operational_metrics_cover_acceptance_latency_and_effective_tokens() -> None:
    metrics = _TestSpecDecodingProm(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
        capability=_capability(),
    )
    stats = SpecDecodingStats.new(3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    stats.observe_step(
        num_forwards=1,
        num_committed_tokens=4,
        proposer_latency_seconds=0.012,
        verification_latency_seconds=0.034,
    )

    metrics.observe(stats)

    assert metrics.counter_spec_decode_num_draft_tokens[0].increments == [3]
    assert metrics.counter_spec_decode_num_accepted_tokens[0].increments == [2]
    assert metrics.counter_spec_decode_num_rejected_tokens[0].increments == [1]
    assert metrics.gauge_spec_decode_acceptance_rate[0].sets == [pytest.approx(2 / 3)]
    assert metrics.gauge_spec_decode_effective_tokens_per_forward[0].sets == [4]
    assert metrics.histogram_spec_decode_proposer_latency[0].observations == [0.012]
    assert metrics.histogram_spec_decode_verification_latency[0].observations == [0.034]


def test_operational_contract_is_present_in_interval_log() -> None:
    stats = SpecDecodingStats.new(3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    stats.observe_step(
        num_forwards=1,
        num_committed_tokens=4,
        proposer_latency_seconds=0.012,
        verification_latency_seconds=0.034,
    )
    logging = SpecDecodingLogging()
    logging.observe(stats)
    messages: list[str] = []

    logging.log(lambda message, *args: messages.append(message % args))

    assert len(messages) == 1
    assert "Accepted: 2 tokens" in messages[0]
    assert "Drafted: 3 tokens" in messages[0]
    assert "Rejected: 1 tokens" in messages[0]
    assert "Effective tokens/forward: 4.00" in messages[0]
    assert "Proposer latency: 0.012000 s" in messages[0]
    assert "Verification latency: 0.034000 s" in messages[0]
