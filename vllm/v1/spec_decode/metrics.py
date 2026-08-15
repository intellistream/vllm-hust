# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeCapability, SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_rejected_tokens: int = 0
    num_forwards: int = 0
    num_committed_tokens: int = 0
    proposer_latency_seconds: float = 0.0
    verification_latency_seconds: float = 0.0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        self.num_rejected_tokens += num_draft_tokens - num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        for i in range(num_draft_tokens):
            self.num_draft_tokens_per_pos[i] += 1

    def observe_step(
        self,
        *,
        num_forwards: int,
        num_committed_tokens: int,
        proposer_latency_seconds: float,
        verification_latency_seconds: float,
    ) -> None:
        """Record host-observed proposer and verification phase metrics."""
        self.num_forwards += num_forwards
        self.num_committed_tokens += num_committed_tokens
        self.proposer_latency_seconds += proposer_latency_seconds
        self.verification_latency_seconds += verification_latency_seconds


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self, is_diffusion: bool = False):
        # Diffusion (dLLM) models reuse the spec-decode data path with
        # overloaded semantics, so the raw spec-decode framing (drafts, bonus
        # token, per-position vector) is logged with diffusion-native terms.
        self.is_diffusion = is_diffusion
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.num_forwards: list[int] = []
        self.num_committed_tokens: list[int] = []
        self.proposer_latency_seconds: list[float] = []
        self.verification_latency_seconds: list[float] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )
        self.num_forwards.append(spec_decoding_stats.num_forwards)
        self.num_committed_tokens.append(spec_decoding_stats.num_committed_tokens)
        self.proposer_latency_seconds.append(
            spec_decoding_stats.proposer_latency_seconds
        )
        self.verification_latency_seconds.append(
            spec_decoding_stats.verification_latency_seconds
        )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        num_rejected_tokens = num_draft_tokens - num_accepted_tokens
        num_forwards = np.sum(self.num_forwards)
        num_committed_tokens = np.sum(self.num_committed_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        if self.is_diffusion:
            self._log_diffusion(
                log_fn,
                num_denoising_steps=num_drafts,
                num_canvas_tokens=num_draft_tokens,
                num_committed_tokens=num_accepted_tokens,
                committed_throughput=accepted_throughput,
            )
            self.reset()
            return

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = (
            1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else float("nan")
        )
        effective_tokens_per_forward = (
            num_committed_tokens / num_forwards if num_forwards > 0 else float("nan")
        )

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = (
            np.sum(pos_matrix, axis=0) / num_drafts
            if num_drafts > 0
            else np.zeros(pos_matrix.shape[-1] if pos_matrix.ndim > 1 else 0)
        )
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Rejected: %d tokens, "
            "Effective tokens/forward: %.2f, "
            "Proposer latency: %.6f s, "
            "Verification latency: %.6f s, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            num_rejected_tokens,
            effective_tokens_per_forward,
            np.sum(self.proposer_latency_seconds),
            np.sum(self.verification_latency_seconds),
            rates_str,
            draft_acceptance_rate,
        )
        self.reset()

    def _log_diffusion(
        self,
        log_fn,
        num_denoising_steps: int,
        num_canvas_tokens: int,
        num_committed_tokens: int,
        committed_throughput: float,
    ):
        # Each "draft" is one denoising step that re-evaluates the canvas block
        # and finalizes some of its positions.
        mean_committed_per_step = (
            num_committed_tokens / num_denoising_steps
            if num_denoising_steps > 0
            else float("nan")
        )
        mean_steps_per_canvas = (
            num_canvas_tokens / num_committed_tokens
            if num_committed_tokens > 0
            else float("nan")
        )

        log_fn(
            "DiffusionDecoding metrics: "
            "Committed token throughput: %.2f tokens/s, "
            "Mean denoising steps per canvas: %.2f, "
            "Mean tokens committed per denoising step: %.2f, "
            "Committed: %d tokens, "
            "Denoising steps: %d, "
            "Canvas positions evaluated: %d",
            committed_throughput,
            mean_steps_per_canvas,
            mean_committed_per_step,
            num_committed_tokens,
            num_denoising_steps,
            num_canvas_tokens,
        )


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter
    _gauge_cls = prometheus_client.Gauge
    _histogram_cls = prometheus_client.Histogram

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        is_diffusion: bool = False,
        capability: SpeculativeCapability | None = None,
    ):
        if capability is not None:
            capability_labelnames = labelnames + [
                "requested_method",
                "detected_checkpoint_method",
                "resolved_method",
                "proposer",
                "platform",
                "status",
            ]
            capability_values = {
                idx: values
                + [
                    capability.requested_method,
                    capability.detected_checkpoint_method,
                    capability.resolved_method,
                    capability.proposer,
                    capability.platform,
                    capability.status,
                ]
                for idx, values in per_engine_labelvalues.items()
            }
            self.gauge_spec_decode_capability = create_metric_per_engine(
                self._gauge_cls(
                    name="vllm:spec_decode_capability",
                    documentation="Resolved speculative decoding capability.",
                    labelnames=capability_labelnames,
                ),
                capability_values,
            )
            for gauge in self.gauge_spec_decode_capability.values():
                gauge.set(1)

        # Diffusion (dLLM) models reuse the spec-decode counters but expose them
        # under diffusion-native names; the per-position acceptance vector does
        # not apply, so it is omitted.
        self.is_diffusion = is_diffusion
        self.spec_decoding_enabled = speculative_config is not None or is_diffusion
        if not self.spec_decoding_enabled:
            return

        if is_diffusion:
            counter_specs = [
                ("vllm:diffusion_num_denoising_steps", "Number of denoising steps."),
                (
                    "vllm:diffusion_num_canvas_positions",
                    "Number of canvas positions evaluated.",
                ),
                (
                    "vllm:diffusion_num_committed_tokens",
                    "Number of committed (finalized) tokens.",
                ),
            ]
        else:
            counter_specs = [
                ("vllm:spec_decode_num_drafts", "Number of spec decoding drafts."),
                ("vllm:spec_decode_num_draft_tokens", "Number of draft tokens."),
                ("vllm:spec_decode_num_accepted_tokens", "Number of accepted tokens."),
                ("vllm:spec_decode_num_rejected_tokens", "Number of rejected tokens."),
            ]

        counters = [
            create_metric_per_engine(
                self._counter_cls(name=name, documentation=doc, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for name, doc in counter_specs
        ]
        # num_drafts/num_draft_tokens/num_accepted_tokens map onto denoising
        # steps/canvas positions/committed tokens in the diffusion path.
        self.counter_spec_decode_num_drafts = counters[0]
        self.counter_spec_decode_num_draft_tokens = counters[1]
        self.counter_spec_decode_num_accepted_tokens = counters[2]
        self.counter_spec_decode_num_rejected_tokens = (
            counters[3] if not is_diffusion else None
        )

        if not is_diffusion:
            self.gauge_spec_decode_acceptance_rate = create_metric_per_engine(
                self._gauge_cls(
                    name="vllm:spec_decode_acceptance_rate",
                    documentation="Accepted draft token fraction.",
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            self.gauge_spec_decode_effective_tokens_per_forward = (
                create_metric_per_engine(
                    self._gauge_cls(
                        name="vllm:spec_decode_effective_tokens_per_forward",
                        documentation=(
                            "Mean committed tokens per target-model forward."
                        ),
                        labelnames=labelnames,
                    ),
                    per_engine_labelvalues,
                )
            )
            latency_buckets = (
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1.0,
                5.0,
            )
            self.histogram_spec_decode_proposer_latency = create_metric_per_engine(
                self._histogram_cls(
                    name="vllm:spec_decode_proposer_latency_seconds",
                    documentation="Host-observed proposer phase latency.",
                    labelnames=labelnames,
                    buckets=latency_buckets,
                ),
                per_engine_labelvalues,
            )
            self.histogram_spec_decode_verification_latency = create_metric_per_engine(
                self._histogram_cls(
                    name="vllm:spec_decode_verification_latency_seconds",
                    documentation=("Host-observed target verification phase latency."),
                    labelnames=labelnames,
                    buckets=latency_buckets,
                ),
                per_engine_labelvalues,
            )

        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        if not is_diffusion:
            assert speculative_config is not None
            num_spec_tokens = speculative_config.num_speculative_tokens
            pos_labelnames = labelnames + ["position"]
            base_counter = self._counter_cls(
                name="vllm:spec_decode_num_accepted_tokens_per_pos",
                documentation="Accepted tokens per draft position.",
                labelnames=pos_labelnames,
            )
            self.counter_spec_decode_num_accepted_tokens_per_pos = {
                idx: [
                    base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)
                ]
                for idx, lv in per_engine_labelvalues.items()
            }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        if self.counter_spec_decode_num_rejected_tokens is not None:
            self.counter_spec_decode_num_rejected_tokens[engine_idx].inc(
                spec_decoding_stats.num_rejected_tokens
            )
            acceptance_rate = (
                spec_decoding_stats.num_accepted_tokens
                / spec_decoding_stats.num_draft_tokens
                if spec_decoding_stats.num_draft_tokens
                else 0.0
            )
            self.gauge_spec_decode_acceptance_rate[engine_idx].set(acceptance_rate)
            effective_tokens = (
                spec_decoding_stats.num_committed_tokens
                / spec_decoding_stats.num_forwards
                if spec_decoding_stats.num_forwards
                else 0.0
            )
            self.gauge_spec_decode_effective_tokens_per_forward[engine_idx].set(
                effective_tokens
            )
            if spec_decoding_stats.proposer_latency_seconds > 0:
                self.histogram_spec_decode_proposer_latency[engine_idx].observe(
                    spec_decoding_stats.proposer_latency_seconds
                )
            if spec_decoding_stats.verification_latency_seconds > 0:
                self.histogram_spec_decode_verification_latency[engine_idx].observe(
                    spec_decoding_stats.verification_latency_seconds
                )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos.get(engine_idx, [])
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
