# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic component port of LightLLM's Past-Future admission policy."""

from __future__ import annotations

from bisect import bisect
from collections import deque
from dataclasses import asdict, dataclass
import random
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PastFutureRequestState:
    request_id: str
    computed_tokens: int
    completed_output_tokens: int
    max_output_tokens: int


@dataclass(frozen=True)
class PastFutureAdmissionDecision:
    request_id: str
    sampled_remaining_tokens: int
    estimated_peak_tokens: int
    available_tokens: int
    admitted: bool

    def to_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


class PastFuturePolicy:
    """Frozen-history, seeded port of LightLLM's ``FuturePastReqQueue``."""

    WINDOW_SIZE = 40
    MINIMUM_SAMPLES = 200
    MAXIMUM_LISTS = 5
    RESERVED_FRACTION = 0.05

    def __init__(self, *, seed: int, initial_output_tokens: int = 512) -> None:
        if initial_output_tokens <= 0:
            raise ValueError("initial_output_tokens must be positive")
        self.seed = seed
        self.initial_output_tokens = initial_output_tokens
        self._numpy_rng = np.random.RandomState(seed)
        self._python_rng = random.Random(seed)
        self.history_output_tokens: deque[int] = deque(
            [initial_output_tokens] * (self.WINDOW_SIZE // 2),
            maxlen=self.WINDOW_SIZE,
        )

    def record_completed_output(self, output_tokens: int) -> None:
        if output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")
        self.history_output_tokens.append(output_tokens)

    def _sample_running_remaining(self, request: PastFutureRequestState) -> int:
        history = sorted(self.history_output_tokens)
        completed = request.completed_output_tokens
        start = bisect(history, completed)
        sample_range = [completed, *history[start:]]
        if sample_range[-1] < request.max_output_tokens:
            sample_range.append(request.max_output_tokens)
        if len(sample_range) == 1:
            return 0
        position = self._numpy_rng.random_sample() * (len(sample_range) - 1)
        lower_index = int(position)
        lower, upper = sample_range[lower_index : lower_index + 2]
        sampled_total = round(lower + (upper - lower) * (position - lower_index))
        return max(sampled_total - completed, 0)

    @staticmethod
    def _peak_tokens(pairs: Iterable[tuple[int, int]]) -> int:
        ordered = sorted(pairs, key=lambda pair: -pair[1])
        if not ordered:
            return 0
        remaining = np.array([pair[1] for pair in ordered], dtype=np.int64)
        processed = np.array([pair[0] for pair in ordered], dtype=np.int64)
        batch_sizes = np.arange(1, len(ordered) + 1, dtype=np.int64)
        return int(np.max(remaining * batch_sizes + np.cumsum(processed)))

    def decide(
        self,
        *,
        running: Iterable[PastFutureRequestState],
        candidate: PastFutureRequestState,
        max_kv_tokens: int,
        retained_kv_tokens: int = 0,
    ) -> PastFutureAdmissionDecision:
        if max_kv_tokens <= 0 or retained_kv_tokens < 0:
            raise ValueError("invalid KV token budget")
        pairs = [
            (state.computed_tokens, self._sample_running_remaining(state))
            for state in running
        ]
        sampled_remaining = self._python_rng.choice(tuple(self.history_output_tokens))
        pairs.append((candidate.computed_tokens, sampled_remaining))
        peak = self._peak_tokens(pairs)
        available = int(max_kv_tokens * (1 - self.RESERVED_FRACTION)) - retained_kv_tokens
        return PastFutureAdmissionDecision(
            request_id=candidate.request_id,
            sampled_remaining_tokens=sampled_remaining,
            estimated_peak_tokens=peak,
            available_tokens=available,
            admitted=peak < available,
        )
