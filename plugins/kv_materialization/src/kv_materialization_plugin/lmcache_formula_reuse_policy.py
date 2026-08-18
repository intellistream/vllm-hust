"""Formula-driven selection of a shorter, contiguous LMCache prefix hit.

This module has no vLLM or NPU dependency so its safety boundaries can be
unit-tested independently.  All lengths are token counts; selected reuse is
always a contiguous prefix [0, selected_tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class FormulaReuseConfig:
    """Calibrated two-stage model for one prompt pass.

    The defaults are the 24-layer Qwen1.5-1.8B Ascend 910B2 fit collected in
    this project on 2026-08-11.  Production experiments should record all six
    coefficients in their effective config rather than treating them as a
    universal hardware model.
    """

    block_size: int = 128
    lmcache_chunk_size: int = 128
    layer_count: int = 24
    copy_fixed_ms: float = 1.058769054878092
    copy_per_token_ms: float = 0.009721907080673584
    prefill_fixed_ms: float = 8.85512586947587
    prefill_per_token_ms: float = 0.007009970208182733
    prefill_token_context_ms: float = 1.2829906283732969e-06
    # A sub-20 ms predicted gain is below the current end-to-end noise floor
    # of the 4K-prefix Ascend validation. Keep native full reuse by default;
    # exploratory experiments may override this explicitly.
    min_predicted_improvement_ms: float = 20.0

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.lmcache_chunk_size <= 0:
            raise ValueError("block_size and lmcache_chunk_size must be positive")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        values = (
            self.copy_fixed_ms,
            self.copy_per_token_ms,
            self.prefill_fixed_ms,
            self.prefill_per_token_ms,
            self.prefill_token_context_ms,
            self.min_predicted_improvement_ms,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("formula coefficients must be finite and non-negative")

    @property
    def alignment(self) -> int:
        return math.lcm(self.block_size, self.lmcache_chunk_size)


@dataclass(frozen=True, slots=True)
class FormulaReuseDecision:
    """Auditable result of selecting a contiguous external prefix."""

    available_tokens: int
    local_tokens: int
    selected_tokens: int
    predicted_selected_ms: float
    predicted_full_reuse_ms: float
    candidate_count: int
    reason: str

    @property
    def predicted_improvement_ms(self) -> float:
        return self.predicted_full_reuse_ms - self.predicted_selected_ms


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _copy_total_ms(selected_tokens: int, local_tokens: int, config: FormulaReuseConfig) -> float:
    copied_tokens = selected_tokens - local_tokens
    if copied_tokens <= 0:
        return 0.0
    return config.copy_fixed_ms + config.copy_per_token_ms * copied_tokens


def _prefill_total_ms(selected_tokens: int, prompt_tokens: int, config: FormulaReuseConfig) -> float:
    computed_tokens = max(1, prompt_tokens - selected_tokens)
    return (
        config.prefill_fixed_ms
        + config.prefill_per_token_ms * computed_tokens
        + config.prefill_token_context_ms * computed_tokens * prompt_tokens
    )


def predicted_pipeline_ms(selected_tokens: int, local_tokens: int, prompt_tokens: int, config: FormulaReuseConfig) -> float:
    """Predict the dependency-correct, two-resource layer pipeline makespan."""
    if not 0 <= local_tokens <= selected_tokens <= prompt_tokens:
        raise ValueError("token lengths must satisfy 0 <= local <= selected <= prompt")
    copy_per_layer = _copy_total_ms(selected_tokens, local_tokens, config) / config.layer_count
    prefill_per_layer = _prefill_total_ms(selected_tokens, prompt_tokens, config) / config.layer_count
    return copy_per_layer + (config.layer_count - 1) * max(copy_per_layer, prefill_per_layer) + prefill_per_layer


def select_formula_reuse_tokens(
    available_tokens: int,
    local_tokens: int,
    prompt_tokens: int,
    config: FormulaReuseConfig,
) -> FormulaReuseDecision:
    """Select a block-aligned prefix without inventing a non-contiguous hit.

    `available_tokens` is LMCache's actual longest contiguous hit.  The output
    can only shorten it; it can never discard a vLLM-local hit.  If a shorter
    point does not beat full reuse by the configured guard, the native longest
    hit is retained exactly.
    """
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (available_tokens, local_tokens, prompt_tokens)):
        raise ValueError("token counts must be integers")
    if not 0 <= local_tokens <= available_tokens <= prompt_tokens:
        raise ValueError("token lengths must satisfy 0 <= local <= available <= prompt")

    full_ms = predicted_pipeline_ms(available_tokens, local_tokens, prompt_tokens, config)
    alignment = config.alignment
    first = _ceil_to_multiple(local_tokens, alignment)
    candidates = [available_tokens]
    candidates.extend(range(first, available_tokens, alignment))
    candidates = sorted({candidate for candidate in candidates if local_tokens <= candidate <= available_tokens})

    best_tokens = available_tokens
    best_ms = full_ms
    for candidate in candidates:
        candidate_ms = predicted_pipeline_ms(candidate, local_tokens, prompt_tokens, config)
        if candidate_ms < best_ms:
            best_tokens, best_ms = candidate, candidate_ms

    if best_tokens == available_tokens:
        reason = "full_reuse_predicted_best"
    elif full_ms - best_ms < config.min_predicted_improvement_ms:
        best_tokens, best_ms = available_tokens, full_ms
        reason = "improvement_below_guard"
    else:
        reason = "formula_shorter_prefix_predicted_best"

    return FormulaReuseDecision(
        available_tokens=available_tokens,
        local_tokens=local_tokens,
        selected_tokens=best_tokens,
        predicted_selected_ms=best_ms,
        predicted_full_reuse_ms=full_ms,
        candidate_count=len(candidates),
        reason=reason,
    )
