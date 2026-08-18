from kv_materialization_plugin.lmcache_formula_reuse_policy import (
    FormulaReuseConfig,
    predicted_pipeline_ms,
    select_formula_reuse_tokens,
)


def test_shorter_prefix_is_selected_when_copy_dominates() -> None:
    decision = select_formula_reuse_tokens(
        available_tokens=4096,
        local_tokens=0,
        prompt_tokens=4224,
        config=FormulaReuseConfig(min_predicted_improvement_ms=0.0),
    )
    assert decision.selected_tokens < decision.available_tokens
    assert decision.selected_tokens % 128 == 0
    assert decision.predicted_improvement_ms > 0
    assert decision.reason == "formula_shorter_prefix_predicted_best"


def test_full_prefix_is_kept_when_continuous_prefill_is_long() -> None:
    decision = select_formula_reuse_tokens(
        available_tokens=4096,
        local_tokens=0,
        prompt_tokens=12288,
        config=FormulaReuseConfig(),
    )
    assert decision.selected_tokens == 4096
    assert decision.reason == "full_reuse_predicted_best"


def test_default_guard_keeps_full_reuse_below_noise_floor() -> None:
    decision = select_formula_reuse_tokens(
        available_tokens=4096,
        local_tokens=0,
        prompt_tokens=4224,
        config=FormulaReuseConfig(),
    )
    assert decision.selected_tokens == 4096
    assert decision.reason == "improvement_below_guard"


def test_local_hit_is_never_discarded_and_alignment_is_respected() -> None:
    decision = select_formula_reuse_tokens(
        available_tokens=2048,
        local_tokens=512,
        prompt_tokens=2304,
        config=FormulaReuseConfig(block_size=128, lmcache_chunk_size=256),
    )
    assert decision.selected_tokens >= 512
    assert decision.selected_tokens % 256 == 0


def test_prediction_models_dependency_correct_two_stage_pipeline() -> None:
    config = FormulaReuseConfig(layer_count=2, copy_fixed_ms=0, copy_per_token_ms=1, prefill_fixed_ms=0, prefill_per_token_ms=1, prefill_token_context_ms=0)
    # h=4: copy=4, prefill=6, per-layer pipeline = 2 + max(2,3) + 3 = 8.
    assert predicted_pipeline_ms(4, 0, 10, config) == 8
