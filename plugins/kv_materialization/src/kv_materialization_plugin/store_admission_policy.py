"""
本模块的作用：执行 deadline-feasible、SLO-rescue/value-per-byte 写回准入。
输入：请求可见的复用概率/延迟预测、真实画像服务时间、KV 字节和策略阈值。
输出：是否允许 D2H 写回、slack、value density、SLO rescue 与拒绝原因。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoreAdmissionConfig:
    """生产写回准入的冻结阈值与真实画像参数。"""

    min_value_ms_per_gib: float = 800.0
    deadline_safety_margin_ms: float = 10.0
    reuse_ttft_slo_ms: float = 400.0
    slo_rescue_bonus_ms: float = 400.0
    store_fixed_ms: float = 50.0
    store_ms_per_gib: float = 950.0


@dataclass(frozen=True, slots=True)
class StoreAdmissionInput:
    """仅包含写回决策时可见的预测与对象属性。"""

    predicted_reuse_probability: float
    predicted_reuse_delay_ms: float
    recompute_ttft_ms: float
    load_ttft_ms: float
    kv_bytes: int


@dataclass(frozen=True, slots=True)
class StoreAdmissionDecision:
    """可审计的写回准入结果。"""

    admit: bool
    reason: str
    predicted_store_ms: float
    predicted_slack_ms: float
    expected_value_ms: float
    value_ms_per_gib: float
    predicted_slo_rescue: bool


def decide_store_admission(
    request: StoreAdmissionInput,
    config: StoreAdmissionConfig,
) -> StoreAdmissionDecision:
    """按发布 deadline、SLO rescue 和单位字节价值决定是否写回。"""
    if not 0.0 <= request.predicted_reuse_probability <= 1.0:
        raise ValueError("predicted_reuse_probability 必须位于 [0, 1]")
    if request.predicted_reuse_delay_ms < 0.0:
        raise ValueError("predicted_reuse_delay_ms 不能为负")
    if request.kv_bytes <= 0:
        raise ValueError("kv_bytes 必须为正")

    gib = request.kv_bytes / (1024 ** 3)
    store_ms = config.store_fixed_ms + config.store_ms_per_gib * gib
    slack_ms = (
        request.predicted_reuse_delay_ms
        - config.deadline_safety_margin_ms
        - store_ms
    )
    slo_rescue = (
        request.recompute_ttft_ms > config.reuse_ttft_slo_ms
        and request.load_ttft_ms <= config.reuse_ttft_slo_ms
    )
    expected_value = request.predicted_reuse_probability * (
        max(0.0, request.recompute_ttft_ms - request.load_ttft_ms)
        + (config.slo_rescue_bonus_ms if slo_rescue else 0.0)
    )
    density = expected_value / gib
    if expected_value <= 0.0:
        reason = "non_positive_value"
    elif density < config.min_value_ms_per_gib:
        reason = "low_value_density"
    elif slack_ms < 0.0:
        reason = "negative_deadline_slack"
    else:
        reason = "admitted"
    return StoreAdmissionDecision(
        admit=reason == "admitted",
        reason=reason,
        predicted_store_ms=store_ms,
        predicted_slack_ms=slack_ms,
        expected_value_ms=expected_value,
        value_ms_per_gib=density,
        predicted_slo_rescue=slo_rescue,
    )


__all__ = [
    "StoreAdmissionConfig",
    "StoreAdmissionDecision",
    "StoreAdmissionInput",
    "decide_store_admission",
]
