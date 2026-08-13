"""
本文件的作用：验证生产写回准入的 deadline、价值密度和 SLO rescue 不变量。
输入：内嵌预测、KV 大小和冻结策略参数。
输出：pytest 回归测试结果。
"""

import pytest

from kv_materialization_plugin.store_admission_policy import (
    StoreAdmissionConfig,
    StoreAdmissionInput,
    decide_store_admission,
)


def test_slo_rescue_with_positive_slack_is_admitted() -> None:
    decision = decide_store_admission(
        StoreAdmissionInput(0.8, 2000.0, 520.0, 398.0, 469_762_048),
        StoreAdmissionConfig(),
    )
    assert decision.admit
    assert decision.predicted_slo_rescue
    assert decision.predicted_slack_ms > 0


def test_negative_deadline_slack_is_rejected() -> None:
    decision = decide_store_admission(
        StoreAdmissionInput(1.0, 100.0, 520.0, 398.0, 469_762_048),
        StoreAdmissionConfig(),
    )
    assert not decision.admit
    assert decision.reason == "negative_deadline_slack"


def test_non_positive_materialization_value_is_rejected() -> None:
    decision = decide_store_admission(
        StoreAdmissionInput(1.0, 2000.0, 100.0, 150.0, 117_440_512),
        StoreAdmissionConfig(),
    )
    assert not decision.admit
    assert decision.reason == "non_positive_value"


def test_invalid_probability_fails_closed() -> None:
    with pytest.raises(ValueError):
        decide_store_admission(
            StoreAdmissionInput(1.1, 2000.0, 520.0, 398.0, 469_762_048),
            StoreAdmissionConfig(),
        )
