"""External Connector plugin for dynamic CPU KV materialization."""

from kv_materialization_plugin.decision import (
    MaterializationDecision,
    MaterializationDecisionConfig,
    MaterializationObservation,
    choose_materialization,
)

__all__ = [
    "MaterializationDecision",
    "MaterializationDecisionConfig",
    "MaterializationObservation",
    "choose_materialization",
]
