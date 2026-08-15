# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checkpoint-aware speculative decoding capability negotiation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

SpeculativeCapabilityMethod = Literal["none", "mtp", "draft_model", "ngram", "dspark"]
SpeculativeCapabilityStatus = Literal["disabled", "enabled", "unavailable"]

_DSPARK_MARKERS = (
    "dspark_block_size",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
)
_MTP_MARKERS = (
    "n_predict",
    "num_nextn_predict_layers",
    "num_mtp_modules",
)


@dataclass(frozen=True)
class SpeculativeCapability:
    """Machine-readable result of speculative capability negotiation."""

    requested_method: SpeculativeCapabilityMethod
    detected_checkpoint_method: SpeculativeCapabilityMethod
    resolved_method: SpeculativeCapabilityMethod
    proposer: str
    platform: str
    status: SpeculativeCapabilityStatus
    missing_capability: str | None = None
    remediation: str | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible capability payload."""
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        return payload


class SpeculativeCapabilityError(ValueError):
    """Raised when requested speculation cannot be enabled safely."""

    def __init__(self, capability: SpeculativeCapability):
        self.capability = capability
        super().__init__(
            "Speculative capability negotiation failed: "
            + json.dumps(capability.to_dict(), sort_keys=True)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the structured failure payload."""
        return self.capability.to_dict()


def normalize_speculative_method(method: str | None) -> SpeculativeCapabilityMethod:
    """Map runtime-specific method names to the public capability taxonomy."""
    if method is None:
        return "none"
    if method in {"mtp", "deepseek_mtp", "mimo_mtp"} or method.endswith("_mtp"):
        return "mtp"
    if method in {"ngram", "ngram_gpu"}:
        return "ngram"
    if method == "dspark":
        return "dspark"
    return "draft_model"


def detect_checkpoint_speculative_capability(
    hf_config: Any,
) -> tuple[SpeculativeCapabilityMethod, tuple[str, ...]]:
    """Detect embedded speculative modules without using checkpoint names.

    DSpark markers intentionally take precedence over generic MTP counters.
    Some DSpark checkpoints retain ``num_nextn_predict_layers`` for model
    compatibility; treating that field as MTP would select the wrong proposer.
    """
    config = _config_mapping(hf_config)
    dspark_evidence = tuple(
        key for key in _DSPARK_MARKERS if config.get(key) is not None
    )
    architectures = tuple(config.get("architectures") or ())
    if dspark_evidence or any("DSpark" in str(arch) for arch in architectures):
        architecture_evidence = tuple(
            f"architecture:{arch}" for arch in architectures if "DSpark" in str(arch)
        )
        return "dspark", dspark_evidence + architecture_evidence

    mtp_evidence = tuple(
        key
        for key in _MTP_MARKERS
        if isinstance(config.get(key), int) and config[key] > 0
    )
    model_type = str(config.get("model_type") or "")
    mtp_architectures = tuple(
        f"architecture:{arch}" for arch in architectures if "MTP" in str(arch)
    )
    if model_type.endswith("_mtp"):
        mtp_evidence += (f"model_type:{model_type}",)
    if mtp_evidence or mtp_architectures:
        return "mtp", mtp_evidence + mtp_architectures
    return "none", ()


def resolve_speculative_capability(
    *,
    requested_method: str | None,
    hf_config: Any,
    platform: str,
    registered_proposers: Mapping[str, str],
    enforce_checkpoint_match: bool = True,
) -> SpeculativeCapability:
    """Resolve a request against checkpoint and proposer capabilities."""
    requested = normalize_speculative_method(requested_method)
    detected, evidence = detect_checkpoint_speculative_capability(hf_config)
    if requested == "none":
        return SpeculativeCapability(
            requested_method="none",
            detected_checkpoint_method=detected,
            resolved_method="none",
            proposer="none",
            platform=platform,
            status="disabled",
            evidence=evidence,
        )

    # Prefer an exact runtime method registration (for example ``eagle3`` or
    # ``custom_class``) before falling back to the normalized public family.
    # This keeps the public taxonomy stable without hiding which proposer the
    # runtime actually selected.
    proposer = registered_proposers.get(requested_method or "")
    if proposer is None:
        proposer = registered_proposers.get(requested)
    if proposer is None:
        return SpeculativeCapability(
            requested_method=requested,
            detected_checkpoint_method=detected,
            resolved_method="none",
            proposer="none",
            platform=platform,
            status="unavailable",
            missing_capability=f"{platform}:{requested}_proposer",
            remediation=(
                f"Install or update the {platform} platform plugin so it registers "
                f"a {requested} proposer, or choose a supported speculative method."
            ),
            evidence=evidence,
        )

    checkpoint_bound = requested in {"mtp", "dspark"} and enforce_checkpoint_match
    if checkpoint_bound and detected != requested:
        return SpeculativeCapability(
            requested_method=requested,
            detected_checkpoint_method=detected,
            resolved_method="none",
            proposer=proposer,
            platform=platform,
            status="unavailable",
            missing_capability=f"checkpoint:{requested}",
            remediation=(
                f"Use method='{detected}' for this checkpoint"
                if detected != "none"
                else f"Use a checkpoint with embedded {requested} modules"
            ),
            evidence=evidence,
        )

    return SpeculativeCapability(
        requested_method=requested,
        detected_checkpoint_method=detected,
        resolved_method=requested,
        proposer=proposer,
        platform=platform,
        status="enabled",
        evidence=evidence,
    )


def _config_mapping(hf_config: Any) -> Mapping[str, Any]:
    if isinstance(hf_config, Mapping):
        return hf_config
    if hasattr(hf_config, "to_dict"):
        return hf_config.to_dict()
    return vars(hf_config)
