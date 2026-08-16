# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.speculative_capability import (
    SpeculativeCapabilityError,
    detect_checkpoint_speculative_capability,
    resolve_speculative_capability,
)

PORTABLE_PROPOSERS = {
    "mtp": "builtin:mtp",
    "draft_model": "builtin:draft_model",
    "ngram": "builtin:ngram",
}


def test_dspark_markers_take_precedence_over_mtp_counter() -> None:
    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [40, 41, 42],
    }

    method, evidence = detect_checkpoint_speculative_capability(config)

    assert method == "dspark"
    assert "dspark_block_size" in evidence
    assert "num_nextn_predict_layers" not in evidence


def test_dspark_checkpoint_rejects_mtp_request() -> None:
    capability = resolve_speculative_capability(
        requested_method="mtp",
        hf_config={
            "num_nextn_predict_layers": 1,
            "dspark_markov_rank": 256,
        },
        platform="cuda",
        registered_proposers={**PORTABLE_PROPOSERS, "dspark": "EagleProposer"},
    )

    assert capability.status == "unavailable"
    assert capability.requested_method == "mtp"
    assert capability.detected_checkpoint_method == "dspark"
    assert capability.missing_capability == "checkpoint:mtp"
    with pytest.raises(SpeculativeCapabilityError) as exc_info:
        raise SpeculativeCapabilityError(capability)
    assert exc_info.value.to_dict()["detected_checkpoint_method"] == "dspark"


def test_mtp_checkpoint_resolves_mtp_proposer() -> None:
    capability = resolve_speculative_capability(
        requested_method="mtp",
        hf_config={"num_nextn_predict_layers": 1},
        platform="cuda",
        registered_proposers=PORTABLE_PROPOSERS,
    )

    assert capability.status == "enabled"
    assert capability.resolved_method == "mtp"
    assert capability.proposer == "builtin:mtp"


@pytest.mark.parametrize("registered_proposers", [{}, {"ngram": "NgramProposer"}])
def test_missing_or_unsupported_platform_proposer_fails_closed(
    registered_proposers: dict[str, str],
) -> None:
    capability = resolve_speculative_capability(
        requested_method="mtp",
        hf_config={"num_nextn_predict_layers": 1},
        platform="unsupported",
        registered_proposers=registered_proposers,
    )

    assert capability.status == "unavailable"
    assert capability.resolved_method == "none"
    assert capability.missing_capability == "unsupported:mtp_proposer"
    assert "platform plugin" in (capability.remediation or "")


def test_no_spec_baseline_reports_detected_but_disabled() -> None:
    capability = resolve_speculative_capability(
        requested_method=None,
        hf_config={"dspark_noise_token_id": 128799},
        platform="ascend",
        registered_proposers=PORTABLE_PROPOSERS,
    )

    assert capability.status == "disabled"
    assert capability.requested_method == "none"
    assert capability.detected_checkpoint_method == "dspark"
    assert capability.resolved_method == "none"
    assert capability.proposer == "none"


def test_external_draft_model_does_not_require_target_embedded_modules() -> None:
    capability = resolve_speculative_capability(
        requested_method="draft_model",
        hf_config={},
        platform="cuda",
        registered_proposers=PORTABLE_PROPOSERS,
    )

    assert capability.status == "enabled"
    assert capability.detected_checkpoint_method == "none"
    assert capability.resolved_method == "draft_model"


def test_exact_runtime_method_proposer_precedes_family_fallback() -> None:
    capability = resolve_speculative_capability(
        requested_method="eagle3",
        hf_config={},
        platform="cuda",
        registered_proposers={
            "draft_model": "builtin:draft_model",
            "eagle3": "builtin:eagle3",
        },
    )

    assert capability.requested_method == "draft_model"
    assert capability.resolved_method == "draft_model"
    assert capability.proposer == "builtin:eagle3"
