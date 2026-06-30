# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.sparsity.logits_processors import (
    FORCE_FIRST_TOKEN_EXTRA_ARG,
    ForceFirstTokenLogitsProcessor,
)


def test_force_first_token_processor_masks_only_first_step():
    processor = ForceFirstTokenLogitsProcessor(
        vllm_config=None,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )
    params = SamplingParams(
        extra_args={FORCE_FIRST_TOKEN_EXTRA_ARG: 2},
    )

    request_processor = processor.new_req_logits_processor(params)
    assert request_processor is not None

    logits = torch.arange(5, dtype=torch.float32)
    forced = request_processor([], logits)
    assert torch.isneginf(forced[[0, 1, 3, 4]]).all()
    assert forced[2].item() == 0

    unmodified = request_processor([2], logits)
    torch.testing.assert_close(unmodified, logits)


def test_force_first_token_processor_ignores_requests_without_extra_arg():
    processor = ForceFirstTokenLogitsProcessor(
        vllm_config=None,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )
    params = SamplingParams()

    assert processor.new_req_logits_processor(params) is None


def test_force_first_token_processor_validates_extra_arg_type():
    params = SamplingParams(
        extra_args={FORCE_FIRST_TOKEN_EXTRA_ARG: "2"},
    )

    with pytest.raises(ValueError, match="force_first_token_id must be an int"):
        ForceFirstTokenLogitsProcessor.validate_params(params)
