# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import Mock

import numpy as np

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput


def test_async_output_materializes_and_trims_sampled_token_ids():
    output = AsyncOutput.__new__(AsyncOutput)
    output.copy_event = Mock()
    output.sampled_token_ids = np.array([[11, -1, -1], [21, 22, 23]])
    output.num_sampled_tokens_np = np.array([1, 3])
    output.model_runner_output = ModelRunnerOutput(
        req_ids=["request-0", "request-1"],
        req_id_to_index={"request-0": 0, "request-1": 1},
    )
    output.num_nans = None
    output.logprobs_tensors = None
    output.prompt_logprobs_dict = {}

    result = output.get_output()

    output.copy_event.synchronize.assert_called_once_with()
    assert result.sampled_token_ids == [[11], [21, 22, 23]]
