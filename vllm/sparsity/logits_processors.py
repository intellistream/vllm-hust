# SPDX-License-Identifier: Apache-2.0
"""Logits processors used by activation-sparsity validation scripts."""

from __future__ import annotations

from typing import Any

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

FORCE_FIRST_TOKEN_EXTRA_ARG = "force_first_token_id"


class _ForceFirstTokenRequestProcessor:
    """Force only the first generated token for one request.

    The second generated token and later tokens are left unmodified. This lets
    a validation script teacher-force a one-token bridge and then read the next
    token logprob from a real decode step without corrupting that logprob.
    """

    def __init__(self, token_id: int):
        self.token_id = int(token_id)

    def __call__(
        self,
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if output_token_ids:
            return logits
        if self.token_id < 0 or self.token_id >= logits.shape[-1]:
            raise ValueError(
                f"{FORCE_FIRST_TOKEN_EXTRA_ARG}={self.token_id} is outside "
                f"the logits vocab size {logits.shape[-1]}."
            )
        logits = logits.clone()
        logits.fill_(float("-inf"))
        logits[self.token_id] = 0
        return logits


class ForceFirstTokenLogitsProcessor(AdapterLogitsProcessor):
    """Per-request bridge-token forcing for decode-path PPL scoring."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra_args: dict[str, Any] | None = sampling_params.extra_args
        if not extra_args or FORCE_FIRST_TOKEN_EXTRA_ARG not in extra_args:
            return
        token_id = extra_args[FORCE_FIRST_TOKEN_EXTRA_ARG]
        if not isinstance(token_id, int):
            raise ValueError(
                f"{FORCE_FIRST_TOKEN_EXTRA_ARG} must be an int, got "
                f"{type(token_id).__name__}."
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> _ForceFirstTokenRequestProcessor | None:
        extra_args = params.extra_args
        if not extra_args or FORCE_FIRST_TOKEN_EXTRA_ARG not in extra_args:
            return None
        return _ForceFirstTokenRequestProcessor(
            int(extra_args[FORCE_FIRST_TOKEN_EXTRA_ARG])
        )
