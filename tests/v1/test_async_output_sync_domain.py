# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


class FakeAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, domain: object) -> None:
        self._domain = domain

    @property
    def synchronization_domain(self) -> object:
        return self._domain

    def synchronize(self) -> None:
        pass

    def get_output(self) -> ModelRunnerOutput:
        raise NotImplementedError

    def get_output_without_sync(self) -> ModelRunnerOutput:
        raise NotImplementedError


def test_batch_synchronization_requires_stream_identity() -> None:
    first_domain = object()
    first = FakeAsyncOutput(first_domain)

    assert first.can_batch_synchronize_with(FakeAsyncOutput(first_domain))
    assert not first.can_batch_synchronize_with(FakeAsyncOutput(object()))
