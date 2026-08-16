# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public worker-wrapper seal for owner-local prefix lookup and commit."""

import pytest

from tests.v1.worker.test_request_owned_boundary import (
    _FakeWorker,
    _kv_cache_config,
    _output,
    _real_vllm_config,
)
from vllm.v1.core.request_owned_prefix import OwnerPrefixDescriptor
from vllm.v1.core.sched.ownership import (
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


def _wrapper() -> WorkerWrapperBase:
    wrapper = WorkerWrapperBase(global_rank=0)
    wrapper.vllm_config = _real_vllm_config()
    wrapper.vllm_config.cache_config.enable_prefix_caching = True
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    wrapper.worker = _FakeWorker()
    wrapper.mm_receiver_cache = None
    wrapper.initialize_from_config([_kv_cache_config(num_blocks=64)])
    return wrapper


def _reserve(request_id: str, command_seq: int) -> OwnerCommand:
    key = OwnerLeaseKey(request_id, 0)
    return OwnerCommand(
        key=key,
        owner_id=0,
        command_seq=command_seq,
        kind=OwnerCommandKind.RESERVE,
        required_num_tokens=12,
        allocation=OwnerAllocationDescriptor(
            key=key,
            num_prompt_tokens=8,
            num_computed_tokens=0,
            num_tokens=12,
            status=OwnerAdmissionStatus.WAITING,
            prefix=OwnerPrefixDescriptor((b"prefix-0", b"prefix-1")),
        ),
    )


def test_wrapper_reports_exact_hit_and_starts_first_forward_at_hit_boundary():
    wrapper = _wrapper()
    first = _reserve("first", 1)
    reserve_step = _output(step_seq=1)
    reserve_step.owner_commands = [first]
    receipt = wrapper.execute_model(reserve_step).owner_receipt_batches[0].events[0]
    assert receipt.accepted
    assert receipt.prefix_cache_hit_tokens == 0

    compute_step = _output(step_seq=2)
    compute_step.num_scheduled_tokens = {"first": 8}
    compute_step.total_num_scheduled_tokens = 8
    compute_step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=first.key,
            owner_id=0,
            step_seq=2,
            command_seq=1,
            runnable_num_tokens=12,
        )
    ]
    wrapper.execute_model(compute_step)

    release_step = _output(step_seq=3)
    release_step.owner_commands = [
        OwnerCommand(
            key=first.key,
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.RELEASE,
            required_num_tokens=12,
        )
    ]
    assert wrapper.execute_model(release_step).owner_receipt_batches[0].events[
        0
    ].accepted

    hit = _reserve("hit", 3)
    hit_step = _output(step_seq=4)
    hit_step.owner_commands = [hit]
    hit_receipt = wrapper.execute_model(hit_step).owner_receipt_batches[0].events[0]
    assert hit_receipt.accepted
    assert hit_receipt.prefix_cache_hit_tokens == 4

    suffix_step = _output(step_seq=5)
    suffix_step.num_scheduled_tokens = {"hit": 4}
    suffix_step.total_num_scheduled_tokens = 4
    suffix_step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=hit.key,
            owner_id=0,
            step_seq=5,
            command_seq=3,
            runnable_num_tokens=12,
        )
    ]
    wrapper.execute_model(suffix_step)
    metadata = wrapper.worker.metadata_handoffs[-1]
    (entry,) = metadata.entries
    assert entry.pre_step_num_computed_tokens == 4
    assert entry.post_step_num_tokens == 8
