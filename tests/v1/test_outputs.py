# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from unittest import TestCase

from vllm.v1.core.sched.ownership import OwnerLeaseKey, OwnerReceipt, OwnerReceiptBatch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, LogprobsLists, ModelRunnerOutput


class TestLogprobsLists(TestCase):
    def setUp(self):
        self.logprobsLists = LogprobsLists(
            logprob_token_ids=[
                [1, 2],  # Request 0 token 0
                [3, 4],  # Request 0 token 1
                [5, 6],  # Request 1 token 0
                [7, 8],  # Request 1 token 1
                [9, 10],  # Request 1 token 2
                [11, 12],  # Request 2 token 0
                [13, 14],  # Request 2 token 1
                [15, 16],  # Request 2 token 2
                [17, 18],  # Request 2 token 3
            ],
            logprobs=[
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
                [1.1, 1.2],
                [1.3, 1.4],
                [1.5, 1.6],
                [1.7, 1.8],
            ],
            sampled_token_ranks=[1, 3, 5, 7, 9, 11, 13, 15, 17],
            cu_num_generated_tokens=[0, 2, 5, 9],
        )

    def test_slice_without_cu_num_generated_tokens(self):
        """Test slicing without cu_num_generated_tokens"""
        logprobsLists = LogprobsLists(
            logprob_token_ids=[[1], [2], [3]],
            logprobs=[[0.1], [0.2], [0.3]],
            sampled_token_ranks=[1, 2, 3],
            cu_num_generated_tokens=None,
        )

        sliced = logprobsLists.slice_request(1, num_positions=2)
        assert sliced.logprob_token_ids == [[2], [3]]
        assert sliced.logprobs == [[0.2], [0.3]]
        assert sliced.sampled_token_ranks == [2, 3]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_from_start(self):
        """Test slicing from the start position"""
        sliced = self.logprobsLists.slice_request(0, num_positions=5)
        assert len(sliced.logprob_token_ids) == 5
        assert sliced.logprob_token_ids == [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_from_middle(self):
        """Test slicing from the middle position"""
        sliced = self.logprobsLists.slice_request(1, num_positions=7)
        assert len(sliced.logprob_token_ids) == 7
        assert sliced.logprob_token_ids == [
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
        ]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_single_request(self):
        """Test slicing a single request"""
        sliced = self.logprobsLists.slice_request(1, num_positions=3)
        assert len(sliced.logprob_token_ids) == 3
        assert sliced.logprob_token_ids == [[5, 6], [7, 8], [9, 10]]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_last_request(self):
        """Test slicing the last request"""
        sliced = self.logprobsLists.slice_request(2, num_positions=4)
        assert len(sliced.logprob_token_ids) == 4
        assert sliced.logprob_token_ids == [[11, 12], [13, 14], [15, 16], [17, 18]]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_all_requests(self):
        """Test slicing all requests (full slice)"""
        sliced = self.logprobsLists.slice_request(0, num_positions=9)
        assert len(sliced.logprob_token_ids) == 9  # All tokens
        assert sliced.logprob_token_ids == self.logprobsLists.logprob_token_ids
        assert sliced.cu_num_generated_tokens is None


class TestModelRunnerOutputOwnerReceiptBatches(TestCase):
    def test_default_is_none(self):
        """Backward compatible default: owner_receipt_batches is None unless
        a worker opts in, and the shared EMPTY singleton stays safe."""
        output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
        assert output.owner_receipt_batches is None
        assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None

    def test_pickle_roundtrip(self):
        """Owner receipt batches survive the pickle wire format used by the
        multiproc message queues."""
        receipt = OwnerReceipt(
            key=OwnerLeaseKey(request_id="req-0", owner_epoch=1),
            owner_id=1,
            command_seq=2,
            accepted=True,
            runnable_num_tokens=10,
        )
        batch = OwnerReceiptBatch(
            owner_rank=1,
            emitted_step_seq=7,
            events=(receipt,),
            free_capacity=42,
        )
        output = ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            owner_receipt_batches=[batch],
        )
        restored = pickle.loads(pickle.dumps(output))
        assert restored.owner_receipt_batches == [batch]
        assert restored.owner_receipt_batches[0].events == (receipt,)
