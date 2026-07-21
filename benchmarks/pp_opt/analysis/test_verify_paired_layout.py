import copy
import unittest

from verify_paired_layout import _scheduler_coverage, compare_manifests


class ComparePairedLayoutTest(unittest.TestCase):
    def manifest(
        self,
        layout: str = "layout-a",
        token_hash: str = "tokens-a",
        *,
        concurrent: bool = False,
    ):
        coverage = {
            "request_count": 1,
            "candidate_microbatch_count": 1,
            "block_reuse_after_finish_count": 0,
        }
        if concurrent:
            coverage = {
                "request_count": 3,
                "candidate_microbatch_count": 2,
                "block_reuse_after_finish_count": 1,
            }
        return {
            "layout_sha256": layout,
            "coverage": coverage,
            "outputs": [
                {
                    "request_index": 0,
                    "output_length": 3,
                    "actual_output_length": 3,
                    "output_token_ids_sha256": token_hash,
                }
            ],
        }

    def test_matching_layout_and_tokens_pass_exactly(self):
        result = compare_manifests(self.manifest(), self.manifest())
        self.assertEqual(result["verdict"], "passed_exact_narrow")

    def test_cross_layout_is_incomparable_not_failed(self):
        result = compare_manifests(self.manifest("layout-a"), self.manifest("layout-b"))
        self.assertEqual(result["verdict"], "incomparable_layout")
        self.assertEqual(result["output_checks"], [])

    def test_matching_layout_with_token_mismatch_fails(self):
        result = compare_manifests(
            self.manifest(token_hash="tokens-a"),
            self.manifest(token_hash="tokens-b"),
        )
        self.assertEqual(result["verdict"], "failed_exact")

    def test_incomplete_output_fails_matching_layout(self):
        candidate = copy.deepcopy(self.manifest())
        candidate["outputs"][0]["actual_output_length"] = 2
        result = compare_manifests(self.manifest(), candidate)
        self.assertEqual(result["verdict"], "failed_exact")

    def test_single_request_cannot_pass_concurrent_gate(self):
        result = compare_manifests(
            self.manifest(), self.manifest(), scope="concurrent_recycle"
        )
        self.assertEqual(result["verdict"], "insufficient_concurrent_coverage")
        self.assertFalse(result["concurrent_coverage"]["passed"])

    def test_concurrent_recycle_coverage_passes_exact_gate(self):
        result = compare_manifests(
            self.manifest(concurrent=True),
            self.manifest(concurrent=True),
            scope="concurrent_recycle",
        )
        self.assertEqual(result["verdict"], "passed_exact_concurrent_recycle")

    def test_missing_candidate_microbatch_diversity_blocks_gate(self):
        baseline = self.manifest(concurrent=True)
        candidate = self.manifest(concurrent=True)
        candidate["coverage"]["candidate_microbatch_count"] = 1
        result = compare_manifests(baseline, candidate, scope="concurrent_recycle")
        self.assertEqual(result["verdict"], "insufficient_concurrent_coverage")
        self.assertFalse(
            result["concurrent_coverage"]["checks"][
                "candidate_has_distinct_microbatches"
            ]
        )

    def test_missing_reuse_after_finish_blocks_gate(self):
        baseline = self.manifest(concurrent=True)
        candidate = self.manifest(concurrent=True)
        baseline["coverage"]["block_reuse_after_finish_count"] = 0
        result = compare_manifests(baseline, candidate, scope="concurrent_recycle")
        self.assertEqual(result["verdict"], "insufficient_concurrent_coverage")
        self.assertFalse(
            result["concurrent_coverage"]["checks"][
                "baseline_reuses_block_after_finish"
            ]
        )


class SchedulerCoverageTest(unittest.TestCase):
    def event(self, microbatch_id, states, finished=()):
        return {
            "microbatch_id": microbatch_id,
            "finished_req_ids": list(finished),
            "scheduler_output": {"finished_req_ids": list(finished)},
            "request_states": [
                {"request_id": request_id, "block_ids": [block_ids]}
                for request_id, block_ids in states
            ],
        }

    def test_detects_completion_then_block_ownership_transfer(self):
        coverage = _scheduler_coverage(
            [
                self.event(0, [("a", [1])]),
                self.event(1, [("a", [1]), ("b", [2])]),
                self.event(0, [("b", [2])], finished=("a",)),
                self.event(0, [("b", [2]), ("c", [1])]),
            ]
        )
        self.assertEqual(coverage["request_count"], 3)
        self.assertEqual(coverage["candidate_microbatch_count"], 2)
        self.assertEqual(coverage["completion_count"], 1)
        self.assertEqual(coverage["block_reuse_after_finish_count"], 1)
        self.assertEqual(
            coverage["block_reuse_after_finish_events"][0]["previous_owner"],
            "request_0",
        )
        self.assertEqual(
            coverage["block_reuse_after_finish_events"][0]["new_owner"],
            "request_2",
        )

    def test_does_not_count_transfer_before_previous_owner_finishes(self):
        coverage = _scheduler_coverage(
            [
                self.event(0, [("a", [1])]),
                self.event(1, [("b", [1])]),
            ]
        )
        self.assertEqual(coverage["block_reuse_after_finish_count"], 0)


if __name__ == "__main__":
    unittest.main()
