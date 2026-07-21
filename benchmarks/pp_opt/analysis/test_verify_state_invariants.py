import unittest

from verify_state_invariants import (
    _scheduler_steps,
    _worker_steps_by_rank,
    verify_trace_events,
)


def scheduler_step(
    table,
    *,
    request_id="r",
    finished=(),
    sched_step_seq=None,
    processed_step_seq=None,
    num_scheduled_tokens=1,
):
    step = {
        "event_seq": 1,
        "finished_req_ids": list(finished),
        "scheduler_output": {
            "finished_req_ids": list(finished),
            "total_num_scheduled_tokens": max(num_scheduled_tokens, 1),
            "num_scheduled_tokens": {request_id: num_scheduled_tokens},
        },
        "request_states": [{"request_id": request_id, "block_ids": [list(table)]}],
    }
    if sched_step_seq is not None:
        step["sched_step_seq"] = sched_step_seq
    if processed_step_seq is not None:
        step["processed_step_seq"] = processed_step_seq
    return step


def worker_step(
    table,
    position,
    slot,
    *,
    request_id="r",
    diagnostic_step=None,
):
    step = {
        "event_seq": 1,
        "request_ids": [request_id],
        "num_scheduled_tokens": {request_id: 1},
        "block_tables": {"0": [{"request_index": 0, "block_ids": list(table)}]},
        "positions": {"numel": 1, "values": [position]},
        "slot_mappings": {"0": {"numel": 1, "values": [slot]}},
    }
    if diagnostic_step is not None:
        step["diagnostic_step"] = diagnostic_step
    return step


class StateInvariantTest(unittest.TestCase):
    def test_matching_authoritative_table_and_slot_pass(self):
        result = verify_trace_events(
            [scheduler_step([0, 2])],
            [worker_step([0, 2], position=128, slot=256)],
        )
        self.assertTrue(result["state_invariants_passed"])

    def test_graph_padding_uses_zero_position_tail(self):
        worker = worker_step([0, 2], position=128, slot=256)
        worker["positions"] = {"numel": 8, "values": [128, 0, 0, 0, 0, 0, 0, 0]}
        result = verify_trace_events([scheduler_step([0, 2])], [worker])
        self.assertTrue(result["state_invariants_passed"])

        worker["positions"]["values"][-1] = 9
        result = verify_trace_events([scheduler_step([0, 2])], [worker])
        self.assertEqual(
            result["violation_counts"]["worker_positions_padding_mismatch"], 1
        )

    def test_filters_zero_token_scheduler_and_empty_worker_warmup(self):
        zero_scheduler = scheduler_step([], num_scheduled_tokens=0)
        zero_scheduler["scheduler_output"]["total_num_scheduled_tokens"] = 0
        active_scheduler = scheduler_step([2])
        zero_worker = worker_step([], position=0, slot=0)
        zero_worker["request_ids"] = []
        zero_worker["num_scheduled_tokens"] = {}
        zero_worker["diagnostic_step"] = 0
        active_worker = worker_step([2], position=0, slot=256)
        active_worker.update(
            {
                "event": "attention_input_before_forward",
                "pp_rank": 0,
                "tp_rank": 0,
            }
        )
        self.assertEqual(_scheduler_steps([zero_scheduler, active_scheduler]), [])
        active_scheduler["event"] = "scheduler_exit"
        self.assertEqual(
            _scheduler_steps([zero_scheduler, active_scheduler]), [active_scheduler]
        )
        workers = _worker_steps_by_rank([zero_worker, active_worker])
        self.assertEqual(workers, {(0, 0): [active_worker]})

    def test_stale_worker_table_fails(self):
        result = verify_trace_events(
            [scheduler_step([0, 2])],
            [worker_step([1, 2], position=128, slot=256)],
        )
        self.assertEqual(
            result["violation_counts"]["worker_scheduler_block_table_mismatch"],
            1,
        )

    def test_duplicate_worker_alias_fails(self):
        result = verify_trace_events(
            [scheduler_step([1, 2])],
            [worker_step([1, 2, 1], position=128, slot=256)],
        )
        self.assertEqual(
            result["violation_counts"]["worker_duplicate_nonzero_block"], 1
        )

    def test_slot_mapping_must_follow_authoritative_table(self):
        result = verify_trace_events(
            [scheduler_step([0, 2])],
            [worker_step([0, 2], position=128, slot=128)],
        )
        self.assertEqual(result["violation_counts"]["slot_mapping_mismatch"], 1)

    def test_conflicting_live_ownership_fails(self):
        scheduler = scheduler_step([3], request_id="a")
        scheduler["request_states"].append({"request_id": "b", "block_ids": [[3]]})
        worker = worker_step([3], position=0, slot=384, request_id="a")
        result = verify_trace_events([scheduler], [worker])
        self.assertEqual(
            result["violation_counts"]["scheduler_conflicting_live_ownership"],
            1,
        )

    def test_reuse_after_finish_records_transition(self):
        first = scheduler_step([4], request_id="a")
        second = scheduler_step([4], request_id="b", finished=("a",))
        first_worker = worker_step([4], position=0, slot=512, request_id="a")
        second_worker = worker_step([4], position=0, slot=512, request_id="b")
        result = verify_trace_events([first, second], [first_worker, second_worker])
        self.assertTrue(result["state_invariants_passed"])
        self.assertTrue(result["reuse_transitions"][0]["previous_owner_finished"])

    def test_finished_owner_allows_lagging_processed_fence(self):
        result = verify_trace_events(
            [
                scheduler_step(
                    [4], request_id="a", sched_step_seq=10, processed_step_seq=8
                ),
                scheduler_step(
                    [4],
                    request_id="b",
                    finished=("a",),
                    sched_step_seq=11,
                    processed_step_seq=8,
                ),
            ],
            [
                worker_step(
                    [4], position=0, slot=512, request_id="a", diagnostic_step=10
                ),
                worker_step(
                    [4], position=0, slot=512, request_id="b", diagnostic_step=11
                ),
            ],
        )
        self.assertTrue(result["state_invariants_passed"])

    def test_reuse_before_processed_fence_fails(self):
        result = verify_trace_events(
            [
                scheduler_step(
                    [4], request_id="a", sched_step_seq=1, processed_step_seq=0
                ),
                scheduler_step(
                    [4], request_id="b", sched_step_seq=2, processed_step_seq=0
                ),
            ],
            [
                worker_step(
                    [4], position=0, slot=512, request_id="a", diagnostic_step=1
                ),
                worker_step(
                    [4], position=0, slot=512, request_id="b", diagnostic_step=2
                ),
            ],
        )
        self.assertEqual(result["violation_counts"]["unsafe_block_reuse_transition"], 1)

    def test_zero_based_request_fence_allows_processed_reuse(self):
        first = scheduler_step(
            [4], request_id="a", sched_step_seq=10, processed_step_seq=8
        )
        first["request_states"][0]["last_sched_seq"] = 9
        second = scheduler_step(
            [4], request_id="b", sched_step_seq=11, processed_step_seq=9
        )
        second["request_states"][0]["last_sched_seq"] = 11
        result = verify_trace_events(
            [first, second],
            [
                worker_step(
                    [4], position=0, slot=512, request_id="a", diagnostic_step=10
                ),
                worker_step(
                    [4], position=0, slot=512, request_id="b", diagnostic_step=11
                ),
            ],
        )
        self.assertTrue(result["state_invariants_passed"])
        self.assertEqual(result["reuse_transitions"][0]["last_owner_sched_step_seq"], 9)

    def test_reuse_after_authoritative_release_before_finish_passes(self):
        released = scheduler_step(
            [],
            request_id="a",
            sched_step_seq=2,
            processed_step_seq=1,
            num_scheduled_tokens=0,
        )
        released_worker = worker_step([], position=0, slot=0, request_id="a")
        released_worker["num_scheduled_tokens"] = {"a": 0}
        released_worker["positions"] = {"numel": 0, "values": []}
        released_worker["slot_mappings"] = {"0": {"numel": 0, "values": []}}
        result = verify_trace_events(
            [
                scheduler_step(
                    [4], request_id="a", sched_step_seq=1, processed_step_seq=0
                ),
                released,
                scheduler_step(
                    [4], request_id="b", sched_step_seq=3, processed_step_seq=2
                ),
            ],
            [
                worker_step(
                    [4], position=0, slot=512, request_id="a", diagnostic_step=1
                ),
                released_worker,
                worker_step(
                    [4], position=0, slot=512, request_id="b", diagnostic_step=3
                ),
            ],
        )
        self.assertTrue(result["state_invariants_passed"])
        self.assertTrue(
            result["reuse_transitions"][0]["all_workers_removed_previous_owner"]
        )
        self.assertTrue(result["reuse_transitions"][0]["observed_absent_release"])

    def test_same_block_id_in_different_groups_is_not_a_conflict(self):
        scheduler = scheduler_step([4], request_id="a")
        scheduler["request_states"] = [
            {"request_id": "a", "block_ids": [[4], [5]]},
            {"request_id": "b", "block_ids": [[6], [4]]},
        ]
        worker = worker_step([4], position=0, slot=512, request_id="a")
        worker["block_tables"]["1"] = [{"request_index": 0, "block_ids": [5]}]
        worker["slot_mappings"]["1"] = {"numel": 1, "values": [640]}
        result = verify_trace_events([scheduler], [worker])
        self.assertTrue(result["state_invariants_passed"])

    def test_all_four_worker_ranks_must_match_exactly(self):
        scheduler = scheduler_step([0, 2])
        workers = {
            rank: [worker_step([0, 2], position=128, slot=256)]
            for rank in ((0, 0), (0, 1), (1, 0), (1, 1))
        }
        workers[(1, 1)][0]["block_tables"]["0"][0]["block_ids"] = [1, 2]
        result = verify_trace_events(
            [scheduler],
            workers,
            expected_worker_ranks={(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertEqual(result["worker_steps_by_rank"]["pp1_tp1"], 1)
        self.assertEqual(result["violation_counts"]["worker_rank_state_mismatch"], 1)
        self.assertEqual(
            result["violation_counts"]["worker_scheduler_block_table_mismatch"],
            1,
        )

    def test_missing_worker_rank_fails(self):
        result = verify_trace_events(
            [scheduler_step([0, 2])],
            {(0, 0): [worker_step([0, 2], position=128, slot=256)]},
            expected_worker_ranks={(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertEqual(result["violation_counts"]["missing_worker_step"], 1)

    def test_reuse_fails_until_every_worker_removed_previous_owner(self):
        schedulers = [
            scheduler_step([4], request_id="a", sched_step_seq=1),
            scheduler_step([4], request_id="b", sched_step_seq=2, processed_step_seq=1),
        ]
        workers = {}
        for rank in ((0, 0), (0, 1), (1, 0), (1, 1)):
            workers[rank] = [
                worker_step(
                    [4], position=0, slot=512, request_id="a", diagnostic_step=1
                ),
                worker_step(
                    [4], position=0, slot=512, request_id="b", diagnostic_step=2
                ),
            ]
        workers[(1, 1)][1] = worker_step(
            [4], position=0, slot=512, request_id="a", diagnostic_step=2
        )
        result = verify_trace_events(
            schedulers,
            workers,
            expected_worker_ranks={(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertEqual(result["violation_counts"]["unsafe_block_reuse_transition"], 1)
        self.assertFalse(
            result["reuse_transitions"][0]["all_workers_removed_previous_owner"]
        )


if __name__ == "__main__":
    unittest.main()
