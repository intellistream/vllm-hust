#!/usr/bin/env python3
"""Verify scheduler/worker KV ownership invariants from PP state-flow evidence."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    events = []
    for raw_path in sorted(glob.glob(str(run_dir / "state_flow.pid-*.jsonl"))):
        with open(raw_path, encoding="utf-8") as source:
            events.extend(json.loads(line) for line in source)
    return events


def _scheduler_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            event
            for event in events
            if event.get("event") == "scheduler_exit"
            and (event.get("scheduler_output") or {}).get(
                "total_num_scheduled_tokens", 0
            )
        ),
        key=lambda event: int(event["event_seq"]),
    )


def _worker_steps_by_rank(
    events: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    by_rank: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event") != "attention_input_before_forward" or not event.get(
            "request_ids"
        ):
            continue
        rank = (int(event["pp_rank"]), int(event["tp_rank"]))
        by_rank.setdefault(rank, []).append(event)
    for steps in by_rank.values():
        steps.sort(key=lambda event: int(event["event_seq"]))
    return by_rank


def _group_zero(table: list[list[int]]) -> list[list[int]]:
    return [[int(block) for block in group] for group in table]


def verify_trace_events(
    scheduler_steps: list[dict[str, Any]],
    worker_steps: (list[dict[str, Any]] | dict[tuple[int, int], list[dict[str, Any]]]),
    *,
    block_size: int = 128,
    expected_worker_ranks: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    workers_by_rank = (
        {(0, 0): worker_steps} if isinstance(worker_steps, list) else worker_steps
    )
    workers_by_rank = dict(workers_by_rank)
    for rank in expected_worker_ranks or ():
        workers_by_rank.setdefault(rank, [])
    for rank, steps in workers_by_rank.items():
        if len(steps) != len(scheduler_steps):
            violations.append(
                {
                    "kind": "scheduler_worker_step_count_mismatch",
                    "rank": list(rank),
                    "scheduler_steps": len(scheduler_steps),
                    "worker_steps": len(steps),
                }
            )

    historical_owner: dict[tuple[int, int], tuple[str, int]] = {}
    released_ownership_keys: set[tuple[int, int]] = set()
    finished: set[str] = set()
    reuse_transitions: list[dict[str, Any]] = []
    checked_steps = 0
    for ordinal, scheduler in enumerate(scheduler_steps):
        sched_seq = int(scheduler.get("sched_step_seq", ordinal + 1))
        processed_step_seq = int(scheduler.get("processed_step_seq", sched_seq))
        current_workers = {
            rank: steps[ordinal] if ordinal < len(steps) else None
            for rank, steps in workers_by_rank.items()
        }
        missing_ranks = [
            rank for rank, worker in current_workers.items() if worker is None
        ]
        if missing_ranks:
            violations.append(
                {
                    "kind": "missing_worker_step",
                    "ordinal": ordinal,
                    "sched_step_seq": sched_seq,
                    "ranks": [list(rank) for rank in missing_ranks],
                }
            )
        else:
            checked_steps += 1

        newly_finished = set(scheduler.get("finished_req_ids", []))
        newly_finished.update(
            (scheduler.get("scheduler_output") or {}).get("finished_req_ids", [])
        )
        finished.update(newly_finished)
        states = {
            state["request_id"]: _group_zero(state.get("block_ids", []))
            for state in scheduler.get("request_states", [])
        }
        state_last_sched_seq = {
            state["request_id"]: int(state.get("last_sched_seq", sched_seq))
            for state in scheduler.get("request_states", [])
        }

        live_owner: dict[tuple[int, int], str] = {}
        for request_id, groups in states.items():
            for group_index, table in enumerate(groups):
                nonzero = [block for block in table if block != 0]
                duplicates = sorted(
                    block for block in set(nonzero) if nonzero.count(block) > 1
                )
                if duplicates:
                    violations.append(
                        {
                            "kind": "scheduler_duplicate_nonzero_block",
                            "ordinal": ordinal,
                            "request_id": request_id,
                            "group_index": group_index,
                            "block_ids": duplicates,
                        }
                    )
                for block in nonzero:
                    ownership_key = (group_index, block)
                    other = live_owner.get(ownership_key)
                    if other is not None and other != request_id:
                        violations.append(
                            {
                                "kind": "scheduler_conflicting_live_ownership",
                                "ordinal": ordinal,
                                "group_index": group_index,
                                "block_id": block,
                                "owners": sorted({other, request_id}),
                            }
                        )
                    live_owner[ownership_key] = request_id
                    previous = historical_owner.get(ownership_key)
                    if previous is not None and previous[0] != request_id:
                        previous_request_id, last_owner_sched_seq = previous
                        previous_owner_finished = previous_request_id in finished
                        observed_absent_release = (
                            ownership_key in released_ownership_keys
                        )
                        all_workers_removed_previous_owner = True
                        for worker in current_workers.values():
                            if worker is None:
                                all_workers_removed_previous_owner = False
                                continue
                            worker_ids = list(worker["request_ids"])
                            rows = worker.get("block_tables", {}).get(
                                str(group_index), []
                            )
                            for row in rows:
                                request_index = int(row["request_index"])
                                if (
                                    request_index < len(worker_ids)
                                    and worker_ids[request_index] == previous_request_id
                                    and block in [int(x) for x in row["block_ids"]]
                                ):
                                    all_workers_removed_previous_owner = False
                        transition = {
                            "ordinal": ordinal,
                            "sched_step_seq": sched_seq,
                            "processed_step_seq": processed_step_seq,
                            "group_index": group_index,
                            "block_id": block,
                            "previous_owner": previous_request_id,
                            "new_owner": request_id,
                            "previous_owner_finished": previous_owner_finished,
                            "observed_absent_release": observed_absent_release,
                            "last_owner_sched_step_seq": last_owner_sched_seq,
                            "all_workers_removed_previous_owner": (
                                all_workers_removed_previous_owner
                            ),
                        }
                        reuse_transitions.append(transition)
                        direct_transition_safe = (
                            last_owner_sched_seq <= processed_step_seq
                            and all_workers_removed_previous_owner
                        )
                        if not (
                            previous_owner_finished
                            or observed_absent_release
                            or direct_transition_safe
                        ):
                            violations.append(
                                {"kind": "unsafe_block_reuse_transition", **transition}
                            )
                        released_ownership_keys.discard(ownership_key)
                    historical_owner[ownership_key] = (
                        request_id,
                        state_last_sched_seq[request_id],
                    )

        released_ownership_keys.update(set(historical_owner) - set(live_owner))

        expected_scheduled = {
            request_id: int(token_count)
            for request_id, token_count in (
                (scheduler.get("scheduler_output") or {})
                .get("num_scheduled_tokens", {})
                .items()
            )
        }
        canonical_worker: dict[str, Any] | None = None
        for rank, worker in current_workers.items():
            if worker is None:
                continue
            worker_ids = list(worker["request_ids"])
            worker_scheduled = {
                request_id: int(token_count)
                for request_id, token_count in worker.get(
                    "num_scheduled_tokens", {}
                ).items()
            }
            rank_payload = {"pp_rank": rank[0], "tp_rank": rank[1]}
            if worker_scheduled != expected_scheduled:
                violations.append(
                    {
                        "kind": "worker_scheduler_request_mismatch",
                        "ordinal": ordinal,
                        **rank_payload,
                        "scheduler": expected_scheduled,
                        "worker": worker_scheduled,
                    }
                )
            if len(worker_ids) != len(set(worker_ids)) or set(worker_ids) != set(
                expected_scheduled
            ):
                violations.append(
                    {
                        "kind": "worker_request_rows_mismatch",
                        "ordinal": ordinal,
                        **rank_payload,
                        "worker_request_ids": worker_ids,
                        "scheduler_request_ids": sorted(expected_scheduled),
                    }
                )

            if canonical_worker is not None:
                for field in (
                    "request_ids",
                    "num_scheduled_tokens",
                    "block_tables",
                    "positions",
                    "slot_mappings",
                ):
                    if worker.get(field) != canonical_worker.get(field):
                        violations.append(
                            {
                                "kind": "worker_rank_state_mismatch",
                                "ordinal": ordinal,
                                **rank_payload,
                                "field": field,
                            }
                        )
            else:
                canonical_worker = worker

            worker_tables = worker.get("block_tables", {})
            expected_group_keys = {
                str(group_index)
                for request_id in worker_ids
                for group_index in range(len(states.get(request_id, [])))
            }
            if set(worker_tables) != expected_group_keys:
                violations.append(
                    {
                        "kind": "worker_group_set_mismatch",
                        "ordinal": ordinal,
                        **rank_payload,
                        "scheduler_groups": sorted(expected_group_keys),
                        "worker_groups": sorted(worker_tables),
                    }
                )
            for group_key, rows in worker_tables.items():
                row_indices = [int(row["request_index"]) for row in rows]
                if sorted(row_indices) != list(range(len(worker_ids))):
                    violations.append(
                        {
                            "kind": "worker_block_table_rows_mismatch",
                            "ordinal": ordinal,
                            **rank_payload,
                            "group_index": int(group_key),
                            "request_indices": row_indices,
                        }
                    )
                row_by_index = {
                    int(row["request_index"]): [int(x) for x in row["block_ids"]]
                    for row in rows
                }
                for request_index, request_id in enumerate(worker_ids):
                    actual = row_by_index.get(request_index, [])
                    groups = states.get(request_id)
                    expected = (
                        groups[int(group_key)]
                        if groups is not None and int(group_key) < len(groups)
                        else None
                    )
                    if expected is None or actual != expected:
                        violations.append(
                            {
                                "kind": "worker_scheduler_block_table_mismatch",
                                "ordinal": ordinal,
                                **rank_payload,
                                "request_id": request_id,
                                "group_index": int(group_key),
                                "scheduler_table": expected,
                                "worker_table": actual,
                            }
                        )
                    nonzero = [block for block in actual if block != 0]
                    duplicates = sorted(
                        block for block in set(nonzero) if nonzero.count(block) > 1
                    )
                    if duplicates:
                        violations.append(
                            {
                                "kind": "worker_duplicate_nonzero_block",
                                "ordinal": ordinal,
                                **rank_payload,
                                "request_id": request_id,
                                "group_index": int(group_key),
                                "block_ids": duplicates,
                            }
                        )

            positions = worker.get("positions", {}).get("values")
            position_numel = int(worker.get("positions", {}).get("numel", 0))
            expected_token_count = sum(worker_scheduled.values())
            if positions is None or position_numel < expected_token_count:
                violations.append(
                    {
                        "kind": "worker_positions_active_prefix_missing",
                        "ordinal": ordinal,
                        **rank_payload,
                        "expected": expected_token_count,
                        "actual": position_numel,
                    }
                )
                continue
            if len(positions) != position_numel or any(
                int(position) != 0 for position in positions[expected_token_count:]
            ):
                violations.append(
                    {
                        "kind": "worker_positions_padding_mismatch",
                        "ordinal": ordinal,
                        **rank_payload,
                        "active_tokens": expected_token_count,
                        "position_numel": position_numel,
                    }
                )
            for group_key, slot_summary in worker.get("slot_mappings", {}).items():
                slots = slot_summary.get("values")
                slot_numel = int(slot_summary.get("numel", 0))
                if (
                    slots is None
                    or slot_numel != expected_token_count
                    or len(slots) != slot_numel
                ):
                    violations.append(
                        {
                            "kind": "worker_slots_length_mismatch",
                            "ordinal": ordinal,
                            **rank_payload,
                            "group_index": int(group_key),
                        }
                    )
                    continue
                token_offset = 0
                for request_id in worker_ids:
                    token_count = worker_scheduled.get(request_id, 0)
                    groups = states.get(request_id)
                    table = (
                        groups[int(group_key)]
                        if groups is not None and int(group_key) < len(groups)
                        else []
                    )
                    for local_index in range(token_count):
                        index = token_offset + local_index
                        position = int(positions[index])
                        logical_block = position // block_size
                        if logical_block >= len(table):
                            violations.append(
                                {
                                    "kind": "slot_position_outside_scheduler_table",
                                    "ordinal": ordinal,
                                    **rank_payload,
                                    "request_id": request_id,
                                    "position": position,
                                }
                            )
                            continue
                        expected_slot = table[logical_block] * block_size + (
                            position % block_size
                        )
                        if int(slots[index]) != expected_slot:
                            violations.append(
                                {
                                    "kind": "slot_mapping_mismatch",
                                    "ordinal": ordinal,
                                    **rank_payload,
                                    "request_id": request_id,
                                    "position": position,
                                    "expected_slot": expected_slot,
                                    "worker_slot": int(slots[index]),
                                }
                            )
                    token_offset += token_count

    counts: dict[str, int] = {}
    for violation in violations:
        kind = violation["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "scheduler_steps": len(scheduler_steps),
        "worker_steps": sum(len(steps) for steps in workers_by_rank.values()),
        "worker_steps_by_rank": {
            f"pp{rank[0]}_tp{rank[1]}": len(steps)
            for rank, steps in sorted(workers_by_rank.items())
        },
        "worker_ranks": [list(rank) for rank in sorted(workers_by_rank)],
        "checked_steps": checked_steps,
        "violation_count": len(violations),
        "violation_counts": counts,
        "first_violations": violations[:20],
        "reuse_transitions": reuse_transitions,
        "state_invariants_passed": not violations,
    }


def _length_check(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "client_results.json"
    if not path.is_file():
        return {"available": False, "passed": False}
    with path.open(encoding="utf-8") as source:
        client = json.load(source)
    results = client.get("results", [])
    successful = [result for result in results if result.get("success")]
    complete = [
        result
        for result in successful
        if int(result["actual_output_length"]) == int(result["output_length"])
    ]
    return {
        "available": True,
        "total": len(results),
        "successful": len(successful),
        "complete_length": len(complete),
        "passed": len(results) > 0 and len(results) == len(successful) == len(complete),
    }


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    events = _load_events(run_dir)
    scheduler = _scheduler_steps(events)
    workers = _worker_steps_by_rank(events)
    length = _length_check(run_dir)
    if not scheduler or not workers:
        return {
            "schema": "vllm.pp_state_invariant_report.v1",
            "run_dir": str(run_dir),
            "verdict": "insufficient_state_evidence",
            "length_check": length,
            "state_evidence": {
                "scheduler_steps": len(scheduler),
                "worker_steps": sum(len(steps) for steps in workers.values()),
                "worker_steps_by_rank": {
                    f"pp{rank[0]}_tp{rank[1]}": len(steps)
                    for rank, steps in sorted(workers.items())
                },
            },
        }
    invariants = verify_trace_events(
        scheduler,
        workers,
        expected_worker_ranks={(0, 0), (0, 1), (1, 0), (1, 1)},
    )
    verdict = (
        "passed_state_invariants"
        if invariants["state_invariants_passed"] and length["passed"]
        else "failed_state_invariants"
    )
    return {
        "schema": "vllm.pp_state_invariant_report.v1",
        "run_dir": str(run_dir),
        "verdict": verdict,
        "length_check": length,
        "state_evidence": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = verify_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
