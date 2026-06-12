"""Adapter: llm-serving-workloads → segment reuse JSONL workloads.

Generates E||B (envelope||body) workloads from the llm-serving-workloads
library for segment reuse evaluation. Each workload family has a known
structure where some blocks are stable (body, shared across requests)
and some are dynamic (envelope, unique per request).

Supported workload families:
- rag-followup: retrieval context (body) + query/answer (envelope)
- session-affine: service/tenant config (body) + session/turn (envelope)
- tool-scaffold: scaffold/tools (body) + transcript/action (envelope)
- shared-prefix-multi-tenant: shared prefix (body) + tenant tail (envelope)
- long-context-doc-analysis: doc sections (body) + analysis query (envelope)
- dynamic-rag-corpus-update: stable docs (body) + dynamic query (envelope)
- memory-write-then-reuse: memory config (body) + write/retrieve (envelope)

Usage:
    python benchmarks/workload_from_library.py \
        --families rag-followup session-affine tool-scaffold \
        --requests-per-family 24 \
        --output workload_library.jsonl

The output JSONL is compatible with benchmark_segment_reuse.py format
and can be used with run_segment_reuse_eval.sh directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Workload-family-aware E||B decomposition
# ---------------------------------------------------------------------------

# Each family's prompt is a list of blocks joined by "\n\n".
# We classify blocks as "body" (stable, shared) or "envelope" (dynamic).
# The block order in the original prompt is B||E (body first, envelope last).

FAMILY_BLOCK_ROLES: dict[str, dict[str, Any]] = {
    "rag-followup": {
        # prompt = scaffold, synopsis, *doc_blocks, previous_answer, question
        "body_blocks": "front",  # first N blocks are body
        "envelope_blocks": "back",  # last 2 blocks are envelope
        "envelope_block_count": 2,
        "description": "RAG follow-up: retrieval context (body) + query (envelope)",
    },
    "session-affine-multi-turn": {
        # prompt = service, tenant, recovery, session, recent_turn, user_turn
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,  # recent_turn_block, user_turn_block
        "description": "Session affine: service config (body) + turn (envelope)",
    },
    "session-affine-bursty": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Session burst: service config (body) + burst (envelope)",
    },
    "tool-scaffold-agent": {
        # prompt = scaffold, tool_schema, plan, transcript, next_action
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,  # transcript_block, next_action_block
        "description": "Tool scaffold: agent setup (body) + action (envelope)",
    },
    "long-context-doc-analysis": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Doc analysis: document sections (body) + query (envelope)",
    },
    "shared-prefix-multi-tenant-assistant": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Multi-tenant: shared prefix (body) + tenant tail (envelope)",
    },
    "dynamic-rag-corpus-update": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Dynamic RAG: stable docs (body) + dynamic query (envelope)",
    },
    "memory-write-then-reuse": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Memory: config (body) + write/retrieve (envelope)",
    },
    "repo-aware-coding-assistant": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Coding: repo context (body) + edit task (envelope)",
    },
    "experiment-planning-assistant": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Experiment: plan config (body) + branch (envelope)",
    },
    "simulation-analysis-verification": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Simulation: setup (body) + analysis step (envelope)",
    },
    "async-document-pipeline": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Pipeline: template (body) + job step (envelope)",
    },
    "realtime-voice-assistant": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Voice: voice config (body) + utterance (envelope)",
    },
    "session-continuation-maintenance": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Continuation: session config (body) + turn (envelope)",
    },
    "preemption-resume-long-decode": {
        "body_blocks": "front",
        "envelope_blocks": "back",
        "envelope_block_count": 2,
        "description": "Preemption: context (body) + resume segment (envelope)",
    },
}


class WhitespaceTokenizer:
    """Simple whitespace tokenizer matching llm-serving-workloads convention."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


def decompose_prompt(prompt: str, family: str) -> tuple[str, str, int]:
    """Split a B||E prompt into (envelope, body, envelope_word_count).

    The original prompt has body blocks first, envelope blocks last,
    separated by double newlines.

    Returns:
        (envelope_text, body_text, envelope_word_count)
    """
    role_spec = FAMILY_BLOCK_ROLES.get(family)
    if role_spec is None:
        # Unknown family: treat entire prompt as body (no stitch).
        return "", prompt, 0

    env_count = role_spec["envelope_block_count"]

    # Split into blocks by "\n\n".
    blocks = prompt.split("\n\n")

    if len(blocks) <= env_count:
        # Not enough blocks to decompose.
        return "", prompt, 0

    body_blocks = blocks[:-env_count]
    envelope_blocks = blocks[-env_count:]

    body_text = "\n\n".join(body_blocks)
    envelope_text = "\n\n".join(envelope_blocks)
    envelope_word_count = len(envelope_text.split())

    return envelope_text, body_text, envelope_word_count


def compose_eb_prompt(envelope_text: str, body_text: str) -> str:
    """Compose an E||B prompt (envelope first, body second)."""
    if not envelope_text:
        return body_text
    if not body_text:
        return envelope_text
    return f"{envelope_text}\n\n---\n\n{body_text}"


def generate_family_workload(
    dataset_name: str,
    num_requests: int,
    seed: int = 42,
) -> list[dict]:
    """Generate E||B workload requests for a single family.

    Uses llm-serving-workloads to generate the base prompts, then
    decomposes and recomposes them in E||B order.

    Returns a list of JSONL-compatible dicts.
    """
    try:
        from llm_serving_workloads import (
            generate_repo_local_workload_requests,
            is_repo_local_workload_dataset,
            WORKLOAD_BENCHMARK_SHAPE_CATALOG,
        )
    except ImportError:
        print(
            "ERROR: llm-serving-workloads not installed. "
            "Install with: pip install -e /path/to/llm-serving-workloads",
            file=sys.stderr,
        )
        return []

    if not is_repo_local_workload_dataset(dataset_name):
        print(f"WARNING: {dataset_name} is not a repo-local dataset", file=sys.stderr)
        return []

    # Get default shape from catalog.
    shape = dict(WORKLOAD_BENCHMARK_SHAPE_CATALOG.get(dataset_name, {}))
    num_groups = shape.pop("num_groups", 8)
    prompts_per_group = shape.pop("prompts_per_group", 4)
    system_prompt_len = shape.pop("system_prompt_len", 768)
    question_len = shape.pop("question_len", 64)
    output_len = shape.pop("output_len", 96)

    # Adjust to get approximately num_requests total.
    total_from_shape = num_groups * prompts_per_group
    if total_from_shape < num_requests:
        # Scale up groups.
        num_groups = max(num_groups, (num_requests + prompts_per_group - 1) // prompts_per_group)

    # Generate base requests using llm-serving-workloads.
    tokenizer = WhitespaceTokenizer()
    rows = generate_repo_local_workload_requests(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        dp_size=8,
        num_prompts=num_requests,
        num_groups=num_groups,
        system_prompt_len=system_prompt_len,
        question_len=question_len,
        output_len=output_len,
        seed=seed,
        **shape,
    )

    # Decompose each prompt and recompose as E||B.
    results: list[dict] = []
    for idx, row in enumerate(rows[:num_requests]):
        envelope_text, body_text, env_word_count = decompose_prompt(
            row.prompt, dataset_name
        )

        # E||B order: envelope first, body second.
        eb_prompt = compose_eb_prompt(envelope_text, body_text)
        total_words = len(eb_prompt.split())

        # Adjust envelope count for the separator tokens ("---").
        if envelope_text and body_text:
            env_word_count += 1  # account for "---" separator

        req = {
            "prompt": eb_prompt,
            "prompt_len": total_words,
            "expected_output_len": row.output_len,
            "envelope_token_count": env_word_count if env_word_count > 0 else None,
            "family": dataset_name,
            "request_id": f"{dataset_name}-{idx:04d}",
            "order": "E||B",
        }

        # Preserve metadata for analysis.
        if row.repo_local_metadata:
            req["_metadata"] = {
                "primary_anchor_id": row.repo_local_metadata.get("primary_anchor_id", ""),
                "anchor_index": row.repo_local_metadata.get("anchor_index", 0),
                "turn_index": row.repo_local_metadata.get("turn_index", 0),
                "home_rank": row.repo_local_metadata.get("home_rank", 0),
            }

        results.append(req)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_FAMILIES = [
    "rag-followup",
    "session-affine-multi-turn",
    "tool-scaffold-agent",
]

ALL_FAMILIES = sorted(FAMILY_BLOCK_ROLES.keys())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate segment reuse workloads from llm-serving-workloads."
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILIES,
        choices=ALL_FAMILIES,
        help="Workload families to include.",
    )
    parser.add_argument(
        "--requests-per-family",
        type=int,
        default=24,
        help="Number of requests per family.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="workload_library.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List supported families and exit.",
    )
    args = parser.parse_args()

    if args.list_families:
        for name in ALL_FAMILIES:
            role = FAMILY_BLOCK_ROLES[name]
            print(f"  {name}: {role['description']}")
        return 0

    all_requests: list[dict] = []

    for family in args.families:
        print(f"Generating {args.requests_per_family} requests for {family}...")
        family_requests = generate_family_workload(
            dataset_name=family,
            num_requests=args.requests_per_family,
            seed=args.seed,
        )
        if family_requests:
            all_requests.extend(family_requests)
            stitch_eligible = sum(
                1 for r in family_requests if r["envelope_token_count"] is not None
            )
            avg_len = sum(r["prompt_len"] for r in family_requests) / len(family_requests)
            avg_env = sum(
                r["envelope_token_count"]
                for r in family_requests
                if r["envelope_token_count"] is not None
            ) / max(stitch_eligible, 1)
            print(
                f"  {family}: {len(family_requests)} reqs, "
                f"avg_prompt_len={avg_len:.0f}, "
                f"avg_envelope={avg_env:.0f}, "
                f"stitch_eligible={stitch_eligible}"
            )
        else:
            print(f"  {family}: 0 requests (generation failed)")

    if not all_requests:
        print("ERROR: No requests generated.", file=sys.stderr)
        return 1

    # Write JSONL.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in all_requests:
            f.write(json.dumps(req) + "\n")

    # Summary.
    print(f"\nTotal: {len(all_requests)} requests -> {output_path}")
    families_used = set(r["family"] for r in all_requests)
    print(f"Families: {', '.join(sorted(families_used))}")
    stitch_total = sum(1 for r in all_requests if r["envelope_token_count"] is not None)
    print(f"Stitch eligible: {stitch_total}/{len(all_requests)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
