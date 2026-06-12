"""Generate segment reuse workload for vLLM benchmarking.

Creates a synthetic workload with three prompt families that exhibit
different envelope||body decomposition patterns:

1. obvious (24 reqs): Fixed system prompt first, dynamic user question.
   Prefix caching works well. Serves as the "easy" baseline.

2. semi (24 reqs): Dynamic envelope (user question + RAG context) then
   stable body (system prompt + tools). Prefix caching fails at token 0
   because envelope varies per request.

3. agent (24 reqs): Dynamic envelope (user query + retrieved docs) then
   large stable body (persona + tools + detailed instructions). The body
   is large (500-1500 tokens) and shared across requests.

Each request is a JSONL line compatible with `vllm bench serve`:
  {"prompt": "...", "prompt_len": N, "expected_output_len": M,
   "envelope_token_count": E}

The `envelope_token_count` field enables segment reuse when
VLLM_SEGMENT_REUSE=stitch is set on the server.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path


def random_words(count: int, rng: random.Random) -> str:
    """Generate approximately `count` random words."""
    words = []
    remaining = count
    while remaining > 0:
        word_len = rng.randint(3, 8)
        word = "".join(rng.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
        remaining -= 1
    return " ".join(words)


def generate_system_prompt(rng: random.Random, base_len: int = 200) -> str:
    """Generate a stable system prompt (the 'body' component)."""
    return (
        "You are a helpful AI assistant specialized in answering questions "
        "accurately and concisely. " + random_words(base_len, rng)
    )


def generate_tools(rng: random.Random, count: int = 3) -> str:
    """Generate tool definitions (part of the body)."""
    tools = []
    for i in range(count):
        tool_name = f"tool_{i}"
        tool_desc = random_words(30, rng)
        tools.append(f"- {tool_name}: {tool_desc}")
    return "Available tools:\n" + "\n".join(tools)


def generate_envelope(rng: random.Random, env_len: int = 80) -> str:
    """Generate a dynamic envelope (the request-specific part)."""
    user_question = "What is " + random_words(10, rng) + "?"
    context = "Context: " + random_words(env_len, rng)
    return f"User: {user_question}\n{context}\n"


def generate_obvious_request(
    rng: random.Random, shared_system: str, idx: int
) -> dict:
    """Generate an 'obvious' request where system prompt comes first.

    Structure: [system_prompt] [user_question]
    Prefix cache works well because the front is shared.
    """
    user_q = "Question " + str(idx) + ": " + random_words(30, rng) + "?"
    prompt = f"{shared_system}\n\nUser: {user_q}"
    # Envelope is just the user question at the end.
    envelope_tokens = len(user_q.split()) + 2  # +2 for "User:" prefix
    return {
        "prompt": prompt,
        "prompt_len": len(prompt.split()),
        "expected_output_len": 50,
        "envelope_token_count": None,  # No stitch needed; prefix works
        "family": "obvious",
    }


def generate_semi_request(
    rng: random.Random, shared_body: str, idx: int
) -> dict:
    """Generate a 'semi' request with dynamic envelope before shared body.

    Structure: [user_question + RAG context] [system_prompt + tools]
    Prefix cache fails because envelope varies per request.
    """
    envelope = generate_envelope(rng, env_len=40)
    prompt = f"{envelope}\n{shared_body}"
    envelope_tokens = len(envelope.split())
    return {
        "prompt": prompt,
        "prompt_len": len(prompt.split()),
        "expected_output_len": 50,
        "envelope_token_count": envelope_tokens,
        "family": "semi",
    }


def generate_agent_request(
    rng: random.Random, shared_body: str, idx: int
) -> dict:
    """Generate an 'agent' request with dynamic envelope + large body.

    Structure: [user_query + retrieved_docs] [persona + tools + instructions]
    The body is large and shared. Prefix cache completely fails.
    """
    # Larger envelope with retrieved document context.
    retrieved_docs = "Retrieved documents:\n" + random_words(100, rng)
    user_query = "Based on the documents above, " + random_words(15, rng) + "?"
    envelope = f"{user_query}\n\n{retrieved_docs}"
    prompt = f"{envelope}\n\n---\n\n{shared_body}"
    envelope_tokens = len(envelope.split())
    return {
        "prompt": prompt,
        "prompt_len": len(prompt.split()),
        "expected_output_len": 80,
        "envelope_token_count": envelope_tokens,
        "family": "agent",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate segment reuse workload for vLLM benchmarking."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="workload_segment_reuse.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--obvious-count", type=int, default=24,
        help="Number of obvious-template requests.",
    )
    parser.add_argument(
        "--semi-count", type=int, default=24,
        help="Number of semi-template requests.",
    )
    parser.add_argument(
        "--agent-count", type=int, default=24,
        help="Number of agent-workflow requests.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Shared prompts for each family.
    shared_system_obvious = generate_system_prompt(rng, base_len=200)
    shared_body_semi = (
        generate_system_prompt(rng, base_len=150)
        + "\n\n" + generate_tools(rng, count=3)
    )
    shared_body_agent = (
        generate_system_prompt(rng, base_len=300)
        + "\n\n" + generate_tools(rng, count=5)
        + "\n\nDetailed instructions: " + random_words(200, rng)
    )

    requests: list[dict] = []

    # Generate obvious requests.
    for i in range(args.obvious_count):
        req = generate_obvious_request(rng, shared_system_obvious, i)
        req["request_id"] = f"obvious-{i:04d}"
        requests.append(req)

    # Generate semi requests.
    for i in range(args.semi_count):
        req = generate_semi_request(rng, shared_body_semi, i)
        req["request_id"] = f"semi-{i:04d}"
        requests.append(req)

    # Generate agent requests.
    for i in range(args.agent_count):
        req = generate_agent_request(rng, shared_body_agent, i)
        req["request_id"] = f"agent-{i:04d}"
        requests.append(req)

    # Write JSONL.
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Print summary.
    print(f"Generated {len(requests)} requests -> {output_path}")
    for family in ["obvious", "semi", "agent"]:
        family_reqs = [r for r in requests if r["family"] == family]
        avg_len = sum(r["prompt_len"] for r in family_reqs) / len(family_reqs)
        stitch_count = sum(
            1 for r in family_reqs if r["envelope_token_count"] is not None
        )
        print(
            f"  {family}: {len(family_reqs)} reqs, "
            f"avg_prompt_len={avg_len:.0f}, "
            f"stitch_eligible={stitch_count}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
