"""Reorder requests from E||B to B||E for the reorder baseline.

Reads a JSONL workload file and reorders each stitch-eligible request
so the body comes before the envelope. This allows prefix caching to
work on the now-front-aligned body.

Input format (JSONL):
  {"prompt": "...", "envelope_token_count": E, ...}

Output format (JSONL):
  {"prompt": "...", "envelope_token_count": null, ...}
  (reordered to B||E, no stitch needed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reorder_request(req: dict) -> dict:
    """Reorder a request from E||B to B||E.

    If the request has envelope_token_count, swap the envelope and body.
    The reordered request will have prefix cache hits on the body.
    """
    envelope_count = req.get("envelope_token_count")
    if envelope_count is None:
        # No stitch metadata; return as-is.
        return req

    prompt = req["prompt"]
    words = prompt.split()

    if envelope_count >= len(words):
        # Edge case: envelope covers all tokens.
        return req

    envelope_words = words[:envelope_count]
    body_words = words[envelope_count:]

    # Reorder: body first, then envelope.
    reordered_prompt = " ".join(body_words) + "\n" + " ".join(envelope_words)

    result = dict(req)
    result["prompt"] = reordered_prompt
    result["prompt_len"] = len(reordered_prompt.split())
    # Remove envelope_token_count since prefix cache now handles it.
    result["envelope_token_count"] = None
    result["_original_family"] = req.get("family", "unknown")
    result["family"] = "reordered"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reorder requests from E||B to B||E for reorder baseline."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input JSONL workload file.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSONL file (default: input_reordered.jsonl).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".reordered.jsonl")

    requests: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))

    reordered = [reorder_request(req) for req in requests]

    with open(output_path, "w") as f:
        for req in reordered:
            f.write(json.dumps(req) + "\n")

    # Print summary.
    n_reordered = sum(1 for r in reordered if r.get("family") == "reordered")
    print(f"Reordered {n_reordered}/{len(reordered)} requests -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
