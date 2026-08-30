# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online correctness entry for full-graph-parallel (dual-stream) replay.

This is a correctness smoke entry, NOT an acceptance or performance test:
graph capture/replay correctness is covered by the host tests in
``tests/v1/cudagraph/test_cudagraph_dispatch.py``.

Start the server with split-batch replay enabled, then run this script to
send concurrent chat requests. Any failed request aborts with a non-zero
exit code. Use ``--output`` to write a JSONL transcript so two runs
(baseline vs plugin) can be diffed for output consistency.

Server flags used during development:

  --additional-config '{
      "split_batch_config": {
        "enabled": true,
        "mode": "inplace_parallel",
        "num_splits": 2,
        "enable_parallel_streams": true,
        "enable_inplace_lazy_capture": true,
        "inplace_split_planner_policy": "largest_lower",
        "inplace_offset_match_policy": "exact",
        "inplace_parallel_replay_policy": "full_graph_parallel",
        "inplace_offset_capture_sizes": [8, 16, 32, 64],
        "parallel_capture_sizes": [8, 16, 32, 64]
      }
    }'
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
                         "cudagraph_capture_sizes": [8, 16, 32, 64, 128]}'
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import openai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent chat-requests smoke test for "
        "full-graph-parallel (dual-stream) replay."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "empty"))
    parser.add_argument(
        "--model",
        default=os.environ.get("FULL_GRAPH_PARALLEL_MODEL"),
        help="Served model name (must match the server's --served-model-name).",
    )
    parser.add_argument("--num-requests", type=int, default=80)
    parser.add_argument("--max-workers", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL file for paired baseline-vs-plugin diffing.",
    )
    return parser.parse_args()


def send_chat(
    client: openai.OpenAI, model: str, messages: list[dict], max_tokens: int
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("empty response content")
    return content


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit(
            "--model (or the FULL_GRAPH_PARALLEL_MODEL env var) is required"
        )

    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
    messages_list = [
        [{"role": "user", "content": "你好，请介绍一下自己"}]
        for _ in range(args.num_requests)
    ]

    def task(index: int, messages: list[dict]) -> tuple[int, str | None, str | None]:
        try:
            return index, send_chat(client, args.model, messages, args.max_tokens), None
        except Exception as exc:
            return index, None, f"{type(exc).__name__}: {exc}"

    print(f"Sending {args.num_requests} chat requests to {args.base_url} ...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(lambda pair: task(*pair), enumerate(messages_list)))

    rows = []
    error_count = 0
    for index, content, error in sorted(results):
        rows.append({"index": index, "response": content, "error": error})
        if error is not None:
            error_count += 1
            print(f"[{index}] ERROR: {error}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Transcript written to {args.output}")

    print(f"Received {args.num_requests - error_count}/{args.num_requests} responses")
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
