#!/usr/bin/env python3
"""LongBench-v2 client for validating KV tiering/offload serving.

This script sends a long OpenAI-compatible chat request to a running vLLM
server and prints the filesystem secondary-tier state before and after.

Example:
    python3 vllm/v1/kv_offload/tiering/my_tests/test_offload.py \
      --url http://127.0.0.1:8081/v1/chat/completions \
      --fs-root /tmp/vllm_kv_tiering_fs_verify \
      --max-context-chars 60000 \
      --repeat 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "/root/datasets/LongBench-v2/data.json"
DEFAULT_MODEL = "/root/models/Qwen2.5-7B-Instruct"
DEFAULT_URL = "http://127.0.0.1:8081/v1/chat/completions"
DEFAULT_FS_ROOT = "/tmp/vllm_kv_tiering_fs_verify"


def iter_json_array(path: str, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    pos = 0

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk and not buffer[pos:].strip():
                return
            buffer = buffer[pos:] + chunk
            pos = 0

            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1

                if not started:
                    if pos >= len(buffer):
                        break
                    if buffer[pos] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    started = True
                    pos += 1
                    continue

                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) and buffer[pos] == ",":
                    pos += 1
                    continue
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                if pos >= len(buffer):
                    break

                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                if not isinstance(obj, dict):
                    raise ValueError("LongBench-v2 entry is not a JSON object")
                pos = end
                yield obj


def choose_sample(path: str, index: int | None, require_length: str) -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    for i, item in enumerate(iter_json_array(path)):
        if index is not None:
            if i == index:
                return item
            continue
        if item.get("length") == require_length:
            return item
        if selected is None:
            selected = item

    if index is not None:
        raise IndexError(f"dataset index {index} not found")
    if selected is None:
        raise ValueError(f"no samples found in {path}")
    return selected


def build_prompt(sample: dict[str, Any], max_context_chars: int) -> str:
    context = str(sample.get("context", ""))
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars]

    choices = "\n".join(
        f"{letter}. {sample.get(f'choice_{letter}', '')}" for letter in "ABCD"
    )
    return (
        "Answer the following LongBench-v2 multiple-choice question. "
        "Use only the provided context. Reply with the best option letter and "
        "one short sentence.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{sample.get('question', '')}\n\n"
        f"Choices:\n{choices}\n"
    )


def fs_stats(root: str) -> tuple[int, int, list[str]]:
    root_path = Path(root)
    if not root_path.exists():
        return 0, 0, []

    count = 0
    total = 0
    examples: list[str] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        # config.json proves FS tier initialization, but not KV block storage.
        if path.name == "config.json":
            continue
        count += 1
        try:
            total += path.stat().st_size
        except OSError:
            pass
        if len(examples) < 10:
            examples.append(str(path))
    return count, total, examples


def post_chat(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    seed: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return json.loads(body)


def stable_json_hash(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fs-root", default=DEFAULT_FS_ROOT)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--require-length", default="long")
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    sample = choose_sample(args.dataset, args.index, args.require_length)
    prompt = build_prompt(sample, args.max_context_chars)
    payload_fingerprint = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n": 1,
        "stream": False,
        "seed": args.seed,
    }

    print("=" * 80)
    print("LongBench-v2 sample")
    print("=" * 80)
    print("id:", sample.get("_id"))
    print("domain:", sample.get("domain"))
    print("length:", sample.get("length"))
    print("context chars used:", min(len(str(sample.get("context", ""))), args.max_context_chars))
    print("prompt chars:", len(prompt))
    print("prompt sha256:", hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    print("payload sha256:", stable_json_hash(payload_fingerprint))
    print(
        "sampling:",
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
    )
    print("expected answer:", sample.get("answer"))

    before_count, before_bytes, before_examples = fs_stats(args.fs_root)
    print("\nFS tier before:")
    print("data files:", before_count)
    print("data bytes:", before_bytes)
    for path in before_examples:
        print(" ", path)

    responses: list[str] = []
    finish_reasons: list[Any] = []
    for i in range(args.repeat):
        print("\n" + "=" * 80)
        print(f"Request {i + 1}/{args.repeat}")
        print("=" * 80)
        start = time.time()
        response = post_chat(
            args.url,
            args.model,
            prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        elapsed = time.time() - start
        choice = response["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        responses.append(text)
        finish_reasons.append(finish_reason)
        usage = response.get("usage", {})
        print("elapsed_sec:", f"{elapsed:.2f}")
        print("usage:", usage)
        print("finish_reason:", finish_reason)
        print("response sha256:", hashlib.sha256(text.encode("utf-8")).hexdigest())
        print("response:", text)

        count, size, examples = fs_stats(args.fs_root)
        print("FS tier data files:", count)
        print("FS tier data bytes:", size)
        for path in examples:
            print(" ", path)

    after_count, after_bytes, _ = fs_stats(args.fs_root)
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("new data files:", after_count - before_count)
    print("new data bytes:", after_bytes - before_bytes)
    if responses:
        unique_responses = len(set(responses))
        print("unique responses:", unique_responses)
        print("finish reasons:", finish_reasons)
        if unique_responses == 1:
            print("PASS: repeated responses are identical.")
        else:
            print("FAIL: repeated responses differ.")
    if after_count > before_count or after_bytes > before_bytes:
        print("PASS: FS secondary tier appears to have stored KV block data.")
    else:
        print(
            "WARN: no FS secondary data growth observed. The request may not "
            "have triggered store/cascade, or the server may need a smaller "
            "cpu_bytes_to_use / longer prompt."
        )


if __name__ == "__main__":
    main()
