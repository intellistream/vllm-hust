# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Download a long-form Hugging Face dataset and convert it into a JSONL dataset
compatible with ``vllm.benchmarks.datasets.CustomDataset``.

This is a convenience wrapper around
``vllm.benchmarks.datasets.create_txt_slices_dataset.create_txt_slices_jsonl``.

Example
-------

Download PG-19 training samples, merge them into a plain-text corpus, and
generate 64K-token benchmark prompts::

    uv run python benchmarks/prepare_long_context_dataset.py \\
        --hf-dataset pg19 \\
        --split train \\
        --text-column text \\
        --max-records 200 \\
        --corpus-path data/pg19.txt \\
        --output-path data/pg19_64k.jsonl \\
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
        --input-len 65536 \\
        --output-len 128 \\
        --num-prompts 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from vllm.benchmarks.datasets.create_txt_slices_dataset import (
    create_txt_slices_jsonl,
)


def dump_hf_text_corpus(
    *,
    hf_dataset: str,
    config_name: str | None,
    split: str,
    text_column: str,
    max_records: int | None,
    corpus_path: Path,
) -> Path:
    dataset = load_dataset(hf_dataset, name=config_name, split=split)

    if max_records is not None:
        dataset = dataset.select(range(min(max_records, len(dataset))))

    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    with corpus_path.open("w", encoding="utf-8") as f:
        wrote_any = False
        for row in dataset:
            text = row.get(text_column)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            if wrote_any:
                f.write("\n\n")
            f.write(text)
            wrote_any = True

    if not wrote_any:
        raise ValueError(
            f"No non-empty string values found in column '{text_column}' "
            f"for dataset '{hf_dataset}' split '{split}'."
        )

    return corpus_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a long-text HF dataset and build a vLLM custom "
        "benchmark dataset with large token windows.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="pg19",
        help="Hugging Face dataset name. Default: pg19.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional dataset config/subset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to read. Default: train.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing plain text. Default: text.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=200,
        help="Maximum number of HF records to merge into the corpus. "
        "Default: 200.",
    )
    parser.add_argument(
        "--corpus-path",
        required=True,
        help="Path to write the merged plain-text corpus.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write the output JSONL dataset.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer name or local path.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of benchmark prompts to generate. Default: 100.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=65536,
        help="Prompt length in tokens. Default: 65536.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Expected output length in tokens. Default: 128.",
    )
    parser.add_argument(
        "--range-ratio",
        type=str,
        default="0.0",
        help="Forwarded to create_txt_slices_dataset. Default: 0.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for prompt slicing. Default: 0.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for tokenizer loading.",
    )
    args = parser.parse_args()

    try:
        range_ratio: float | dict[str, float] = float(args.range_ratio)
    except ValueError:
        range_ratio = json.loads(args.range_ratio)

    corpus_path = dump_hf_text_corpus(
        hf_dataset=args.hf_dataset,
        config_name=args.config_name,
        split=args.split,
        text_column=args.text_column,
        max_records=args.max_records,
        corpus_path=Path(args.corpus_path),
    )

    create_txt_slices_jsonl(
        input_path=str(corpus_path),
        output_path=args.output_path,
        tokenizer_name=args.tokenizer,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        range_ratio=range_ratio,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
