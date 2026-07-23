# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download and convert the Mooncake conversation trace for the PP benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "3cca71daccf2a7afb8fe3f0295358f70e3a69fdb"
SOURCE_URL = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/"
    f"{SOURCE_COMMIT}/FAST25-release/traces/conversation_trace.jsonl"
)
SOURCE_SHA256 = "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df"
OUTPUT_SHA256 = "a793b1bf8e64152376f3e476f01f4bf91eb080a71f734265cc75635271307762"
OUTPUT_ROWS = 12_031
OUTPUT_FIELDS = ("timestamp", "input_length", "total_length")
DEFAULT_OUTPUT = Path(__file__).with_name("conversation_trace.csv")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(destination: Path) -> None:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "vllm-pp-opt-benchmark"},
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as file,
    ):
        shutil.copyfileobj(response, file)


def convert_source(source: Path, destination: Path) -> int:
    with (
        source.open("r", encoding="utf-8") as input_file,
        destination.open("w", newline="", encoding="utf-8") as output_file,
    ):
        writer = csv.DictWriter(
            output_file,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        rows_written = 0
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                timestamp = int(record["timestamp"])
                input_length = int(record["input_length"])
                output_length = int(record["output_length"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid Mooncake conversation record on line {line_number}"
                ) from error

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "input_length": input_length,
                    "total_length": input_length + output_length,
                }
            )
            rows_written += 1
    return rows_written


def prepare_trace(output: Path, input_path: Path | None = None) -> None:
    if output.is_file() and file_sha256(output) == OUTPUT_SHA256:
        print(f"Mooncake conversation trace is ready: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp_dir:
        temp_dir_path = Path(temp_dir)
        source = input_path or temp_dir_path / "conversation_trace.jsonl"
        if input_path is None:
            print(f"Downloading {SOURCE_URL}")
            download_source(source)

        source_digest = file_sha256(source)
        if source_digest != SOURCE_SHA256:
            raise ValueError(
                "Mooncake source checksum mismatch: "
                f"expected={SOURCE_SHA256}, actual={source_digest}"
            )

        converted = temp_dir_path / "conversation_trace.csv"
        rows_written = convert_source(source, converted)
        if rows_written != OUTPUT_ROWS:
            raise ValueError(
                "Mooncake conversation row count mismatch: "
                f"expected={OUTPUT_ROWS}, actual={rows_written}"
            )

        output_digest = file_sha256(converted)
        if output_digest != OUTPUT_SHA256:
            raise ValueError(
                "Converted conversation trace checksum mismatch: "
                f"expected={OUTPUT_SHA256}, actual={output_digest}"
            )
        converted.replace(output)

    print(f"Prepared {rows_written} Mooncake conversation requests at {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--input",
        type=Path,
        help="Use an existing upstream JSONL file instead of downloading it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_trace(args.output, args.input)


if __name__ == "__main__":
    main()
