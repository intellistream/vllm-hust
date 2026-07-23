# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download and convert the public BurstGPT trace for the PP benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "8345c824bf744e21692186af2835521ba75e5f6d"
SOURCE_URL = (
    "https://raw.githubusercontent.com/HPMLL/BurstGPT/"
    f"{SOURCE_COMMIT}/data/BurstGPT_1.csv"
)
SOURCE_SHA256 = "46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a"
OUTPUT_SHA256 = "ebe08e38908795f3078940bbee3a151c84388450df8e453d8d9b89e90b500984"
OUTPUT_ROWS = 1_429_737
OUTPUT_FIELDS = ("timestamp", "input_length", "total_length")
DEFAULT_OUTPUT = Path(__file__).with_name("BurstGPT_1.csv")


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
        source.open("r", newline="", encoding="utf-8") as input_file,
        destination.open("w", newline="", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file)
        required_fields = {"Timestamp", "Request tokens", "Total tokens"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                "Unexpected BurstGPT schema: "
                f"required={sorted(required_fields)}, actual={reader.fieldnames}"
            )

        writer = csv.DictWriter(
            output_file,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        rows_written = 0
        for row in reader:
            writer.writerow(
                {
                    "timestamp": int(row["Timestamp"]),
                    "input_length": int(row["Request tokens"]),
                    "total_length": int(row["Total tokens"]),
                }
            )
            rows_written += 1
    return rows_written


def prepare_trace(output: Path, input_path: Path | None = None) -> None:
    if output.is_file() and file_sha256(output) == OUTPUT_SHA256:
        print(f"BurstGPT trace is ready: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp_dir:
        temp_dir_path = Path(temp_dir)
        source = input_path or temp_dir_path / "BurstGPT_1.source.csv"
        if input_path is None:
            print(f"Downloading {SOURCE_URL}")
            download_source(source)

        source_digest = file_sha256(source)
        if source_digest != SOURCE_SHA256:
            raise ValueError(
                "BurstGPT source checksum mismatch: "
                f"expected={SOURCE_SHA256}, actual={source_digest}"
            )

        converted = temp_dir_path / "BurstGPT_1.csv"
        rows_written = convert_source(source, converted)
        if rows_written != OUTPUT_ROWS:
            raise ValueError(
                "BurstGPT row count mismatch: "
                f"expected={OUTPUT_ROWS}, actual={rows_written}"
            )

        output_digest = file_sha256(converted)
        if output_digest != OUTPUT_SHA256:
            raise ValueError(
                "Converted BurstGPT checksum mismatch: "
                f"expected={OUTPUT_SHA256}, actual={output_digest}"
            )
        converted.replace(output)

    print(f"Prepared {rows_written} BurstGPT requests at {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--input",
        type=Path,
        help="Use an existing upstream CSV instead of downloading it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_trace(args.output, args.input)


if __name__ == "__main__":
    main()
