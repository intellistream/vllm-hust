# SPDX-License-Identifier: Apache-2.0
"""Run calibration traces against one instrumented vLLM server."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

PROXY_VARIABLES = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--kv-cache-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--max-workers", type=int, default=1024)
    parser.add_argument("--health-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _workloads(workload_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in workload_dir.glob("*.csv")
        if path.name != "profile_workload_summary.csv"
    )


def _request_count(workload: Path) -> int:
    with workload.open(newline="", encoding="utf-8") as source:
        return sum(1 for _ in csv.DictReader(source))


def _healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/health", timeout=5
        ) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def _wait_for_server(process: subprocess.Popen, base_url: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _healthy(base_url):
            return
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before becoming healthy")
        time.sleep(5)
    raise RuntimeError(f"timed out after {timeout}s waiting for vLLM")


def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _profile_files(profile_base: Path) -> list[Path]:
    return sorted(profile_base.parent.glob(f"{profile_base.stem}_pp*_tp*.csv"))


def _profile_offsets(profile_base: Path) -> dict[Path, int]:
    return {path: path.stat().st_size for path in _profile_files(profile_base)}


def _extract_profiles(
    profile_base: Path, offsets: dict[Path, int], output_dir: Path
) -> list[Path]:
    extracted = []
    for profile_path in sorted(set(_profile_files(profile_base)) | set(offsets)):
        if not profile_path.exists():
            continue
        offset = offsets.get(profile_path, 0)
        if profile_path.stat().st_size <= offset:
            continue
        with profile_path.open(newline="", encoding="utf-8") as source:
            header = next(csv.reader(source))
        with profile_path.open("rb") as source:
            source.seek(offset)
            appended = source.read().decode("utf-8")
        rows = [row for row in csv.reader(io.StringIO(appended)) if row]
        if offset == 0 and rows and rows[0] == header:
            rows = rows[1:]
        if not rows:
            continue
        output_path = output_dir / profile_path.name
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(rows)
        extracted.append(output_path)
    return extracted


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"completed": {}}
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(checkpoint, output, indent=2)
        output.write("\n")
    temporary.replace(path)


def _validate_result(path: Path, expected_requests: int) -> None:
    with path.open(encoding="utf-8") as source:
        result = json.load(source)
    requests = result.get("results", [])
    failures = [request for request in requests if request.get("success") is not True]
    if len(requests) != expected_requests or failures:
        raise RuntimeError(
            f"client result has {len(requests)}/{expected_requests} requests and "
            f"{len(failures)} failures"
        )


def main() -> None:
    args = parse_args()
    workloads = _workloads(args.workload_dir)
    if not workloads:
        raise SystemExit(f"no workloads found in {args.workload_dir}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path) if args.resume else {"completed": {}}
    profile_base = args.run_dir / "profile_stream" / "profile.csv"
    profile_base.parent.mkdir(parents=True, exist_ok=True)
    server_log = args.run_dir / "server.log"

    environment = os.environ.copy()
    for variable in PROXY_VARIABLES:
        environment.pop(variable, None)
    environment.update(
        {
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
            "VLLM_PROFILE_PP_OPT_ENABLED": "1",
            "VLLM_PROFILE_PP_OPT_OUTPUT_PATH": str(profile_base),
            "PATH": (
                f"{Path(args.vllm_bin).resolve().parent}{os.pathsep}"
                f"{environment.get('PATH', '')}"
            ),
        }
    )
    with server_log.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            ["bash", str(args.server_script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            preexec_fn=os.setsid,
        )

    try:
        _wait_for_server(server, args.base_url, args.health_timeout)
        for workload in workloads:
            key = str(workload.resolve())
            existing = checkpoint["completed"].get(key)
            if args.resume and existing:
                profile_dir = Path(existing["profile_dir"])
                if Path(existing["result_json"]).is_file() and list(
                    profile_dir.glob("profile_pp*_tp*.csv")
                ):
                    print(f"Skipping completed workload: {workload.name}")
                    continue

            request_count = _request_count(workload)
            output_dir = args.run_dir / workload.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            result_json = output_dir / "client_results.json"
            client_log = output_dir / "client.log"
            offsets = _profile_offsets(profile_base)
            command = [
                args.python,
                str(args.client),
                "--workload",
                str(workload),
                "--kv-cache-size",
                str(args.kv_cache_size),
                "--base-url",
                args.base_url,
                "--model-name",
                args.model_name,
                "--max-model-len",
                str(args.max_model_len),
                "--request-num",
                str(request_count),
                "--ignore-timestamps",
                "--vocab-size",
                str(args.vocab_size),
                "--max-workers",
                str(args.max_workers),
                "--output",
                str(result_json),
                "--log",
                str(client_log),
            ]
            print(f"Running {workload.name} ({request_count} requests)")
            subprocess.run(command, check=True, env=environment)
            _validate_result(result_json, request_count)
            extracted = _extract_profiles(profile_base, offsets, output_dir)
            if not extracted:
                raise RuntimeError(f"no profile rows captured for {workload.name}")
            checkpoint["completed"][key] = {
                "workload": str(workload),
                "request_num": request_count,
                "profile_dir": str(output_dir),
                "result_json": str(result_json),
                "client_log": str(client_log),
                "server_log": str(server_log),
            }
            _save_checkpoint(checkpoint_path, checkpoint)
    finally:
        _stop_process_group(server)


if __name__ == "__main__":
    main()
