# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Driver program for running inference workloads from CSV files.

Usage:
    python benchmarks/pp_opt/client.py -w workload.csv -k 100000 \
        --base-url http://localhost:8000 --model-name example-model

    # Ignore timestamps (send ASAP when KV cache available)
    python benchmarks/pp_opt/client.py -w workload.csv -k 100000 \
        --ignore-timestamps --model-name example-model

    # Slow down timestamps by 2x
    python benchmarks/pp_opt/client.py -w workload.csv -k 100000 -r 2.0 \
        --model-name example-model

    # Speed up timestamps by 2x
    python benchmarks/pp_opt/client.py -w workload.csv -k 100000 -r 0.5 \
        --model-name example-model
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import openai


def setup_logging(log_file: str, verbose: bool = False) -> logging.Logger:
    """
    Set up logging to both console and file.

    Args:
        log_file: Path to the log file
        verbose: If True, also print to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("workload_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler - always log to file
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - always show completions
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


@dataclass
class WorkloadEntry:
    """A single entry from the workload CSV file."""

    timestamp: float  # in milliseconds
    input_length: int
    output_length: int
    line_number: int  # original line number in the file (for stable sorting)

    @classmethod
    def from_dict(cls, d: dict, line_number: int) -> "WorkloadEntry":
        input_length = int(d["input_length"])
        if "output_length" in d:
            output_length = int(d["output_length"])
        elif "total_length" in d:
            output_length = int(d["total_length"]) - input_length
        else:
            raise ValueError("Missing 'output_length' or 'total_length' column")
        return cls(
            timestamp=float(d["timestamp"]),
            input_length=input_length,
            output_length=output_length,
            line_number=line_number,
        )


@dataclass
class RequestResult:
    """Result of a single request."""

    request_id: int
    input_length: int
    output_length: int
    actual_output_length: int
    output_token_ids_sha256: str | None
    scheduled_time_ms: float  # when the request was scheduled to be sent
    actual_send_time_ms: float  # when it was actually sent (after KV wait)
    actual_send_timestamp_s: float  # absolute Unix timestamp when sent
    completion_time_ms: float  # when the response was received
    completion_timestamp_s: float  # absolute Unix timestamp when completed
    latency_ms: float  # total latency (completion - actual_send)
    wait_time_ms: float  # time spent waiting for KV cache
    success: bool
    error: str | None = None


class Request:
    """Represents a single inference request."""

    def __init__(self, input_length: int, output_length: int, vocab_size: int):
        self.input_length = input_length
        self.output_length = output_length
        self.vocab_size = vocab_size
        self.input_tokens = [i % vocab_size for i in range(input_length)]

    @property
    def target_length(self) -> int:
        return self.input_length + self.output_length

    def __repr__(self) -> str:
        return (
            f"Request(input_length={self.input_length}, "
            f"output_length={self.output_length})"
        )


class VLLMClient:
    """Client for sending requests to a vLLM server via the OpenAI API."""

    def __init__(
        self,
        base_url: str,
        kv_cache_size: int,
        model_name: str,
        max_workers: int = 256,
        request_timeout: float = 7200.0,
        verbose: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.kv_cache_size = kv_cache_size
        self.model_name = model_name
        self.max_workers = max_workers
        self.verbose = verbose
        openai_base = self.base_url + "/v1"
        self._client = openai.OpenAI(
            base_url=openai_base,
            api_key="dummy",
            max_retries=0,
            timeout=request_timeout,
        )

    def send_request(self, request: Request, request_id: int) -> dict:
        """Send a single request to the vLLM /v1/completions endpoint."""
        start = time.time()
        try:
            response = self._client.completions.create(
                model=self.model_name,
                prompt=request.input_tokens,
                max_tokens=request.output_length,
                temperature=0.0,
                extra_body={"ignore_eos": True, "return_token_ids": True},
            )
            latency = time.time() - start
            token_ids = response.choices[0].token_ids
            return {
                "success": True,
                "latency": latency,
                "error": None,
                "actual_output_length": len(token_ids),
                "output_token_ids_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        except Exception as e:
            latency = time.time() - start
            error = f"{e}\n\nTraceback:\n{traceback.format_exc()}"
            if self.verbose:
                print(f"Request {request_id} failed: {error}")
            return {
                "success": False,
                "latency": latency,
                "error": error,
                "actual_output_length": 0,
                "output_token_ids_sha256": None,
            }


def load_workload(filepath: str) -> list[WorkloadEntry]:
    """Load workload from a CSV file."""
    entries = []
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"timestamp", "input_length"}

        if not reader.fieldnames:
            print("Error: CSV file is empty or missing a header row", file=sys.stderr)
            return entries

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            missing_columns_text = ", ".join(sorted(missing_columns))
            print(
                f"Error: CSV missing required columns: {missing_columns_text}",
                file=sys.stderr,
            )
            return entries

        if not ({"output_length", "total_length"} & set(reader.fieldnames)):
            print(
                "Error: CSV must contain either 'output_length' or "
                "'total_length' column",
                file=sys.stderr,
            )
            return entries

        for row_num, row in enumerate(reader, 2):
            try:
                entry = WorkloadEntry.from_dict(row, row_num)
                if entry.output_length <= 0:
                    print(
                        f"Warning: Skipping row {row_num} with non-positive "
                        "output_length",
                        file=sys.stderr,
                    )
                    continue
                entries.append(entry)
            except (KeyError, TypeError, ValueError) as e:
                print(f"Warning: Skipping invalid row {row_num}: {e}", file=sys.stderr)
    return entries


class TimedWorkloadRunner:
    """
    Runs a workload with timestamp-based scheduling.

    Handles both timestamp-based and ignore-timestamp modes.
    """

    def __init__(
        self,
        client: VLLMClient,
        time_ratio: float = 1.0,
        ignore_timestamps: bool = False,
        verbose: bool = False,
        on_request_sent: Callable[[int, int, int, int, int, int, float, int, int], None]
        | None = None,
        on_request_complete: Callable[["RequestResult", int, int, int, int, int], None]
        | None = None,
    ):
        """
        Initialize the timed workload runner.

        Args:
            client: The client to use for sending requests
            time_ratio: Multiplier for timestamps (>1 = slower, <1 = faster)
            ignore_timestamps: If True, send requests ASAP (only KV-limited)
            verbose: Whether to print detailed progress
            on_request_sent: Callback when request is sent.
                Args: request ID, input/output lengths, progress counters,
                elapsed time, and KV usage.
            on_request_complete: Callback when request completes.
                Args: result, progress counters, and KV usage.
        """
        self.client = client
        self.time_ratio = time_ratio
        self.ignore_timestamps = ignore_timestamps
        self.verbose = verbose
        self.on_request_sent = on_request_sent
        self.on_request_complete = on_request_complete

        # Synchronization for KV cache management
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.current_kv_usage = 0

        # Progress tracking
        self.in_flight_count = 0
        self.completed_count = 0
        self.total_requests = 0

        # Timing
        self.start_time: float | None = None
        self.first_timestamp: float | None = None

    def _get_elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000

    def _wait_for_timestamp(self, timestamp_ms: float) -> None:
        """Wait until the adjusted timestamp has passed."""
        if self.ignore_timestamps:
            return

        # Calculate target time relative to first timestamp
        relative_timestamp = timestamp_ms - (self.first_timestamp or 0)
        adjusted_timestamp = relative_timestamp * self.time_ratio

        # Wait until target time
        while True:
            elapsed = self._get_elapsed_ms()
            remaining = adjusted_timestamp - elapsed
            if remaining <= 0:
                break
            # Sleep for remaining time (convert to seconds)
            time.sleep(min(remaining / 1000, 0.1))  # Max 100ms sleep intervals

    def _wait_for_kv_cache(self, required_size: int) -> float:
        """
        Wait until there's enough KV cache available.

        Returns the time spent waiting in milliseconds.
        """
        wait_start = time.time()

        with self.condition:
            while self.current_kv_usage + required_size > self.client.kv_cache_size:
                self.condition.wait(timeout=0.01)
            self.current_kv_usage += required_size

        return (time.time() - wait_start) * 1000

    def _release_kv_cache(self, size: int) -> None:
        """Release KV cache after request completion."""
        with self.condition:
            self.current_kv_usage -= size
            self.condition.notify_all()

    def _process_request(
        self,
        entry: WorkloadEntry,
        request_id: int,
        scheduled_time_ms: float,
        wait_time_ms: float,
        vocab_size: int,
        pages_used_at_alloc: int,
    ) -> RequestResult:
        """
        Process a single request (KV cache already acquired by main thread).

        Args:
            entry: The workload entry
            request_id: Request ID
            scheduled_time_ms: When the request was scheduled (after timestamp wait)
            wait_time_ms: Time spent waiting for KV cache
            vocab_size: Vocabulary size for request construction
            pages_used_at_alloc: KV pages used snapshot taken right after allocation
        """
        request = Request(
            input_length=entry.input_length,
            output_length=entry.output_length,
            vocab_size=vocab_size,
        )
        kv_size = entry.input_length + entry.output_length

        # Record actual send time when worker starts processing
        actual_send_timestamp_s = time.time()
        actual_send_time_ms = self._get_elapsed_ms()

        # Track in-flight count; use pages_used snapshot from main thread
        # to accurately reflect KV state at the moment this request was allocated.
        with self.lock:
            self.in_flight_count += 1
            in_flight = self.in_flight_count
            completed = self.completed_count
            pages_used = pages_used_at_alloc
            pages_total = self.client.kv_cache_size

        if self.on_request_sent:
            self.on_request_sent(
                request_id,
                entry.input_length,
                entry.output_length,
                in_flight,
                completed,
                self.total_requests,
                actual_send_time_ms,
                pages_used,
                pages_total,
            )

        if self.verbose:
            print(
                f"[Request {request_id}] Sending at {actual_send_time_ms:.1f}ms "
                f"(scheduled: {scheduled_time_ms:.1f}ms, waited: {wait_time_ms:.1f}ms)"
            )

        try:
            # Send request using client
            result = self.client.send_request(request, request_id)

            completion_timestamp_s = time.time()
            completion_time_ms = self._get_elapsed_ms()
            latency_ms = result["latency"] * 1000  # Convert to ms

            request_result = RequestResult(
                request_id=request_id,
                input_length=entry.input_length,
                output_length=entry.output_length,
                actual_output_length=result["actual_output_length"],
                output_token_ids_sha256=result["output_token_ids_sha256"],
                scheduled_time_ms=scheduled_time_ms,
                actual_send_time_ms=actual_send_time_ms,
                actual_send_timestamp_s=actual_send_timestamp_s,
                completion_time_ms=completion_time_ms,
                completion_timestamp_s=completion_timestamp_s,
                latency_ms=latency_ms,
                wait_time_ms=wait_time_ms,
                success=result["success"],
                error=result.get("error"),
            )
        except Exception as e:
            completion_timestamp_s = time.time()
            completion_time_ms = self._get_elapsed_ms()
            request_result = RequestResult(
                request_id=request_id,
                input_length=entry.input_length,
                output_length=entry.output_length,
                actual_output_length=0,
                output_token_ids_sha256=None,
                scheduled_time_ms=scheduled_time_ms,
                actual_send_time_ms=actual_send_time_ms,
                actual_send_timestamp_s=actual_send_timestamp_s,
                completion_time_ms=completion_time_ms,
                completion_timestamp_s=completion_timestamp_s,
                latency_ms=completion_time_ms - actual_send_time_ms,
                wait_time_ms=wait_time_ms,
                success=False,
                error=str(e),
            )
        finally:
            self._release_kv_cache(kv_size)

        # Track completion and invoke callback
        with self.lock:
            self.completed_count += 1
            self.in_flight_count -= 1
            in_flight = self.in_flight_count
            completed = self.completed_count
            pages_used = self.current_kv_usage
            pages_total = self.client.kv_cache_size

        if self.on_request_complete:
            self.on_request_complete(
                request_result,
                in_flight,
                completed,
                self.total_requests,
                pages_used,
                pages_total,
            )

        return request_result

    def run(
        self, workload: list[WorkloadEntry], vocab_size: int
    ) -> list[RequestResult]:
        """
        Run the workload with timestamp-based scheduling.

        Args:
            workload: List of workload entries to process

        Returns:
            List of results for each request
        """
        if not workload:
            return []

        # Sort workload by timestamp, then by line number (for stable ordering)
        sorted_workload = sorted(workload, key=lambda e: (e.timestamp, e.line_number))

        # Initialize timing and tracking
        self.first_timestamp = sorted_workload[0].timestamp
        self.start_time = time.time()
        self.current_kv_usage = 0
        self.in_flight_count = 0
        self.completed_count = 0
        self.total_requests = len(sorted_workload)

        results: list[RequestResult | None] = [None] * len(sorted_workload)
        futures: dict[Future, int] = {}

        with ThreadPoolExecutor(max_workers=self.client.max_workers) as executor:
            for request_id, entry in enumerate(sorted_workload):
                # Wait for timestamp (if not ignoring)
                self._wait_for_timestamp(entry.timestamp)

                # Record scheduled time (after timestamp wait, before KV wait)
                scheduled_time_ms = self._get_elapsed_ms()

                # Wait for KV cache in main thread to ensure strict ordering
                kv_size = entry.input_length + entry.output_length
                wait_time_ms = self._wait_for_kv_cache(kv_size)

                # Capture pages_used immediately after allocation (before next
                # iteration changes it) so the SEND log reflects the true state
                # at the moment this request's KV was allocated.
                with self.lock:
                    pages_used_snapshot = self.current_kv_usage

                # Submit request to worker thread (KV cache already acquired)
                future = executor.submit(
                    self._process_request,
                    entry,
                    request_id,
                    scheduled_time_ms,
                    wait_time_ms,
                    vocab_size,
                    pages_used_snapshot,
                )
                futures[future] = request_id

            # Collect results
            for future in futures:
                request_id = futures[future]
                try:
                    results[request_id] = future.result()
                except Exception as e:
                    print(f"Error collecting result for request {request_id}: {e}")

        return [r for r in results if r is not None]


def compute_statistics(results: list[RequestResult]) -> dict:
    """Compute aggregate statistics from results."""
    if not results:
        return {}

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    total_time_ms = max(r.completion_time_ms for r in results) if results else 0

    stats = {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_time_ms": total_time_ms,
        "throughput_req_per_sec": (
            len(results) / (total_time_ms / 1000) if total_time_ms > 0 else 0
        ),
    }

    if successful:
        latencies = [r.latency_ms for r in successful]
        wait_times = [r.wait_time_ms for r in successful]

        stats.update(
            {
                "latency_avg_ms": sum(latencies) / len(latencies),
                "latency_min_ms": min(latencies),
                "latency_max_ms": max(latencies),
                "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
                "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
                "wait_time_avg_ms": sum(wait_times) / len(wait_times),
                "wait_time_max_ms": max(wait_times),
            }
        )

    return stats


def print_summary(stats: dict) -> None:
    """Print a summary of the workload results."""
    print("\n" + "=" * 60)
    print("WORKLOAD SUMMARY")
    print("=" * 60)
    print(f"Total requests:     {stats.get('total_requests', 0)}")
    print(f"Successful:         {stats.get('successful_requests', 0)}")
    print(f"Failed:             {stats.get('failed_requests', 0)}")
    print(f"Total time:         {stats.get('total_time_ms', 0):.2f} ms")
    print(f"Throughput:         {stats.get('throughput_req_per_sec', 0):.2f} req/s")
    print("-" * 60)

    if stats.get("latency_avg_ms") is not None:
        print("LATENCY (ms)")
        print(f"  Average:          {stats['latency_avg_ms']:.2f}")
        print(f"  Min:              {stats['latency_min_ms']:.2f}")
        print(f"  Max:              {stats['latency_max_ms']:.2f}")
        print(f"  P50:              {stats['latency_p50_ms']:.2f}")
        print(f"  P99:              {stats['latency_p99_ms']:.2f}")
        print("-" * 60)
        print("KV CACHE WAIT TIME (ms)")
        print(f"  Average:          {stats['wait_time_avg_ms']:.2f}")
        print(f"  Max:              {stats['wait_time_max_ms']:.2f}")

    print("=" * 60 + "\n")


def save_results(
    output_path: str,
    results: list[RequestResult],
    stats: dict,
    config: dict,
) -> None:
    """Save results to a JSON file."""
    output = {
        "config": config,
        "statistics": stats,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference workload from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with vLLM client, 100k KV cache
  python benchmarks/pp_opt/client.py -w workload.csv -k 100000 \
    --model-name example-model

  # Run with vLLM client, ignore timestamps
  python benchmarks/pp_opt/client.py -w workload.csv -k 50000 \
    --ignore-timestamps --model-name example-model

  # Slow down by 2x
  python benchmarks/pp_opt/client.py -w workload.csv -k 100000 -r 2.0 \
    --model-name example-model
        """,
    )

    parser.add_argument(
        "-w",
        "--workload",
        type=str,
        required=True,
        help="Path to workload CSV file",
    )
    parser.add_argument(
        "-k",
        "--kv-cache-size",
        type=int,
        required=True,
        help="Maximum KV cache size in tokens",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the inference server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--ignore-timestamps",
        action="store_true",
        help="Ignore timestamps, send requests ASAP when KV cache is available",
    )
    parser.add_argument(
        "-r",
        "--time-ratio",
        type=float,
        default=1.0,
        help="Time ratio for timestamps: >1 slows down, <1 speeds up (default: 1.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <workload>_results.json)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Served model name for the vLLM request",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=256,
        help="Maximum number of concurrent requests (default: 256)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=7200.0,
        help="Per-request HTTP timeout in seconds (default: 7200)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path for real-time results (default: <workload>_run.log)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size for the model (default: 10000)",
    )
    parser.add_argument(
        "--min-output-length",
        type=int,
        default=1,
        help=(
            "Minimum output length; smaller values are clamped up to this (default: 1)"
        ),
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=None,
        help="Maximum output length; larger values are clamped down after filtering",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Filter out requests whose input + output length exceeds this value",
        required=True,
    )
    parser.add_argument(
        "--request-num",
        type=int,
        help=(
            "Maximum number of requests to send (applied after "
            "--max-model-len filtering)"
        ),
        required=True,
    )

    args = parser.parse_args()

    if args.max_output_len is not None and args.max_output_len < 1:
        parser.error("--max-output-len must be >= 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")

    # Validate workload file
    workload_path = Path(args.workload)
    if not workload_path.exists():
        print(f"Error: Workload file not found: {args.workload}", file=sys.stderr)
        sys.exit(1)

    # Set default output path
    if args.output is None:
        args.output = str(workload_path.stem) + "_results.json"

    # Set default log path
    if args.log is None:
        args.log = str(workload_path.stem) + "_run.log"

    # Set up logging
    logger = setup_logging(args.log, args.verbose)
    logger.info("Workload runner started - logging to %s", args.log)

    # Load workload
    print(f"Loading workload from: {args.workload}")
    workload = load_workload(args.workload)
    print(f"Loaded {len(workload)} requests")

    if not workload:
        print("Error: No valid requests in workload file", file=sys.stderr)
        sys.exit(1)

    # Clamp output_length to --min-output-length
    for e in workload:
        if e.output_length < args.min_output_length:
            e.output_length = args.min_output_length

    # Filter by max-model-len
    before = len(workload)
    workload = [
        e for e in workload if e.input_length + e.output_length <= args.max_model_len
    ]
    print(
        f"Filtered by max-model-len={args.max_model_len}: "
        f"{before} -> {len(workload)} requests"
    )

    # Limit by request-num (applied after filtering, but respect timestamp order)
    workload.sort(key=lambda e: (e.timestamp, e.line_number))
    before = len(workload)
    workload = workload[: args.request_num]
    print(
        f"Limited by request-num={args.request_num}: "
        f"{before} -> {len(workload)} requests"
    )

    # Clamp output_length to --max-output-len after filtering and limiting.
    if args.max_output_len is not None:
        capped = 0
        for e in workload:
            if e.output_length > args.max_output_len:
                e.output_length = args.max_output_len
                capped += 1
        print(
            f"Clamped by max-output-len={args.max_output_len}: "
            f"{capped} requests updated"
        )

    # Create client
    client = VLLMClient(
        base_url=args.base_url,
        kv_cache_size=args.kv_cache_size,
        model_name=args.model_name,
        max_workers=args.max_workers,
        request_timeout=args.request_timeout,
        verbose=args.verbose,
    )

    # Configuration for output
    config = {
        "workload_file": str(args.workload),
        "kv_cache_size": args.kv_cache_size,
        "base_url": args.base_url,
        "ignore_timestamps": args.ignore_timestamps,
        "time_ratio": args.time_ratio,
        "max_workers": args.max_workers,
        "request_timeout": args.request_timeout,
        "min_output_length": args.min_output_length,
        "max_output_len": args.max_output_len,
        "max_model_len": args.max_model_len,
        "request_num": args.request_num,
    }

    # Print configuration
    print("\nConfiguration:")
    print(f"  KV cache size:    {args.kv_cache_size:,}")
    print(f"  Base URL:         {args.base_url}")
    print(f"  Ignore timestamps: {args.ignore_timestamps}")
    print(f"  Time ratio:       {args.time_ratio}")
    print(f"  Max workers:      {args.max_workers}")
    print(f"  Min output len:   {args.min_output_length}")
    print(f"  Max output len:   {args.max_output_len}")
    print()

    # Create callbacks for real-time logging
    def log_request_sent(
        request_id: int,
        input_len: int,
        output_len: int,
        in_flight: int,
        completed: int,
        total: int,
        time_ms: float,
        pages_used: int,
        pages_total: int,
    ) -> None:
        pages_free = pages_total - pages_used
        msg = (
            f"[{request_id + 1:4d}/{total}] SEND | "
            f"in={input_len:5d} out={output_len:4d} | "
            f"fly/done/total={in_flight:3d}/{completed:3d}/{total:3d} | "
            f"pages={pages_used:6d}/{pages_total:6d} (free={pages_free:6d}) | "
            f"@{time_ms:8.1f}ms"
        )
        logger.info(msg)

    def log_request_complete(
        result: RequestResult,
        in_flight: int,
        completed: int,
        total: int,
        pages_used: int,
        pages_total: int,
    ) -> None:
        status = "OK" if result.success else "FAIL"
        pages_free = pages_total - pages_used
        msg = (
            f"[{result.request_id + 1:4d}/{total}] {status:4s} | "
            f"in={result.input_length:5d} out={result.output_length:4d} | "
            f"fly/done/total={in_flight:3d}/{completed:3d}/{total:3d} | "
            f"pages={pages_used:6d}/{pages_total:6d} (free={pages_free:6d}) | "
            f"latency={result.latency_ms:8.1f}ms wait={result.wait_time_ms:6.1f}ms | "
            f"@{result.completion_time_ms:8.1f}ms"
        )
        if not result.success and result.error:
            msg += f" | error: {result.error}"
        logger.info(msg)

    # Run workload
    runner = TimedWorkloadRunner(
        client=client,
        time_ratio=args.time_ratio,
        ignore_timestamps=args.ignore_timestamps,
        verbose=args.verbose,
        on_request_sent=log_request_sent,
        on_request_complete=log_request_complete,
    )

    print("Starting workload...")
    logger.info("=" * 70)
    logger.info("WORKLOAD EXECUTION LOG")
    logger.info("=" * 70)
    results = runner.run(workload, args.vocab_size)

    # Compute statistics
    stats = compute_statistics(results)

    # Log completion
    logger.info("=" * 70)
    logger.info(
        "COMPLETED: %d/%d successful",
        stats.get("successful_requests", 0),
        stats.get("total_requests", 0),
    )
    logger.info(
        "Total time: %.1fms | Throughput: %.2f req/s",
        stats.get("total_time_ms", 0),
        stats.get("throughput_req_per_sec", 0),
    )
    if stats.get("latency_avg_ms") is not None:
        logger.info(
            "Latency avg=%.1fms p50=%.1fms p99=%.1fms",
            stats["latency_avg_ms"],
            stats["latency_p50_ms"],
            stats["latency_p99_ms"],
        )
    logger.info("=" * 70)

    # Print summary
    print_summary(stats)

    # Save results
    save_results(args.output, results, stats, config)
    logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()
