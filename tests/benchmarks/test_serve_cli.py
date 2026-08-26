# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import urllib3

from vllm.benchmarks import serve as benchmark_serve

from ..utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

pytestmark = pytest.mark.skip_global_cleanup


def generate_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed certificate for testing."""
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Generate self-signed certificate using openssl
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_file, key_file


class RemoteOpenAIServerSSL(RemoteOpenAIServer):
    """RemoteOpenAIServer subclass that supports SSL with self-signed certs."""

    @property
    def url_root(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _wait_for_server(self, *, url: str, timeout: float):
        """Override to use HTTPS with SSL verification disabled."""
        # Suppress InsecureRequestWarning for self-signed certs
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        start = time.time()
        while True:
            try:
                if requests.get(url, verify=False).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None


@pytest.fixture(scope="function")
def server():
    args = ["--max-model-len", "1024", "--enforce-eager", "--load-format", "dummy"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="function")
def ssl_server():
    """Start a vLLM server with SSL enabled using a self-signed certificate."""
    with tempfile.TemporaryDirectory() as cert_dir:
        cert_file, key_file = generate_self_signed_cert(Path(cert_dir))
        args = [
            "--max-model-len",
            "1024",
            "--enforce-eager",
            "--load-format",
            "dummy",
            "--ssl-certfile",
            str(cert_file),
            "--ssl-keyfile",
            str(key_file),
        ]

        with RemoteOpenAIServerSSL(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.mark.benchmark
def test_bench_serve(server):
    # Test default model detection and input/output len
    command = [
        "vllm",
        "bench",
        "serve",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_insecure(ssl_server):
    """Test --insecure flag with an HTTPS server using a self-signed certificate."""
    base_url = f"https://{ssl_server.host}:{ssl_server.port}"
    command = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
        "--insecure",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_chat(server):
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.asyncio
async def test_resolve_benchmark_base_url_falls_back_to_workstation_gateway(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HUST_BASE_URL", "http://127.0.0.1:8080")

    args = SimpleNamespace(
        allow_local_benchmark_fallback=True,
        base_url=None,
        host="127.0.0.1",
        port=8000,
        endpoint="/v1/completions",
        backend="openai",
        insecure=False,
    )

    async def fake_get_first_model_from_server(base_url, headers, ssl_context):
        if base_url == "http://127.0.0.1:8000":
            raise RuntimeError("default target unreachable")
        assert base_url == "http://127.0.0.1:8080"
        assert ssl_context is None
        return ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct")

    monkeypatch.setattr(
        benchmark_serve,
        "get_first_model_from_server",
        fake_get_first_model_from_server,
    )

    (
        base_url,
        api_url,
        discovered_model,
    ) = await benchmark_serve.resolve_benchmark_base_url(
        args,
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/v1/completions",
        None,
    )

    assert base_url == "http://127.0.0.1:8080"
    assert api_url == "http://127.0.0.1:8080/v1/completions"
    assert discovered_model == (
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    )


@pytest.mark.asyncio
async def test_resolve_benchmark_base_url_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HUST_BASE_URL", "http://127.0.0.1:8080")

    args = SimpleNamespace(
        allow_local_benchmark_fallback=False,
        base_url=None,
        host="127.0.0.1",
        port=8000,
        endpoint="/v1/completions",
        backend="openai",
        insecure=False,
    )

    (
        base_url,
        api_url,
        discovered_model,
    ) = await benchmark_serve.resolve_benchmark_base_url(
        args,
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/v1/completions",
        None,
    )

    assert base_url == "http://127.0.0.1:8000"
    assert api_url == "http://127.0.0.1:8000/v1/completions"
    assert discovered_model is None


def test_initialize_benchmark_tokenizer_falls_back_to_cached_files(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizer_calls: list[str] = []

    def fake_get_tokenizer(tokenizer_id, tokenizer_mode, trust_remote_code):
        tokenizer_calls.append(tokenizer_id)
        if tokenizer_id == "Qwen/Qwen2.5-7B-Instruct":
            raise OSError("network unavailable")
        assert tokenizer_mode == "auto"
        assert trust_remote_code is False
        return object()

    monkeypatch.setattr(benchmark_serve, "get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr(
        benchmark_serve,
        "try_get_cached_tokenizer_dir",
        lambda tokenizer_id: "/tmp/qwen-tokenizer-snapshot",
    )

    tokenizer_id, tokenizer = benchmark_serve.initialize_benchmark_tokenizer(
        "Qwen/Qwen2.5-7B-Instruct",
        "auto",
        False,
    )

    assert tokenizer_id == "/tmp/qwen-tokenizer-snapshot"
    assert tokenizer is not None
    assert tokenizer_calls == [
        "Qwen/Qwen2.5-7B-Instruct",
        "/tmp/qwen-tokenizer-snapshot",
    ]


def test_resolve_preferred_tokenizer_id_uses_cache_for_local_default_target(
    monkeypatch: pytest.MonkeyPatch,
):
    args = SimpleNamespace(
        allow_local_benchmark_fallback=True,
        tokenizer=None,
        base_url=None,
        host="127.0.0.1",
        port=8000,
        endpoint="/v1/completions",
        backend="openai",
        insecure=False,
    )

    monkeypatch.setattr(
        benchmark_serve,
        "try_get_cached_tokenizer_dir",
        lambda tokenizer_id: "/tmp/qwen-tokenizer-snapshot",
    )

    tokenizer_id = benchmark_serve.resolve_preferred_tokenizer_id(
        args,
        "Qwen/Qwen2.5-7B-Instruct",
    )

    assert tokenizer_id == "/tmp/qwen-tokenizer-snapshot"


@pytest.mark.asyncio
async def test_main_async_rejects_dataset_path_for_builtin_random_dataset():
    args = SimpleNamespace(
        seed=0,
        ramp_up_strategy=None,
        label=None,
        base_url="http://127.0.0.1:8000",
        endpoint="/v1/completions",
        host="127.0.0.1",
        port=8000,
        header=None,
        allow_local_benchmark_fallback=False,
        insecure=False,
        model="Qwen/Qwen2.5-7B-Instruct",
        served_model_name="Qwen/Qwen2.5-7B-Instruct",
        skip_tokenizer_init=True,
        dataset_name="random",
        dataset_path="/tmp/requests.json",
        input_len=None,
        output_len=None,
    )

    with pytest.raises(
        ValueError, match="Cannot use 'random' dataset with --dataset-path"
    ):
        await benchmark_serve.main_async(args)
