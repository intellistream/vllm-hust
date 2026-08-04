# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess

import pytest

from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.serve.utils import api_utils
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    sanitize_message,
    should_include_usage,
)


def test_sanitize_message():
    assert (
        sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        == "<_io.BytesIO object>"
    )


@pytest.mark.parametrize(
    ("stream_options", "expected"),
    [
        (None, (True, True)),
        (StreamOptions(include_usage=False), (True, True)),
        (
            StreamOptions(include_usage=False, continuous_usage_stats=False),
            (True, True),
        ),
        (
            StreamOptions(include_usage=True, continuous_usage_stats=False),
            (True, True),
        ),
    ],
)
def test_should_include_usage_force_enables_continuous_usage(stream_options, expected):
    assert should_include_usage(stream_options, True) == expected


def test_ascend_torch_preflight_timeout_defaults_to_60_seconds(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_TORCH_PREFLIGHT_TIMEOUT_S", raising=False)

    assert api_utils._ascend_torch_preflight_timeout_s() == 60


@pytest.mark.parametrize("timeout_s", ["1", "60", "300"])
def test_ascend_torch_preflight_timeout_accepts_bounded_values(
    monkeypatch, timeout_s
):
    monkeypatch.setenv("VLLM_ASCEND_TORCH_PREFLIGHT_TIMEOUT_S", timeout_s)

    assert api_utils._ascend_torch_preflight_timeout_s() == int(timeout_s)


@pytest.mark.parametrize("timeout_s", ["invalid", "0", "301"])
def test_ascend_torch_preflight_timeout_rejects_invalid_values(
    monkeypatch, timeout_s
):
    monkeypatch.setenv("VLLM_ASCEND_TORCH_PREFLIGHT_TIMEOUT_S", timeout_s)

    with pytest.raises(SystemExit, match="must be"):
        api_utils._ascend_torch_preflight_timeout_s()


def test_ascend_torch_preflight_uses_configured_timeout(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setenv("VLLM_ASCEND_TORCH_PREFLIGHT_TIMEOUT_S", "75")
    monkeypatch.setattr(api_utils.subprocess, "run", fake_run)

    api_utils._run_ascend_torch_preflight()

    assert observed["timeout"] == 75
    assert "torch.npu.is_available()" in observed["command"][2]
    assert "torch.zeros(1, device=device)" in observed["command"][2]


def test_ascend_torch_preflight_timeout_remains_fail_closed(monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setenv("VLLM_ASCEND_TORCH_PREFLIGHT_TIMEOUT_S", "75")
    monkeypatch.setattr(api_utils.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="engine startup remains blocked"):
        api_utils._run_ascend_torch_preflight()


class TestGetMaxTokens:
    """Tests for get_max_tokens() to ensure generation_config's max_tokens
    acts as a default when from model author, and as a ceiling when
    explicitly set by the user."""

    def test_default_sampling_params_used_when_no_request_max_tokens(self):
        """When user doesn't specify max_tokens, generation_config default
        should apply."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2048

    def test_request_max_tokens_not_capped_by_default_sampling_params(self):
        """When user specifies max_tokens in request, model author's
        generation_config max_tokens must NOT cap it (fixes #34005)."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 5000

    def test_override_max_tokens_caps_request(self):
        """When user explicitly sets max_tokens, it acts as a ceiling."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_override_max_tokens_used_as_default(self):
        """When no request max_tokens, override still applies as default."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=None,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
            override_max_tokens=2048,
        )
        assert result == 2048

    def test_max_model_len_still_caps_output(self):
        """max_model_len - input_length is always the hard ceiling."""
        result = get_max_tokens(
            max_model_len=3000,
            max_tokens=5000,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 2900  # 3000 - 100

    def test_request_max_tokens_smaller_than_default(self):
        """When user explicitly requests fewer tokens than gen_config default,
        that should be respected."""
        result = get_max_tokens(
            max_model_len=24000,
            max_tokens=512,
            input_length=100,
            default_sampling_params={"max_tokens": 2048},
        )
        assert result == 512

    def test_input_length_exceeds_max_model_len(self):
        with pytest.raises(
            ValueError,
            match="Input length .* exceeds model's maximum context length .*",
        ):
            get_max_tokens(
                max_model_len=100,
                max_tokens=50,
                input_length=150,
                default_sampling_params={"max_tokens": 2048},
            )
