# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from vllm.entrypoints.cli import openai as openai_cli


def _make_connection_error(url: str) -> openai.APIConnectionError:
    request = httpx.Request("GET", url)
    return openai.APIConnectionError(request=request)


def test_interactive_cli_reports_default_server_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            list=Mock(
                side_effect=_make_connection_error(
                    f"{openai_cli.DEFAULT_OPENAI_API_URL}/models"
                )
            )
        )
    )
    monkeypatch.setattr(openai_cli, "OpenAI", Mock(return_value=fake_client))
    args = argparse.Namespace(
        api_key=None,
        model_name=None,
        url=openai_cli.DEFAULT_OPENAI_API_URL,
    )

    with pytest.raises(SystemExit) as exc_info:
        openai_cli._interactive_cli(args)

    message = str(exc_info.value)
    assert openai_cli.DEFAULT_OPENAI_API_URL in message
    assert "vllm serve <model>" in message
    assert "--model-name" in message


def test_chat_command_reports_request_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(
                    side_effect=_make_connection_error("http://example.com/v1/chat")
                )
            )
        )
    )
    monkeypatch.setattr(
        openai_cli,
        "_interactive_cli",
        Mock(return_value=("test-model", fake_client)),
    )
    args = argparse.Namespace(
        model_name="test-model",
        quick="hello",
        system_prompt=None,
        url="http://example.com/v1",
    )

    with pytest.raises(SystemExit) as exc_info:
        openai_cli.ChatCommand.cmd(args)

    message = str(exc_info.value)
    assert "http://example.com/v1" in message
    assert "--url" in message
    assert "--model-name" not in message
