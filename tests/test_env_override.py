# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for generic environment override helpers in vllm.env_override."""

import os
from unittest.mock import patch

import pytest

import vllm.env_override as env_override

pytestmark = pytest.mark.skip_global_cleanup


class TestAscendRuntimePath:
    """Ascend runtime path injection should cover driver-specific layouts."""

    def test_includes_driver_hal_subdirectory(self, monkeypatch, tmp_path):
        """driver/lib64/driver is required for libascend_hal.so on some hosts."""
        ascend_home = tmp_path / "ascend-toolkit"
        for relative_dir in [
            "lib64",
            "runtime/lib64",
            "compiler/lib64",
            "aarch64-linux/lib64",
            "hccl/lib64",
            "fwkacllib/lib64",
            "atc/lib64",
            "bin",
        ]:
            (ascend_home / relative_dir).mkdir(parents=True, exist_ok=True)

        driver_lib64 = "/usr/local/Ascend/driver/lib64"
        driver_driver_lib64 = "/usr/local/Ascend/driver/lib64/driver"
        driver_tools = "/usr/local/Ascend/driver/tools"
        real_isdir = os.path.isdir

        monkeypatch.setenv("VLLM_ASCEND_AUTO_ENV", "1")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
        monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
        monkeypatch.setattr(env_override, "_REEXEC_NEEDED", False)

        with (
            patch(
                "vllm.env_override._detect_ascend_home", return_value=str(ascend_home)
            ),
            patch(
                "vllm.env_override.os.path.isdir",
                side_effect=lambda path: path
                in {
                    driver_lib64,
                    driver_driver_lib64,
                    driver_tools,
                }
                or real_isdir(path),
            ),
        ):
            env_override._maybe_set_ascend_runtime_path()

        ld_library_path_parts = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
        assert driver_lib64 in ld_library_path_parts
        assert driver_driver_lib64 in ld_library_path_parts
        assert ld_library_path_parts.index(driver_lib64) < ld_library_path_parts.index(
            driver_driver_lib64
        )
        assert ld_library_path_parts[-1] == "/usr/lib"


def test_should_reexec_for_loader_env_only_for_controlled_entrypoints(monkeypatch):
    monkeypatch.setattr(env_override, "_REEXEC_NEEDED", True)
    monkeypatch.delenv("_VLLM_INTERNAL_ENV_REEXEC_DONE", raising=False)
    monkeypatch.delenv("VLLM_ALLOW_ENV_REEXEC", raising=False)

    assert env_override._should_reexec_for_loader_env(["/usr/bin/vllm"])
    assert env_override._should_reexec_for_loader_env(["/usr/bin/vllm-hust"])
    assert env_override._should_reexec_for_loader_env(
        ["/tmp/vllm/entrypoints/cli/main.py"]
    )
    assert not env_override._should_reexec_for_loader_env(["pytest"])


def test_should_reexec_for_loader_env_honors_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(env_override, "_REEXEC_NEEDED", True)
    monkeypatch.delenv("_VLLM_INTERNAL_ENV_REEXEC_DONE", raising=False)
    monkeypatch.setenv("VLLM_ALLOW_ENV_REEXEC", "1")

    assert env_override._should_reexec_for_loader_env(["python"])
