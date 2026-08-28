# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/verify_ascend_benchmark_plugin.py"
)


def _write_checkout(plugin_repo: Path) -> str:
    (plugin_repo / "scripts").mkdir(parents=True)
    (plugin_repo / "setup.py").write_text(
        'entry_points={"vllm.platform_plugins": ["ascend = vllm_ascend:register"]}\n',
        encoding="utf-8",
    )
    for name in (
        "install_local_ascend_plugin.sh",
        "use_single_ascend_env.sh",
        "hust_ascend_manager_helper.sh",
    ):
        script = plugin_repo / "scripts" / name
        script.write_text("#!/bin/bash\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    subprocess.run(["git", "init", "-q", str(plugin_repo)], check=True)
    subprocess.run(["git", "-C", str(plugin_repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(plugin_repo),
            "-c",
            "user.name=ci",
            "-c",
            "user.email=ci@localhost",
            "commit",
            "-qm",
            "test checkout",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(plugin_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_checkout_preflight_validates_expected_immutable_sha(
    tmp_path: Path,
):
    plugin_repo = tmp_path / "plugin"
    expected_sha = _write_checkout(plugin_repo)
    output = tmp_path / "checkout.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "checkout",
            "--plugin-repo",
            str(plugin_repo),
            "--expected-sha",
            expected_sha,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "mode": "checkout",
        "plugin_sha": expected_sha,
        "status": "passed",
    }


def test_checkout_preflight_records_missing_bootstrap_contract(tmp_path: Path):
    plugin_repo = tmp_path / "plugin"
    plugin_repo.mkdir()
    output = tmp_path / "checkout.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "checkout",
            "--plugin-repo",
            str(plugin_repo),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "install_local_ascend_plugin.sh" in payload["reason"]


def test_installed_preflight_loads_workspace_modules_and_entry_point(
    tmp_path: Path,
):
    core_repo = tmp_path / "core"
    plugin_repo = tmp_path / "plugin"
    site_packages = tmp_path / "site-packages"
    (core_repo / "vllm").mkdir(parents=True)
    (plugin_repo / "vllm_ascend").mkdir(parents=True)
    dist_info = site_packages / "vllm_ascend_hust-1.0.dist-info"
    xxhash_dist_info = site_packages / "xxhash-3.5.0.dist-info"
    dist_info.mkdir(parents=True)
    xxhash_dist_info.mkdir(parents=True)
    (core_repo / "vllm/__init__.py").write_text("", encoding="utf-8")
    (plugin_repo / "vllm_ascend/__init__.py").write_text(
        "def register():\n    return 'ascend'\n", encoding="utf-8"
    )
    (site_packages / "xxhash.py").write_text("", encoding="utf-8")
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: vllm-ascend-hust\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[vllm.platform_plugins]\nascend = vllm_ascend:register\n",
        encoding="utf-8",
    )
    (xxhash_dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: xxhash\nVersion: 3.5.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "installed.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(core_repo), str(plugin_repo), str(site_packages))
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "installed",
            "--core-repo",
            str(core_repo),
            "--plugin-repo",
            str(plugin_repo),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["platform_entry_point"] == "vllm_ascend:register"
    assert payload["xxhash_version"] == "3.5.0"


def test_installed_preflight_rejects_wrong_callable_entry_point(tmp_path: Path):
    core_repo = tmp_path / "core"
    plugin_repo = tmp_path / "plugin"
    site_packages = tmp_path / "site-packages"
    (core_repo / "vllm").mkdir(parents=True)
    (plugin_repo / "vllm_ascend").mkdir(parents=True)
    dist_info = site_packages / "vllm_ascend_hust-1.0.dist-info"
    xxhash_dist_info = site_packages / "xxhash-3.5.0.dist-info"
    dist_info.mkdir(parents=True)
    xxhash_dist_info.mkdir(parents=True)
    (core_repo / "vllm/__init__.py").write_text("", encoding="utf-8")
    (plugin_repo / "vllm_ascend/__init__.py").write_text(
        "def wrong_register():\n    return 'ascend'\n", encoding="utf-8"
    )
    (site_packages / "xxhash.py").write_text("", encoding="utf-8")
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: vllm-ascend-hust\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[vllm.platform_plugins]\nascend = vllm_ascend:wrong_register\n",
        encoding="utf-8",
    )
    (xxhash_dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: xxhash\nVersion: 3.5.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "installed.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(core_repo), str(plugin_repo), str(site_packages))
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "installed",
            "--core-repo",
            str(core_repo),
            "--plugin-repo",
            str(plugin_repo),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "unexpected Ascend platform entry point" in payload["reason"]
