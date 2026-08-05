# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
WORKFLOW = REPO_ROOT / ".github/workflows/ascend-benchmark-leaderboard.yml"


def script_text(name: str) -> str:
    return (SCRIPT_DIR / name).read_text(encoding="utf-8")


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_run_ascend_benchmark_propagates_benchmark_repo_publish_env():
    text = script_text("run_ascend_benchmark_ci.sh")

    assert "PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}" in text
    sudo_env_block = text[text.index("export_sudo_preserved_env_vars()") :]
    assert "PUBLISH_TO_BENCHMARK_REPO" in sudo_env_block
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then' in text
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" == "1" ]]; then' in text
    assert 'BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \\' in text
    assert 'BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \\' in text


def test_perfgate_store_baseline_cleans_scoped_writer_credentials_on_exit():
    text = script_text("perfgate_store_baseline.sh")

    assert "cleanup() {" in text
    assert "unset PERFGATE_BASELINE_WRITER_TOKEN WRITER_TOKEN" in text
    assert 'rm -rf "$ASKPASS_DIR"' in text
    assert "trap cleanup EXIT" in text


def test_benchmark_snapshot_sync_explains_missing_write_credentials():
    text = script_text("sync_benchmark_snapshots_to_github.sh")
    runner_text = script_text("run_ascend_benchmark_ci.sh")

    assert "L3 benchmark repository publication is enabled" in text
    assert "no cross-repository write credential is available" in text
    assert "VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" in text
    assert "VLLM_HUST_BENCHMARK_GH_TOKEN" in text
    assert "Benchmark repo publish target:" in text
    staging_index = text.index("publication_staging_dir=$(mktemp -d")
    public_validator_index = text.index("validate_public_leaderboard_snapshots.py")
    trend_validator_index = text.index("validate-trend --input")
    git_add_index = text.index('git -C "$BENCHMARK_REPO_DIR" add')
    git_commit_index = text.index('git -C "$BENCHMARK_REPO_DIR" commit')
    git_push_index = text.index('git -C "$BENCHMARK_REPO_DIR" push')
    assert staging_index < public_validator_index < trend_validator_index
    assert trend_validator_index < git_add_index
    assert git_add_index < git_commit_index < git_push_index
    assert "write_github_env GITHUB_SNAPSHOT_SYNC_STATUS rejected" in text
    assert (
        "required_submission_files=(" in text
        and "env-manifest.json" in text
        and "checksums.sha256" in text
    )
    assert "reset_publication_staging()" in text
    assert "reset_publication_staging || return $?" in text
    submit_index = runner_text.index('"$PYTHON_BIN" -m vllm_hust_benchmark.cli submit')
    finalize_index = runner_text.index("\nfinalize_submission_artifact\n", submit_index)
    sync_index = runner_text.index(
        "sync_benchmark_publication_to_github", finalize_index
    )
    assert submit_index < finalize_index < sync_index


def test_same_spec_benchmark_failure_prints_server_log_tail():
    text = script_text("run_ascend_benchmark_ci.sh")
    same_spec_block = text[text.index("run_same_spec_current_benchmark() {") :]

    assert "same_spec_server_log=$RESULT_ROOT/server.stdout.log" in same_spec_block
    assert "print_same_spec_server_log_tail() {" in same_spec_block
    assert "current same-spec vLLM server log tail" in same_spec_block
    assert 'collect_ascend_diagnostics "same-spec-current-failure"' in same_spec_block
    assert 'return "$same_spec_status"' in same_spec_block


def test_same_spec_pr_preview_uses_ascend_compatibility_overlay():
    text = script_text("run_ascend_benchmark_ci.sh")
    same_spec_block = text[text.index("run_same_spec_current_benchmark() {") :]

    assert "SAME_SPEC_PR_PREVIEW_COMPAT=${SAME_SPEC_PR_PREVIEW_COMPAT:-1}" in text
    assert "prepare_same_spec_pr_preview_compat_file() {" in same_spec_block
    assert 'server_parameters["no_enable_chunked_prefill"] = True' in same_spec_block
    assert 'server_parameters["no_enable_prefix_caching"] = True' in same_spec_block
    assert 'client_parameters.setdefault("temperature", 0)' in same_spec_block
    assert '${GITHUB_EVENT_NAME:-}" == "pull_request"' in same_spec_block
    assert '${GITHUB_EVENT_NAME:-}" == "issue_comment"' in same_spec_block
    assert '"$effective_same_spec_file"' in same_spec_block


def test_same_spec_runner_resolves_spec_from_shared_registry():
    text = script_text("run_ascend_benchmark_ci.sh")
    same_spec_block = text[text.index("run_same_spec_current_benchmark() {") :]

    assert "SAME_SPEC_SPEC_FILE=${SAME_SPEC_SPEC_FILE:-}" in text
    assert "vllm_hust_benchmark.perfgate_specs resolve" in same_spec_block
    assert '--scenario "$BENCH_SCENARIO"' in same_spec_block
    assert '--hardware-chip-model "$HARDWARE_CHIP_MODEL"' in same_spec_block
    assert '--repo-root "$VLLM_HUST_BENCHMARK_REPO"' in same_spec_block


def test_e2e_inference_scripts_use_python_http_probe_with_server_log():
    for script_name in (
        "run_e2e_serve_smoke.sh",
        "run_e2e_inference_regression.sh",
    ):
        text = script_text(script_name)

        assert "print_server_log_tail() {" in text
        assert "http_with_server_log() {" in text
        assert "e2e_http_request.py" in text
        assert '"$PYTHON_BIN" "$HTTP_REQUEST_SCRIPT"' in text
        assert "E2E_HTTP_REQUEST_ATTEMPTS" in text
        assert "E2E_HTTP_REQUEST_TIMEOUT_SECONDS" in text
        assert "else\n      rc=$?\n    fi" in text
        assert "failed after ${max_attempts} attempts" in text
        assert "curl -fsS" not in text
        assert "vLLM models endpoint readiness confirmation" in text
        assert (
            "http_with_server_log"
            in text[text.index("completion_response=$(mktemp)") :]
        )


def test_ascend_server_readiness_windows_allow_cold_start():
    for script_name in (
        "run_e2e_serve_smoke.sh",
        "run_e2e_inference_regression.sh",
    ):
        text = script_text(script_name)

        assert "SERVER_READY_MAX_ATTEMPTS=${SERVER_READY_MAX_ATTEMPTS:-300}" in text
        assert 'seq 1 "$SERVER_READY_MAX_ATTEMPTS"' in text
        assert '"$attempt" -eq "$SERVER_READY_MAX_ATTEMPTS"' in text

    benchmark_text = script_text("run_ascend_benchmark_ci.sh")
    assert (
        "SAME_SPEC_READY_TIMEOUT_SECONDS=${SAME_SPEC_READY_TIMEOUT_SECONDS:-1200}"
    ) in benchmark_text


def test_same_spec_readiness_timeout_reaches_runner_in_both_execution_modes():
    workflow = workflow_text()
    benchmark_text = script_text("run_ascend_benchmark_ci.sh")
    same_spec_block = benchmark_text[
        benchmark_text.index("run_same_spec_current_benchmark() {") :
    ]
    preserve_block = benchmark_text[
        benchmark_text.index("SUDO_PRESERVE_ENV_VARS=(") : benchmark_text.index(
            "build_sudo_env_preserve_list()"
        )
    ]

    assert 'SAME_SPEC_READY_TIMEOUT_SECONDS: "2400"' in workflow
    assert (
        "local same_spec_ready_timeout_seconds=${SAME_SPEC_READY_TIMEOUT_SECONDS:-1200}"
    ) in same_spec_block
    assert (
        same_spec_block.count(
            'READY_TIMEOUT_SECONDS="$same_spec_ready_timeout_seconds" \\'
        )
        == 2
    )
    assert "READY_TIMEOUT_SECONDS" in preserve_block
    assert (
        'echo "[same-spec-current] effective readiness timeout: '
        '${same_spec_ready_timeout_seconds}s"'
    ) in same_spec_block


def test_benchmark_pins_named_runner_to_its_npu():
    text = script_text("run_ascend_benchmark_ci.sh")

    assert '"${RUNNER_NAME:-}" =~ npu([0-9]+)$' in text
    assert 'runner_physical_device="${BASH_REMATCH[1]}"' in text
    assert "runner_devnodes=(/dev/davinci[0-9]*)" in text
    assert "export ASCEND_RT_VISIBLE_DEVICES=0" in text
    assert 'export ASCEND_RT_VISIBLE_DEVICES="$runner_physical_device"' in text


def test_benchmark_preserves_hugging_face_xet_setting_for_root_helper():
    text = script_text("run_ascend_benchmark_ci.sh")

    preserve_block = text[
        text.index("SUDO_PRESERVE_ENV_VARS=(") : text.index(
            "build_sudo_env_preserve_list()"
        )
    ]
    assert "HF_HUB_DISABLE_XET" in preserve_block


def test_perfgate_baseline_fetch_bounds_git_network_waits():
    text = script_text("perfgate_fetch_baseline.sh")

    assert "GIT_NETWORK_ATTEMPTS=${GIT_NETWORK_ATTEMPTS:-3}" in text
    assert "GIT_NETWORK_TIMEOUT_SECONDS=${GIT_NETWORK_TIMEOUT_SECONDS:-90}" in text
    assert 'timeout --foreground "${GIT_NETWORK_TIMEOUT_SECONDS}s"' in text
    assert "run_git_network ls-remote" in text
    assert "run_git_network clone" in text
    assert "CENTRAL_REPO_URL=${PERFGATE_CENTRAL_REPO_URL" in text
    assert "baseline-metadata.json" in text
    assert 'write_env PERFGATE_BASELINE_METADATA_FILE "$manifest"' in text
