# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/ascend-benchmark-leaderboard.yml"
)
SCRIPT_DIR = WORKFLOW_PATH.parent / "scripts"
PERFGATE_VALIDATE_REQUIRED_SCRIPT = SCRIPT_DIR / "perfgate_validate_required.sh"


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def workflow_yaml() -> dict:
    return yaml.safe_load(workflow_text())


def test_workflow_dispatch_input_count_stays_within_github_limit():
    inputs = workflow_yaml()[True]["workflow_dispatch"]["inputs"]

    assert len(inputs) <= 10
    assert "metadata_lengths" in inputs
    assert "input_length" not in inputs
    assert "output_length" not in inputs


def test_pr_comment_update_job_has_job_level_issues_write_permission():
    text = workflow_text()

    comment_step = text.index("      - name: Update PR benchmark comment")
    job_permissions = text.rindex("    permissions:", 0, comment_step)
    permissions_block = text[
        job_permissions : text.index("    runs-on:", job_permissions)
    ]

    assert "issues: write" in permissions_block
    assert "pull-requests: write" in permissions_block


def test_issue_comment_non_pr_commands_receive_denial_feedback():
    text = workflow_text()

    assert "if: ${{ github.event.comment.body != '' }}" in text
    assert '--event-payload "$GITHUB_EVENT_PATH"' in text
    assert "needs.issue-comment-command.outputs.deny_reason != ''" in text
    assert "persist-credentials: false" in text
    assert "const safeReason = reason.replace" in text
    assert "`Reason: ${safeReason}`" in text


def test_fork_pr_security_note_is_blocking():
    text = workflow_text()

    assert "Skipping Ascend benchmark on fork PRs" in text
    assert (
        "exit 1"
        in text[
            text.index("fork-pr-security-note:") : text.index("  ascend-benchmark:")
        ]
    )


def test_main_baseline_store_has_spec_file_and_benchmark_repo_checkout():
    text = workflow_text()
    store_job = text[text.index("  store-main-perfgate-baseline:") :]

    assert "TARGET_REPO_SHA: ${{ github.sha }}" in store_job
    assert (
        "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}"
    ) in store_job
    assert (
        "RESULT_ROOT: ${{ github.workspace }}/.benchmarks/ci/ci-"
        "${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}"
    ) in store_job
    assert "PERFGATE_SPEC_FILE:" not in store_job
    assert "MAIN_SAME_SPEC_SPEC_FILE:" not in store_job
    assert (
        "BENCHMARK_REPO_URL: https://github.com/vLLM-HUST/vllm-hust-benchmark.git"
        in store_job
    )
    assert "BENCHMARK_REPO_REF:" in store_job
    assert "Checkout benchmark repo" in store_job
    assert "git@github.com:vLLM-HUST/vllm-hust-benchmark.git" not in store_job
    assert (
        "vllm-hust-benchmark/${{ vars.VLLM_HUST_SAME_SPEC_SPEC_FILE || "
        "vars.VLLM_HUST_MAIN_SAME_SPEC_SPEC_FILE || "
        "'docs/official-baselines/official-ascend-jan-2026-v0180-random-online-"
        "qwen25-14b-910b2.json' }}"
    ) in store_job


def test_benchmark_repo_default_ref_is_main():
    text = workflow_text()

    assert "feature/perfgate-two-stage" not in text
    assert (
        "BENCHMARK_REPO_REF: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment' || github.event_name == "
        "'workflow_dispatch') && (vars.VLLM_HUST_BENCHMARK_REPO_REF || "
        "'main') || 'main' }}"
    ) in text


def test_pr_checkout_urls_use_https_without_publish_ssh_key():
    text = workflow_text()

    assert "format('https://github.com/{0}.git', github.repository)" in text
    assert "format('git@github.com:{0}.git', github.repository)" in text
    assert (
        "github.event_name == 'pull_request' || github.event_name == 'issue_comment'"
    ) in text
    assert "https://github.com/vLLM-HUST/vllm-hust-benchmark.git" in text
    assert "https://github.com/vLLM-HUST/vllm-ascend-hust.git" in text


def test_benchmark_install_removes_conflicting_vllm_provider():
    text = workflow_text()

    install_step = text[
        text.index(
            "      - name: Prepare Ascend runtime and install repos"
        ) : text.index("      - name: Verify installation")
    ]

    assert '"${PYTHON_BIN}" -m pip uninstall -y vllm vllm-hust' in install_step
    assert (
        '"${PYTHON_BIN}" scripts/ensure_vllm_provider.py --remove-conflicts'
        in install_step
    )
    assert '          "${PYTHON_BIN}" scripts/ensure_vllm_provider.py\n' in install_step
    assert (
        install_step.index(
            '"${PYTHON_BIN}" scripts/ensure_vllm_provider.py --remove-conflicts'
        )
        < install_step.index(
            '"${PYTHON_BIN}" -m pip install -e "${VLLM_HUST_BENCHMARK_REPO}[publish]"'
        )
        < install_step.index(
            '          "${PYTHON_BIN}" scripts/ensure_vllm_provider.py\n'
        )
    )


def test_main_benchmark_defaults_match_ascend_main_config():
    text = workflow_text()

    assert "default: Qwen/Qwen2.5-14B-Instruct" in text
    assert "BENCH_SCENARIOS:" in text
    assert "vars.VLLM_HUST_PR_BENCHMARK_SCENARIOS" in text
    assert "vars.VLLM_HUST_MAIN_BENCHMARK_SCENARIOS" in text
    assert "run_ascend_benchmark_scenario_list.sh" in text
    assert "steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'" in text
    assert "multi_scenario_results.tsv" in text
    assert "Perfgate comparison: `skipped for multi-scenario run" in text
    assert "vars.VLLM_HUST_MAIN_BENCHMARK_SCENARIOS == ''" in text
    assert (
        "github.event_name == 'pull_request' || github.event_name == 'issue_comment'"
    ) in text
    assert "&& '3B' || '14B'" in text
    assert "&& 'BF16' || 'FP16'" in text
    assert 'MAX_MODEL_LEN: ""' in text
    assert "&& '64' || '1024'" in text
    assert "&& '16' || '256'" in text
    assert "PERFGATE_SPEC_FILE: ${{ vars.VLLM_HUST_PERFGATE_SPEC_FILE || '' }}" in text
    assert "VLLM_HUST_PERFGATE_HARDWARE_CHIP_MODEL" in text
    assert (
        "HARDWARE_CHIP_MODEL: ${{ vars.VLLM_HUST_PERFGATE_HARDWARE_CHIP_MODEL || '910B2' }}"
        in text
    )
    assert "Resolve perfgate spec for Ascend runner" in text
    assert "Resolve main same-spec file" in text
    assert "resolve_perfgate_spec_file.py" in text
    assert 'echo "SAME_SPEC_SPEC_FILE=$resolved_spec_file" >> "$GITHUB_ENV"' in text
    assert "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json" in text


def test_required_pr_perfgate_is_enforced_and_validated():
    text = workflow_text()

    assert "github.event_name == 'pull_request' && 'enforce'" in text
    assert "Validate required PR perfgate scenario" in text
    assert "Validate required performance gate completion" in text
    assert 'PERFGATE_REQUIRED: "1"' in text
    assert "perfgate_validate_required.sh" in text
    assert "PERFGATE_BASELINE_UNAVAILABLE_REASON" in text
    assert "PERFGATE_STAGE2_NOT_RUN_REASON" in text
    assert "always() && (github.event_name == 'pull_request' || github.event_name == 'issue_comment')" in text
    assert "PERFGATE_STAGE2_REBASE_CONFLICT_FILE" in text


def test_required_perfgate_scripts_fail_fast():
    stage1_script = (SCRIPT_DIR / "perfgate_stage1_compare.sh").read_text(
        encoding="utf-8"
    )
    stage2_script = (
        SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh"
    ).read_text(encoding="utf-8")

    assert 'write_env PERFGATE_STAGE1_COMPLETED 1' in stage1_script
    assert '"$MODE" == "enforce"' in stage1_script
    assert '"$MODE" != "enforce"' in stage2_script
    assert 'write_env PERFGATE_STAGE2_EXECUTED 1' in stage2_script
    assert 'write_env PERFGATE_STAGE2_BASELINE_AVAILABLE "$stage2_baseline_available"' in stage2_script
    assert stage2_script.count('if [[ "$MODE" == "enforce" ]]') >= 2


def test_stage1_comparison_fails_only_in_enforce_mode(tmp_path: Path):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
report_file=""
while (( $# > 0 )); do
  if [[ "$1" == "--report-file" ]]; then
    report_file=$2
    break
  fi
  shift
done
printf '**Overall: FAIL**\n' > "$report_file"
exit 2
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    current.write_text("{}\n", encoding="utf-8")
    baseline.write_text("{}\n", encoding="utf-8")

    common_env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_BASELINE_FILE": str(baseline),
        "PERFGATE_STAGE1_CURRENT_FILE": str(current),
        "PERFGATE_REPORT_FILE": str(tmp_path / "report.md"),
        "GITHUB_ENV": str(tmp_path / "github-env"),
    }
    enforce_result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "perfgate_stage1_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={**common_env, "PERFGATE_MODE": "enforce"},
    )
    report_result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "perfgate_stage1_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={**common_env, "PERFGATE_MODE": "report"},
    )

    assert enforce_result.returncode == 2
    assert report_result.returncode == 0


def test_stage1_missing_baseline_fails_in_enforce_mode(tmp_path: Path):
    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "perfgate_stage1_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PERFGATE_MODE": "enforce",
            "PERFGATE_BASELINE_AVAILABLE": "0",
            "PERFGATE_REPORT_FILE": str(tmp_path / "report.md"),
            "GITHUB_ENV": str(tmp_path / "github-env"),
        },
    )

    assert result.returncode == 2
    assert "Stage 1 performance gate skipped" in result.stdout


def test_required_perfgate_validator_rejects_incomplete_gate():
    result = subprocess.run(
        ["bash", str(PERFGATE_VALIDATE_REQUIRED_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PERFGATE_REQUIRED": "1"},
    )

    assert result.returncode == 2
    assert "incomplete or failed" in result.stderr


def test_required_perfgate_validator_accepts_complete_gate(tmp_path: Path):
    stage1_baseline = tmp_path / "stage1-baseline.json"
    stage2_current = tmp_path / "stage2-current.json"
    stage2_baseline = tmp_path / "stage2-baseline.json"
    report = tmp_path / "perfgate-report.md"
    for path in (stage1_baseline, stage2_current, stage2_baseline, report):
        path.write_text("{}\n", encoding="utf-8")

    env = {
        **os.environ,
        "PERFGATE_REQUIRED": "1",
        "PERFGATE_MODE": "enforce",
        "BENCH_SCENARIO_COUNT": "1",
        "BENCH_SCENARIO": "random-online",
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_STAGE1_COMPLETED": "1",
        "PERFGATE_STAGE1_RESULT": "pass",
        "PERFGATE_STAGE2_EXECUTED": "1",
        "PERFGATE_STAGE2_BASELINE_AVAILABLE": "1",
        "PERFGATE_STAGE2_COMPLETED": "1",
        "PERFGATE_STAGE2_RESULT": "pass",
        "PERFGATE_STAGE2_SKIPPED": "0",
        "PERFGATE_STAGE2_REBASE_CONFLICT": "0",
        "PERFGATE_RESULT": "pass",
        "PERFGATE_BASELINE_FILE": str(stage1_baseline),
        "PERFGATE_STAGE2_B1PRIME_FILE": str(stage2_current),
        "PERFGATE_STAGE2_M2_BASELINE_FILE": str(stage2_baseline),
        "PERFGATE_REPORT_FILE": str(report),
    }
    result = subprocess.run(
        ["bash", str(PERFGATE_VALIDATE_REQUIRED_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "completed successfully" in result.stdout


def test_schedule_runs_registered_multi_scenario_benchmark_publish():
    text = workflow_text()
    workflow = workflow_yaml()[True]

    assert workflow["schedule"][0]["cron"] == "0 17 * * *"
    assert "github.event_name == 'schedule'" in text
    assert "VLLM_HUST_SCHEDULE_BENCHMARK_SCENARIOS" in text
    assert "VLLM_HUST_SCHEDULE_PUBLISH_BENCHMARK != '0'" in text
    for scenario in (
        "random-online",
        "sharegpt-online",
        "prefix-repetition-online",
        "random-latency",
        "sharegpt-throughput",
        "sonnet-throughput",
        "instructcoder-online",
        "agent-research-online",
        "visionarena-online",
    ):
        assert scenario in text


def test_benchmark_script_does_not_force_max_model_len():
    script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/run_ascend_benchmark_ci.sh"
    ).read_text(encoding="utf-8")

    assert "MAX_MODEL_LEN=${MAX_MODEL_LEN:-}" in script
    assert "max_model_len_args=()" in script
    assert '"${max_model_len_args[@]}"' in script
    assert script.count('"${max_model_len_args[@]}"') == 2


def test_benchmark_script_does_not_default_pr_hardware_to_b3():
    script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/run_ascend_benchmark_ci.sh"
    ).read_text(encoding="utf-8")

    assert "HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-910B2}" in script
    assert "HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-910B3}" not in script


def test_benchmark_runner_supports_registry_same_spec_scenarios():
    script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/run_ascend_benchmark_ci.sh"
    ).read_text(encoding="utf-8")

    assert 'if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then' in script
    same_spec_block = script[
        script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then') :
        script.index('else', script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'))
    ]
    assert "EFFECTIVE_CONSTRAINTS_FILE=$SAME_SPEC_CONSTRAINTS_FILE" in same_spec_block
    assert "bench_args=()" in same_spec_block
    assert 'Unsupported BENCH_SCENARIO without same-spec mode' in script
    assert (
        'if [[ "$BENCH_SCENARIO" == "random-online" && "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'
        not in script
    )


def test_ascend_benchmark_installs_no_build_isolation_build_dependencies():
    text = workflow_text()
    stage2_script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/perfgate_stage2_rebase_and_benchmark.sh"
    ).read_text(encoding="utf-8")

    assert "--no-build-isolation" in text
    assert '"setuptools-rust>=1.9.0"' in text
    assert "--no-build-isolation" in stage2_script
    assert '"setuptools-rust>=1.9.0"' in stage2_script


def test_perfgate_spec_resolver_uses_benchmark_registry():
    script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/resolve_perfgate_spec_file.py"
    ).read_text(encoding="utf-8")

    assert "vllm_hust_benchmark" in script
    assert "perfgate_specs.resolve_perfgate_spec_file" in script
    assert "perfgate-ascend-qwen25-3b-910b3.json" not in script


def test_issue_comment_uses_ubuntu_gate_before_self_hosted_runner():
    text = workflow_text()

    assert "issue_comment:" in text
    assert "issue-comment-command:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "needs: [issue-comment-command]" in text
    assert "needs.issue-comment-command.outputs.should_run == '1'" in text


def test_issue_comment_path_uses_pr_head_sha_and_base_sha():
    text = workflow_text()

    assert (
        "TARGET_REPO_SHA: ${{ github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.pr_head_sha || github.sha }}" in text
    )
    assert (
        "PR_HEAD_SHA: ${{ github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.pr_head_sha || "
        "github.event.pull_request.head.sha }}" in text
    )
    assert (
        "PR_BASE_SHA: ${{ github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.pr_base_sha || "
        "github.event.pull_request.base.sha }}" in text
    )


def test_benchmark_run_id_and_summary_use_target_repo_sha():
    text = workflow_text()

    assert (
        "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}" in text
    )
    assert (
        "target_repo_sha = os.environ.get('TARGET_REPO_SHA') or "
        "os.environ['GITHUB_SHA']" in text
    )
    assert (
        "const targetRepoSha = process.env.TARGET_REPO_SHA || process.env.GITHUB_SHA;"
        in text
    )
    assert (
        "ci-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}-"
        "${targetRepoSha}" in text
    )
    assert "f'- Commit: `{target_repo_sha}`'" in text


def test_issue_comment_path_keeps_publish_secrets_disabled():
    text = workflow_text()

    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish_to_hf) "
        "&& secrets.HF_TOKEN" in text
    )
    assert (
        "github.event_name != 'issue_comment') && "
        "secrets.VLLM_HUST_BENCHMARK_GH_TOKEN" in text
    )
    assert (
        "github.event_name != 'issue_comment') && "
        "secrets.VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" in text
    )


def test_pr_comment_update_has_issues_write_permission():
    text = workflow_text()

    assert "issues: write" in text
    assert "github.rest.issues.createComment" in text
    assert "github.rest.issues.updateComment" in text


def test_issue_comment_denial_feedback_is_posted_without_self_hosted_runner():
    text = workflow_text()

    assert (
        "deny_reason: ${{ steps.parse-command.outputs.ASCEND_COMMENT_DENY_REASON }}"
        in text
    )
    assert "issue-comment-denied:" in text
    assert "needs: [issue-comment-command]" in text
    assert "needs.issue-comment-command.outputs.should_run == '0'" in text
    assert "needs.issue-comment-command.outputs.deny_reason != ''" in text
    assert "runs-on: ubuntu-latest" in text
    assert "github.rest.issues.createComment" in text


def test_issue_comment_help_is_posted_without_self_hosted_runner():
    text = workflow_text()

    assert (
        "help_requested: ${{ steps.parse-command.outputs.ASCEND_COMMENT_HELP }}" in text
    )
    assert "issue-comment-help:" in text
    assert "needs.issue-comment-command.outputs.help_requested == '1'" in text
    assert "<!-- ascend-benchmark-command-help -->" in text
    assert "Supported same-repository PR preview commands:" in text
    assert "`/ascend smoke`" in text
    assert "`/ascend scenario random`" in text
    assert "`/ascend group smoke`" in text
    assert (
        "Comment-triggered runs are optional preview checks and are not "
        "required checks." in text
    )
    assert (
        "`/ascend official ...` is reserved for the future formal "
        "leaderboard path and is not supported yet." in text
    )


def test_workflow_dispatch_publish_inputs_are_split():
    text = workflow_text()

    assert "publish_to_benchmark_repo:" in text
    assert "description: Publish benchmark result to HF" in text
    assert (
        "description: Publish benchmark result to the benchmark repo and "
        "refresh leaderboard snapshots" in text
    )
    assert (
        "github.event_name == 'workflow_dispatch' && "
        "inputs.publish_to_benchmark_repo" in text
    )
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish_to_hf "
        "&& secrets.HF_TOKEN != ''" in text
    )
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish_to_hf)) "
        "&& '1' || '0'" not in text
    )


def test_workflow_dispatch_metadata_lengths_are_parsed_from_single_input():
    text = workflow_text()

    assert "metadata_lengths:" in text
    assert "BENCH_METADATA_LENGTHS:" in text
    assert "inputs.metadata_lengths" in text
    assert "Resolve workflow dispatch metadata lengths" in text
    assert "BENCH_INPUT_LEN=$input_len" in text
    assert "BENCH_OUTPUT_LEN=$output_len" in text
    assert "inputs.input_length" not in text
    assert "inputs.output_length" not in text


def test_l3_benchmark_publish_preflight_runs_before_benchmark():
    text = workflow_text()

    preflight_step = text.index("      - name: L3 benchmark publication preflight")
    checkout_step = text.index("      - name: Checkout target repo with retry")
    benchmark_repo_checkout_step = text.index("      - name: Checkout benchmark repo")
    benchmark_step = text.index(
        "      - name: Runner health preflight (before benchmark)"
    )
    summary_step = text.index("      - name: Build benchmark summary artifacts")

    assert "bash .github/workflows/scripts/l3_benchmark_publish_preflight.sh" in text
    assert checkout_step < preflight_step
    assert preflight_step < benchmark_repo_checkout_step
    assert preflight_step < benchmark_step
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT" in text[summary_step:]
    assert "L3_BENCHMARK_PUBLISH_TARGET" in text[summary_step:]
    assert "L3_BENCHMARK_PUBLISH_CREDENTIAL" in text[summary_step:]
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION" in text[summary_step:]
    assert "GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT" in text[summary_step:]


def test_target_checkout_uses_resilient_git_http_retry_settings():
    text = workflow_text()
    checkout_step = text[
        text.index("      - name: Checkout target repo with retry") : text.index(
            "      - name: L3 benchmark publication preflight"
        )
    ]

    assert "GIT_CHECKOUT_RETRY_ATTEMPTS:-6" in checkout_step
    assert "GIT_CHECKOUT_RETRY_DELAY_SECONDS:-30" in checkout_step
    assert "-c http.version=HTTP/1.1" in checkout_step
    assert "-c http.lowSpeedLimit=1024" in checkout_step
    assert "-c http.lowSpeedTime=30" in checkout_step


def test_ascend_torch_stack_is_installed_before_preinstall_preflight():
    text = workflow_text()
    ensure_script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/ensure_ascend_torch_stack.sh"
    ).read_text(encoding="utf-8")

    install_step = text.index("      - name: Install Ascend torch stack for preflight")
    preinstall_preflight_step = text.index(
        "      - name: Runner health preflight (before install)"
    )

    assert install_step < preinstall_preflight_step
    assert "bash .github/workflows/scripts/ensure_ascend_torch_stack.sh" in text
    assert 'ASCEND_TORCH_VERSION="${ASCEND_TORCH_VERSION:-2.10.0}"' in ensure_script
    assert (
        'ASCEND_TORCH_NPU_VERSION="${ASCEND_TORCH_NPU_VERSION:-2.10.0}"'
        in ensure_script
    )
    assert "import torch" in ensure_script
    assert "import torch_npu" in ensure_script


def test_l2_targeted_scenario_registry_is_covered_by_parser_tests():
    parser_script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/parse_ascend_comment_command.py"
    ).read_text(encoding="utf-8")
    parser_tests = (
        Path(__file__).resolve().parent / "test_parse_ascend_comment_command.py"
    ).read_text(encoding="utf-8")
    registry_tests = (
        Path(__file__).resolve().parent / "test_ascend_targeted_scenarios.py"
    ).read_text(encoding="utf-8")

    assert "load_targeted_scenario_registry" in parser_script
    assert "test_parse_group_command_maps_to_supported_group" in parser_tests
    assert "test_load_targeted_scenario_registry_from_repo_file" in registry_tests


def test_issue_comment_non_pr_and_fork_pr_are_not_allowed_by_parser_tests():
    parser_tests = (
        Path(__file__).resolve().parent / "test_parse_ascend_comment_command.py"
    ).read_text(encoding="utf-8")

    assert "test_resolve_issue_comment_pr_context_rejects_non_pr_issue" in parser_tests
    assert "test_resolve_issue_comment_pr_context_rejects_fork_pr" in parser_tests
