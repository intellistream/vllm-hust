# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/ascend-benchmark-leaderboard.yml"
)
RUNNER_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/run_ascend_benchmark_scenario_list.sh"
)
BENCHMARK_RUNNER_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/run_ascend_benchmark_ci.sh"
)


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def runner_script_text() -> str:
    return RUNNER_SCRIPT_PATH.read_text(encoding="utf-8")


def benchmark_runner_script_text() -> str:
    return BENCHMARK_RUNNER_SCRIPT_PATH.read_text(encoding="utf-8")


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


def test_pull_request_trigger_keeps_new_commit_and_reopen_events() -> None:
    text = workflow_text()
    trigger = text[text.index("on:") : text.index("  issue_comment:")]

    assert "types: [labeled, synchronize, reopened]" in trigger


def test_main_push_trigger_and_pr_head_checkout_are_preserved() -> None:
    text = workflow_text()
    trigger = text[text.index("on:") : text.index("  pull_request:")]
    benchmark_job_start = text.index("  ascend-benchmark:")
    benchmark_job = text[
        benchmark_job_start : text.index("    permissions:", benchmark_job_start)
    ]

    assert "push:" in trigger
    assert "branches: [main]" in trigger
    assert "github.event_name == 'push'" in benchmark_job
    assert (
        "TARGET_REPO_SHA: ${{ github.event_name == 'pull_request' && "
        "github.event.pull_request.head.sha || github.event_name == "
        "'issue_comment' && needs.issue-comment-command.outputs.pr_head_sha || "
        "github.sha }}" in text
    )


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


def test_main_baseline_publication_stays_in_trusted_benchmark_job():
    text = workflow_text()
    publish_step = text.index("- name: Publish central perfgate baseline")
    formal_step = text.index("- name: Run benchmark CI and optional formal publish")
    publish_block = text[publish_step:formal_step]

    assert "  store-main-perfgate-baseline:" not in text
    assert "runs-on: ubuntu-latest" not in publish_block
    assert (
        "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}-perfgate"
    ) in publish_block
    assert (
        "RESULT_ROOT: ${{ github.workspace }}/.benchmarks/ci/ci-"
        "${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}-perfgate"
    ) in publish_block
    assert "PERFGATE_BENCHMARK_REPO_DIR:" in publish_block
    assert "PERFGATE_TARGET_GIT_REPOSITORY:" in publish_block
    assert "VLLM_HUST_CENTRAL_BASELINE_WRITER_TOKEN" in publish_block
    assert "perfgate_store_baseline.sh" in publish_block


def test_benchmark_repo_default_ref_is_main():
    text = workflow_text()

    assert "feature/perfgate-two-stage" not in text
    assert "vars.VLLM_HUST_BENCHMARK_REPO_REF || 'main'" in text
    assert "vars.VLLM_HUST_PERFGATE_BENCHMARK_RUNNER_SHA" in text


def test_pr_and_manual_checkout_urls_use_https_without_publish_ssh_key():
    text = workflow_text()

    assert "TARGET_REPO_URL: https://github.com/${{ github.repository }}.git" in text
    assert "https://github.com/vLLM-HUST/vllm-hust-benchmark.git" in text
    assert "https://github.com/vLLM-HUST/vllm-ascend-hust.git" in text
    assert "https://github.com/vLLM-HUST/ascend-runtime-manager.git" in text


def test_benchmark_checkout_retries_have_hard_network_timeouts():
    text = workflow_text()
    checkout_block = text[
        text.index("      - name: Checkout target repo with retry") : text.index(
            "      - name: Prepare Hugging Face cache directories"
        )
    ]

    assert "GIT_CHECKOUT_RETRY_ATTEMPTS:-3" in checkout_block
    assert "GIT_CHECKOUT_TIMEOUT_SECONDS:-90" in checkout_block
    assert checkout_block.count('timeout --foreground "${timeout_seconds}s"') >= 6


def test_benchmark_disables_hugging_face_xet_downloads():
    text = workflow_text()

    assert 'HF_HUB_DISABLE_XET: "1"' in text


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


def test_trusted_main_requires_pinned_public_companion_checkouts():
    text = workflow_text()

    assert "TARGET_REPO_URL: https://github.com/${{ github.repository }}.git" in text
    assert (
        "BENCHMARK_REPO_URL: "
        "https://github.com/vLLM-HUST/vllm-hust-benchmark.git" in text
    )
    assert (
        "VLLM_ASCEND_HUST_REPO_URL: "
        "https://github.com/vLLM-HUST/vllm-ascend-hust.git" in text
    )
    assert (
        "HUST_ASCEND_MANAGER_REPO_URL: "
        "https://github.com/vLLM-HUST/ascend-runtime-manager.git" in text
    )
    assert "- name: Resolve trusted main dependency SHAs" in text
    for name in (
        "PERFGATE_BENCHMARK_RUNNER_SHA",
        "PERFGATE_VLLM_ASCEND_HUST_SHA",
        "PERFGATE_RUNTIME_MANAGER_SHA",
    ):
        assert name in text
    assert "must be configured as a full immutable 40-character Git SHA" in text
    assert 'echo "BENCHMARK_REPO_REF=$PERFGATE_BENCHMARK_RUNNER_SHA"' in text
    assert 'echo "VLLM_ASCEND_HUST_REPO_REF=$PERFGATE_VLLM_ASCEND_HUST_SHA"' in text
    assert 'echo "HUST_ASCEND_MANAGER_REPO_REF=$PERFGATE_RUNTIME_MANAGER_SHA"' in text
    assert "- name: Verify trusted main dependency SHAs" in text
    assert 'rev-parse HEAD)" = "$PERFGATE_BENCHMARK_RUNNER_SHA"' in text
    assert 'rev-parse HEAD)" = "$PERFGATE_VLLM_ASCEND_HUST_SHA"' in text
    assert 'rev-parse HEAD)" = "$PERFGATE_RUNTIME_MANAGER_SHA"' in text


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
        "HARDWARE_CHIP_MODEL: ${{ "
        "vars.VLLM_HUST_PERFGATE_HARDWARE_CHIP_MODEL || '910B2' }}" in text
    )
    assert "Resolve perfgate spec for Ascend runner" in text
    assert "Resolve main same-spec file" in text
    assert "resolve_perfgate_spec_file.py" in text
    assert 'echo "SAME_SPEC_SPEC_FILE=$resolved_spec_file" >> "$GITHUB_ENV"' in text
    assert "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json" in text


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


def test_multi_scenario_runner_prints_failure_diagnostics():
    text = runner_script_text()

    assert "scenario.log" in text
    assert ') 2>&1 | tee "$scenario_log"' in text
    assert "scenario_exit_code=${PIPESTATUS[0]}" in text
    assert "print_multi_scenario_summary" in text
    assert "Failed Ascend benchmark scenario diagnostics" in text
    assert 'print_file_tail "scenario output"' in text
    assert 'print_file_tail "vLLM server log"' in text
    assert 'print_file_tail "runner preflight failure"' in text


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
        script.index(
            'if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'
        ) : script.index(
            "else", script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then')
        )
    ]
    assert "EFFECTIVE_CONSTRAINTS_FILE=$SAME_SPEC_CONSTRAINTS_FILE" in same_spec_block
    assert "bench_args=()" in same_spec_block
    assert "Unsupported BENCH_SCENARIO without same-spec mode" in script
    assert (
        'if [[ "$BENCH_SCENARIO" == "random-online" && '
        '"$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then' not in script
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
        "TARGET_REPO_SHA: ${{ github.event_name == 'pull_request' && "
        "github.event.pull_request.head.sha || github.event_name == "
        "'issue_comment' && "
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


def test_pr_and_comment_runs_target_dedicated_npu_two():
    text = workflow_text()

    assert (
        "(github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && 'npu-2'"
    ) in text


def test_workflow_dispatch_targets_manual_runner_pool():
    text = workflow_text()

    assert (
        "github.event_name == 'workflow_dispatch' && "
        "(vars.VLLM_HUST_MANUAL_RUNNER_LABEL || 'npu-1')"
    ) in text


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

    assert "GIT_CHECKOUT_RETRY_ATTEMPTS:-3" in checkout_step
    assert "GIT_CHECKOUT_RETRY_DELAY_SECONDS:-30" in checkout_step
    assert "GIT_CHECKOUT_TIMEOUT_SECONDS:-90" in checkout_step
    assert 'timeout --foreground "${timeout_seconds}s"' in checkout_step
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
    assert "torch_npu probe failed" in text


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


STORE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/perfgate_store_baseline.sh"
)


def store_script_text() -> str:
    return STORE_SCRIPT_PATH.read_text(encoding="utf-8")


def test_producer_sets_perfgate_measurement_knobs():
    text = workflow_text()

    producer_step = text.index("- name: Run perfgate baseline producer benchmark")
    producer_block = text[
        producer_step : text.index(
            "- name: Upload perfgate producer artifact", producer_step
        )
    ]
    assert 'PERFGATE_WARMUP_RUNS: "1"' in producer_block
    assert 'PERFGATE_MEASURED_RUNS: "3"' in producer_block
    assert 'PERFGATE_AGGREGATION: "primary-median-run"' in producer_block
    assert (
        "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}-perfgate" in producer_block
    )
    assert "merge-base --is-ancestor HEAD origin/main" in producer_block


def test_store_baseline_step_expects_measurement_policy():
    text = workflow_text()

    store_step = text.index("- name: Publish central perfgate baseline")
    formal_step = text.index("- name: Run benchmark CI and optional formal publish")
    store_block = text[store_step:formal_step]

    assert 'PERFGATE_EXPECTED_WARMUP_RUNS: "1"' in store_block
    assert 'PERFGATE_EXPECTED_MEASURED_RUNS: "3"' in store_block
    assert 'PERFGATE_EXPECTED_SCHEMA_VERSION: "perfgate-measurement/v2"' in store_block
    assert 'PERFGATE_EXPECTED_STRATEGY: "warmup+primary-median-run"' in store_block
    assert 'PERFGATE_EXPECTED_AGGREGATION: "primary-median-run"' in store_block


def test_pr_stage1_and_stage2_emit_provenance_with_repeated_measurements():
    text = workflow_text()

    formal_step = text.index("- name: Run benchmark CI and optional formal publish")
    stage1_step = text.index("- name: Performance gate - Stage 1 comparison")
    stage2_step = text.index(
        "- name: Performance gate - Stage 2 trial rebase and benchmark"
    )
    final_compare_step = text.index(
        "- name: Performance gate - two-stage comparison", stage2_step
    )

    pr_block = text[formal_step:stage1_step]
    stage2_block = text[stage2_step:final_compare_step]

    assert "PERFGATE_WARMUP_RUNS:" in pr_block
    assert "PERFGATE_MEASURED_RUNS:" in pr_block
    assert 'PERFGATE_AGGREGATION: "primary-median-run"' in pr_block
    assert 'PERFGATE_WARMUP_RUNS: "1"' in stage2_block
    assert 'PERFGATE_MEASURED_RUNS: "3"' in stage2_block
    assert 'PERFGATE_AGGREGATION: "primary-median-run"' in stage2_block


def test_runner_script_forwards_perfgate_measurement_env():
    text = benchmark_runner_script_text()

    for name in (
        "PERFGATE_AGGREGATION",
        "PERFGATE_MEASURED_RUNS",
        "PERFGATE_WARMUP_RUNS",
    ):
        assert f"\n  {name}\n" in text, f"{name} missing from SUDO_PRESERVE_ENV_VARS"

    assert text.count('PERFGATE_WARMUP_RUNS="${PERFGATE_WARMUP_RUNS:-0}"') >= 2
    assert text.count('PERFGATE_MEASURED_RUNS="${PERFGATE_MEASURED_RUNS:-1}"') >= 2
    assert (
        text.count('PERFGATE_AGGREGATION="${PERFGATE_AGGREGATION:-primary-median-run}"')
        >= 2
    )


def test_runner_script_copies_measurement_json_fail_closed():
    text = benchmark_runner_script_text()

    assert 'cp "$same_spec_submission_dir/measurement.json"' in text
    assert "repeated-run measurement strategy is enabled" in text
    assert '"${PERFGATE_WARMUP_RUNS:-0}" -gt 0' in text
    assert '"${PERFGATE_MEASURED_RUNS:-1}" -gt 1' in text
    assert "perfgate-provenance.json" in text
    assert '"schema_version": "perfgate-runtime-provenance/v1"' in text
    assert 'PERFGATE_CANN_VERSION="${HUST_ASCEND_RUNTIME_VERSION:-}"' in text
    assert 'git -C "${HUST_ASCEND_MANAGER_REPO:-}" rev-parse HEAD' in text
    assert 'PERFGATE_RUNTIME_MANAGER_SHA="$runtime_manager_commit"' in text
    assert '"runtime_manager_sha": one_line' in text


def test_store_baseline_script_uses_central_exact_protocol_and_askpass():
    text = store_script_text()

    assert "vllm_hust_benchmark.perfgate_baselines publish" in text
    assert '--measurement-file "$MEASUREMENT_FILE"' in text
    assert '--target-repository "$TARGET_REPOSITORY"' in text
    assert '--spec-hash "$SPEC_HASH"' in text
    assert '--benchmark-runner-sha "$BENCHMARK_RUNNER_SHA"' in text
    assert '--runtime-manager-sha "$RUNTIME_MANAGER_SHA"' in text
    assert "perfgate-measurement/v2" in text
    assert "warmup+primary-median-run" in text
    assert 'secondary_sort_key == "run_index"' in text
    assert "UPDATE_POINTER=(--update-latest-pointer)" in text
    assert "GIT_ASKPASS" in text
    assert "GIT_TERMINAL_PROMPT=0" in text
    assert "x-access-token" in text
    assert "x-access-token:${" not in text
    assert "verify_raw_result_evidence warmup" in text
    assert "verify_raw_result_evidence per_run" in text
    assert 'sha256sum "$raw_result"' in text
    assert "baselines/$GITHUB_SHA" not in text
    assert "latest-main-pointer.json" not in text


def test_trusted_main_benchmark_runner_supports_runtime_manager_provenance() -> None:
    workflow = workflow_text()
    check_start = workflow.index("- name: Verify trusted main dependency SHAs")
    check_end = workflow.index(
        "- name: Prepare Hugging Face cache directories", check_start
    )
    dependency_check = workflow[check_start:check_end]

    assert "perfgate_baselines publish --help" in dependency_check
    assert "--runtime-manager-sha" in dependency_check
    assert "Pinned benchmark runner does not support" in dependency_check


def test_perfgate_producer_and_store_survive_formal_benchmark_failure():
    text = workflow_text()

    assert "id: perfgate-producer" in text
    assert "if: ${{ github.event_name == 'push'" in text
    assert "id: perfgate-producer\n        if: ${{ always()" not in text
    assert "continue-on-error: true" in text
    assert 'echo "status=failed" >> "$GITHUB_OUTPUT"' in text
    assert 'echo "status=success" >> "$GITHUB_OUTPUT"' in text
    assert (
        "name: ascend-perfgate-${{ github.run_id }}-${{ github.run_attempt }}" in text
    )
    assert text.index("- name: Upload perfgate producer artifact") < text.index(
        "- name: Sanitize runner before central baseline publication"
    )
    assert text.index(
        "- name: Sanitize runner before central baseline publication"
    ) < text.index("- name: Publish central perfgate baseline")
    assert text.index("- name: Publish central perfgate baseline") < text.index(
        "- name: Run benchmark CI and optional formal publish"
    )
    assert "  store-main-perfgate-baseline:" not in text
    assert text.count("secrets.VLLM_HUST_CENTRAL_BASELINE_WRITER_TOKEN") == 1
