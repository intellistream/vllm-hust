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


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def runner_script_text() -> str:
    return RUNNER_SCRIPT_PATH.read_text(encoding="utf-8")


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


def test_main_push_runs_only_the_central_perfgate_producer():
    text = workflow_text()
    standard_step = text[
        text.index(
            "      - name: Run benchmark CI and optional formal publish"
        ) : text.index("      - name: Run main perfgate baseline producer")
    ]
    producer_step = text[
        text.index("      - name: Run main perfgate baseline producer") : text.index(
            "      - name: Publish main perfgate baseline"
        )
    ]
    publish_step = text[
        text.index("      - name: Publish main perfgate baseline") : text.index(
            "      - name: Performance gate - Stage 1 comparison"
        )
    ]

    assert "push:" in text[: text.index("jobs:")]
    assert "branches: [main]" in text[: text.index("jobs:")]
    assert "if: ${{ github.event_name != 'push' }}" in standard_step
    assert "github.event_name == 'push'" in producer_step
    assert "Qwen/Qwen2.5-3B-Instruct" in producer_step
    assert "BENCH_SCENARIO=random-online" in producer_step
    assert "SAME_SPEC_FORCE_PR_PREVIEW_COMPAT=1" in producer_step
    assert "vllm_hust_benchmark.perfgate_specs resolve" in producer_step
    assert "HUST_ASCEND_RUNTIME_VERSION" in producer_step
    assert 'if [[ "$rc" -eq 86' in producer_step
    assert 'if [[ "$rc" -eq 87' not in producer_step
    assert "publish_central_perfgate_baseline.py" in publish_step
    assert '--central-remote "$CENTRAL_BASELINE_REMOTE"' in publish_step
    assert '--branch "$CENTRAL_BASELINE_BRANCH"' in publish_step
    assert '--cann-version "$CENTRAL_BASELINE_CANN_VERSION"' in publish_step
    assert "--update-latest-pointer" in publish_step
    assert "VLLM_HUST_CENTRAL_BASELINE_WRITER_TOKEN" in publish_step
    assert text.count("secrets.VLLM_HUST_CENTRAL_BASELINE_WRITER_TOKEN") == 1
    assert "GIT_ASKPASS" in publish_step
    assert "CENTRAL_BASELINE_WRITER_TOKEN" not in producer_step
    assert "${CANN_VERSION" not in producer_step
    assert "store-main-perfgate-baseline:" not in text
    assert "perfgate_store_baseline.sh" not in text


def test_main_producer_uses_public_pinned_runtime_dependencies():
    text = workflow_text()

    assert "feature/perfgate-two-stage" not in text
    assert "ce380ceb67bd904956d8ebb807a7691234022796" in text
    assert "328ab674dd6889b14441423b182b7fb8fe160fdb" in text
    assert "2a113e514fb4b5678483cb7efaa7fa56140f60d3" in text
    assert "VLLM_ASCEND_HUST_REPO_REF" in text
    assert "HUST_ASCEND_MANAGER_REPO_REF" in text
    assert "CENTRAL_BASELINE_REMOTE: https://github.com" in text


def test_perfgate_consumer_uses_central_exact_baseline_and_pinned_dependencies():
    text = workflow_text()
    fetch_script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/perfgate_fetch_baseline.sh"
    ).read_text(encoding="utf-8")
    fetch_step = text[
        text.index(
            "      - name: Performance gate - fetch Stage 1 baseline"
        ) : text.index("      - name: Run benchmark CI and optional formal publish")
    ]

    assert "https://github.com/vLLM-HUST/vllm-hust-benchmark.git" in fetch_script
    assert '--branch "$CENTRAL_BASELINE_BRANCH"' in fetch_script
    assert "git remote get-url origin" not in fetch_script
    assert "CENTRAL_BASELINE_WRITER_TOKEN" not in fetch_script
    assert "perfgate_fetch_baseline.sh" in fetch_step
    assert "use_single_ascend_env.sh" in fetch_step
    assert 'PERFGATE_ALLOW_BASELINE_FALLBACK: "0"' in fetch_step
    assert "ce380ceb67bd904956d8ebb807a7691234022796" in text
    assert "328ab674dd6889b14441423b182b7fb8fe160fdb" in text
    assert "2a113e514fb4b5678483cb7efaa7fa56140f60d3" in text

    l3_step = text[
        text.index("      - name: L3 benchmark publication preflight") : text.index(
            "      - name: Deepen clone for fork-point detection"
        )
    ]
    website_step = text[
        text.index("      - name: Checkout website repo") : text.index(
            "      - name: Checkout vllm-ascend-hust repo"
        )
    ]
    assert "if: ${{ github.event_name != 'push' }}" in l3_step
    assert "if: ${{ github.event_name != 'push' }}" in website_step


def test_checkout_urls_are_public_and_read_only():
    text = workflow_text()

    assert "TARGET_REPO_URL: https://github.com/${{ github.repository }}.git" in text
    assert "BENCHMARK_REPO_URL: https://github.com" in text
    assert "WEBSITE_REPO_URL: https://github.com" in text
    assert "VLLM_ASCEND_HUST_REPO_URL: https://github.com" in text
    assert "HUST_ASCEND_MANAGER_REPO_URL: https://github.com" in text
    assert "https://github.com/vLLM-HUST/vllm-hust-benchmark.git" in text
    assert "https://github.com/vLLM-HUST/vllm-ascend-hust.git" in text
    assert "git@github.com" not in text
    assert "BENCHMARK_REPO_SSH_KEY" not in text


def test_external_actions_use_immutable_commit_shas():
    text = workflow_text()

    action_references = [
        line.strip().removeprefix("uses: ").split(" #", maxsplit=1)[0]
        for line in text.splitlines()
        if line.strip().startswith("uses: actions/")
    ]
    assert action_references
    for reference in action_references:
        _action, separator, revision = reference.rpartition("@")
        assert separator == "@"
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)


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
    assert "VLLM_HUST_SCHEDULE_PUBLISH_BENCHMARK == '1'" in text
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

    same_spec_marker = 'if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'
    assert same_spec_marker in script
    same_spec_block = script[
        script.index(same_spec_marker) : script.index(
            "else", script.index(same_spec_marker)
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
    assert text.count("secrets.VLLM_HUST_BENCHMARK_GH_TOKEN") == 1
    assert "secrets.VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" not in text


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
