# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Standard
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = (
    ROOT
    / "examples"
    / "disaggregated"
    / "mooncake_connector"
    / "run_mooncake_connector.sh"
)


def test_mooncake_example_stops_only_retained_child_process_groups() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pkill" not in source
    assert "kill -- -$$" not in source
    assert 'setsid "$@"' in source
    assert 'PIDS+=("$!")' in source
    assert 'kill -TERM -- "-${process_group}"' in source
    assert 'kill -KILL -- "-${process_group}"' in source


def test_all_service_launches_use_the_retained_child_helper() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'start_child "prefill$((i+1)).log"' in source
    assert 'start_child "decode$((i+1)).log"' in source
    assert 'start_child "proxy.log"' in source


def test_runner_requires_one_explicit_migration_mode_and_fresh_evidence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "legacy|typed|rollback" in source
    assert '[[ ! -e "$OUTPUT_DIR" ]]' in source
    assert '"provenance_label": "real-online-run"' in source
    assert '"seed": int(${BENCHMARK_SEED@Q})' in source


def test_typed_and_rollback_configs_are_mutually_exclusive() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if [[ "$MODE" == typed ]]' in source
    assert 'export VLLM_EXTENSION_MANIFESTS="$MANIFEST"' in source
    assert "PRODUCER_CONFIG=$TYPED_PRODUCER" in source
    assert "PRODUCER_CONFIG=$LEGACY_PRODUCER" in source
    assert "unset VLLM_EXTENSION_MANIFESTS" in source


def test_runner_requires_matching_successful_preflight() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if not record.get("ready_for_real_online")' in source
    assert 'record.get("topology") != "direct"' in source
    assert "preflight revision differs from current checkout" in source
    assert "runner ports were not all admitted by preflight" in source
    assert "runner GPUs were not all admitted by preflight" in source
    assert "runner GPUs no longer satisfy the free-memory threshold" in source
    assert "runner port {port} is no longer available" in source


def test_wait_loop_fails_when_a_retained_service_exits() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if ! kill -0 "$pid"' in source
    assert "A retained service process exited before port" in source


def test_proxy_readiness_uses_an_endpoint_the_proxy_exposes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local readiness_path=${2:-/health}" in source
    assert 'wait_for_server "$PROXY_PORT" "/openapi.json"' in source


def test_remote_hangup_also_runs_owned_process_cleanup() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "trap 'exit 129' HUP" in source
    assert "trap - EXIT HUP INT TERM USR1" in source
