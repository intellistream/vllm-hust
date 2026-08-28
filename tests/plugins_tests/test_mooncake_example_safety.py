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
