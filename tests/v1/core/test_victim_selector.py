# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.v1.core.sched.victim_selector import (
    NoOpVictimSelector,
    get_victim_selector,
)

pytestmark = pytest.mark.cpu_test


class _ValidSelector:
    vllm_victim_selector_api_version = 1

    @classmethod
    def from_vllm_config(cls, vllm_config):
        return cls()

    def pick_victim(self, running, policy, *, kv_utilization=None, now_s=None):
        return running[-1]

    def emit_observability_log(self, logger, scheduler_name):
        pass

    def export_metrics(self):
        return {}


def _config(**additional_config):
    return SimpleNamespace(additional_config=additional_config)


def _entry_point(name, selector_cls=_ValidSelector):
    ep = Mock(name=f"entry-point-{name}")
    ep.name = name
    ep.dist = SimpleNamespace(name=f"{name}-distribution", version="1.2.3")
    ep.value = f"test_selectors:{selector_cls.__name__}"
    ep.load.return_value = selector_cls
    return ep


def test_no_plugin_uses_default_selector():
    with patch("importlib.metadata.entry_points", return_value=[]):
        selector = get_victim_selector(_config())

    assert isinstance(selector, NoOpVictimSelector)


def test_disabled_plugin_discovery_uses_default_selector():
    with patch("importlib.metadata.entry_points") as discover:
        selector = get_victim_selector(_config(victim_selector_plugin_disabled=True))

    assert isinstance(selector, NoOpVictimSelector)
    discover.assert_not_called()


def test_installed_plugin_is_ignored_without_explicit_selection():
    with patch("importlib.metadata.entry_points") as discover:
        selector = get_victim_selector(_config())

    assert isinstance(selector, NoOpVictimSelector)
    discover.assert_not_called()


def test_requested_plugin_is_selected_deterministically():
    bidkv = _entry_point("bidkv")
    other = _entry_point("other")
    with (
        patch("importlib.metadata.entry_points", return_value=[other, bidkv]),
        patch("vllm.v1.core.sched.victim_selector.logger.info") as log_info,
    ):
        selector = get_victim_selector(_config(victim_selector_plugin="bidkv"))

    assert isinstance(selector, _ValidSelector)
    bidkv.load.assert_called_once_with()
    other.load.assert_not_called()
    log_info.assert_called_once()
    log_format, *log_args = log_info.call_args.args
    assert "distribution=%s==%s" in log_format
    assert "source=%s" in log_format
    assert "api_version=%s" in log_format
    assert log_args[0] == "bidkv"
    assert log_args[1:3] == ["bidkv-distribution", "1.2.3"]
    assert log_args[3].endswith("test_victim_selector.py")
    assert log_args[4] == 1


def test_multiple_plugins_do_not_affect_default_selection():
    with patch("importlib.metadata.entry_points") as discover:
        selector = get_victim_selector(_config())

    assert isinstance(selector, NoOpVictimSelector)
    discover.assert_not_called()


def test_missing_requested_plugin_fails_closed():
    with (
        patch(
            "importlib.metadata.entry_points",
            return_value=[_entry_point("other")],
        ),
        pytest.raises(ValueError, match="is not installed"),
    ):
        get_victim_selector(_config(victim_selector_plugin="bidkv"))


def test_duplicate_requested_plugin_fails_closed():
    eps = [_entry_point("bidkv"), _entry_point("bidkv")]
    with (
        patch("importlib.metadata.entry_points", return_value=eps),
        pytest.raises(RuntimeError, match="Multiple victim selector entry points"),
    ):
        get_victim_selector(_config(victim_selector_plugin="bidkv"))


def test_broken_requested_plugin_fails_closed():
    ep = _entry_point("bidkv")
    ep.load.side_effect = ImportError("broken plugin")
    with (
        patch("importlib.metadata.entry_points", return_value=[ep]),
        pytest.raises(RuntimeError, match="Failed to load requested"),
    ):
        get_victim_selector(_config(victim_selector_plugin="bidkv"))


def test_requested_plugin_discovery_failure_fails_closed():
    with (
        patch(
            "importlib.metadata.entry_points",
            side_effect=RuntimeError("metadata unavailable"),
        ),
        pytest.raises(RuntimeError, match="Failed to discover requested"),
    ):
        get_victim_selector(_config(victim_selector_plugin="bidkv"))


def test_requested_plugin_with_wrong_api_version_fails_closed():
    class OldSelector(_ValidSelector):
        vllm_victim_selector_api_version = 0

    ep = _entry_point("old", OldSelector)
    with (
        patch("importlib.metadata.entry_points", return_value=[ep]),
        pytest.raises(RuntimeError, match="expected 1"),
    ):
        get_victim_selector(_config(victim_selector_plugin="old"))


@pytest.mark.parametrize("requested", ["", "   ", 1])
def test_requested_plugin_name_must_be_a_non_empty_string(requested):
    with pytest.raises(ValueError, match="non-empty string"):
        get_victim_selector(_config(victim_selector_plugin=requested))
