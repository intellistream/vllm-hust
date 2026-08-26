# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B134 events plugin entry point.

Registers :class:`vllm_b134_events.sink.B134JsonlSink` on the generic
engine-core event bus when the plugin is enabled.

The sink is only active when ``B134_EVENTS_FILE`` is set (or a path is passed
explicitly); without it the plugin registers a sink that is a no-op, keeping
the default-off property of the core event outlet.
"""

from __future__ import annotations

from vllm.v1.events import EventBus

from vllm_b134_events.sink import B134JsonlSink

_sink: B134JsonlSink | None = None


def register() -> None:
    """Entry point (``vllm.general_plugins``): attach the B134 JSONL sink."""
    global _sink
    if _sink is not None:
        return  # re-entrant: called once per vLLM process
    _sink = B134JsonlSink()
    EventBus.register_sink(_sink)
    _sink.start()


__all__ = ["register", "B134JsonlSink"]
