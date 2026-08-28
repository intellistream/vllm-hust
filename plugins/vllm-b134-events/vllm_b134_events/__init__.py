# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B134 events plugin entry point.

Registers :class:`vllm_b134_events.sink.B134JsonlSink` on the generic
engine-core event bus when the plugin is enabled.

The sink is only registered when ``B134_EVENTS_FILE`` is set.  This preserves
the default-off property of the core event outlet: hot paths do not construct
or timestamp events merely because the wheel is installed.
"""

from __future__ import annotations

import os

from vllm.v1.events import EventBus
from vllm_b134_events.sink import B134JsonlSink

_sink: B134JsonlSink | None = None


def register() -> None:
    """Entry point (``vllm.general_plugins``): attach the B134 JSONL sink."""
    global _sink
    if _sink is not None:
        return  # re-entrant: called once per vLLM process
    path = os.environ.get("B134_EVENTS_FILE")
    if not path:
        return
    _sink = B134JsonlSink(path)
    EventBus.register_sink(_sink)
    _sink.start()


__all__ = ["register", "B134JsonlSink"]
