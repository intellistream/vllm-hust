# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest

from vllm.v1.engine.core import EngineCore


def test_scheduler_and_process_cleanup_survive_executor_shutdown_failure() -> None:
    engine_core = object.__new__(EngineCore)
    engine_core.structured_output_manager = Mock()
    engine_core.model_executor = Mock()
    engine_core.model_executor.shutdown.side_effect = RuntimeError("executor failed")
    engine_core.scheduler = Mock()

    with (
        patch("vllm.v1.engine.core.gc.unfreeze") as unfreeze,
        patch("vllm.v1.engine.core.cleanup_dist_env_and_memory") as cleanup,
        pytest.raises(RuntimeError, match="executor failed"),
    ):
        engine_core.shutdown()

    engine_core.scheduler.shutdown.assert_called_once_with()
    unfreeze.assert_called_once_with()
    cleanup.assert_called_once_with()
