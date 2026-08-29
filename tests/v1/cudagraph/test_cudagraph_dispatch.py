# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

import vllm.v1.cudagraph_dispatcher as cudagraph_dispatcher_module
from tests.utils import create_new_process_for_each_test
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.lora import LoRAConfig
from vllm.forward_context import (
    BatchDescriptor,
    CUDAGraphRuntimeMetadata,
    set_forward_context,
)
from vllm.platforms import current_platform
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

DEVICE_TYPE = current_platform.device_type


@pytest.fixture(autouse=True)
def _restore_cudagraph_capturing():
    """Restore the global capture-window flag after every test."""
    yield
    set_cudagraph_capturing_enabled(True)


class _RejectingKeyStrategy:
    """Platform strategy that rejects every runtime key admission."""

    def admit_runtime_key(
        self, num_tokens: int, runtime_metadata: CUDAGraphRuntimeMetadata
    ) -> bool:
        return False


class _RaisingKeyStrategy:
    """Platform strategy that fails during runtime key admission."""

    def admit_runtime_key(
        self, num_tokens: int, runtime_metadata: CUDAGraphRuntimeMetadata
    ) -> bool:
        raise RuntimeError("plugin admission failure")


class _AdmittingKeyStrategy:
    """Platform strategy that owns admission of every runtime key."""

    def __init__(self):
        self.admitted = []

    def admit_runtime_key(
        self, num_tokens: int, runtime_metadata: CUDAGraphRuntimeMetadata
    ) -> bool:
        self.admitted.append((num_tokens, runtime_metadata))
        return True


def _make_full_decode_dispatcher(
    max_num_seqs: int = 8, capture_sizes: tuple[int, ...] = (1, 8)
) -> CudagraphDispatcher:
    comp_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        mode=CompilationMode.NONE,
        cudagraph_capture_sizes=list(capture_sizes),
    )
    dispatcher = CudagraphDispatcher(
        _create_vllm_config(comp_config, max_num_seqs=max_num_seqs)
    )
    dispatcher.initialize_cudagraph_keys(
        comp_config.cudagraph_mode, uniform_decode_query_len=1
    )
    return dispatcher


def _runtime_metadata(token_offset: int = 4) -> CUDAGraphRuntimeMetadata:
    return CUDAGraphRuntimeMetadata(
        token_offset=token_offset,
        variant="parallel_replay",
        backend_tag="test_backend",
    )


def _patch_key_strategy(strategy: object):
    """Patch the concrete platform's key-strategy hook.

    Patching ``type(current_platform)`` (not only the base ``Platform``)
    matters because a platform subclass may override the hook; ``mock``'s
    class-attribute protocol restores an inherited-vs-local attribute
    correctly either way.
    """
    return patch.object(
        type(current_platform),
        "get_cudagraph_key_strategy",
        classmethod(lambda cls, vllm_config, _strategy=strategy: _strategy),
    )


# Helper MLP for testing
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _create_vllm_config(
    compilation_config: CompilationConfig,
    max_num_seqs: int = 8,
    lora_config: bool = False,
) -> MagicMock:
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.compilation_config = compilation_config
    mock_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs,
    )
    mock_config.parallel_config = ParallelConfig()
    mock_config.speculative_config = None  # No speculative decoding
    mock_config.num_speculative_tokens = 0
    if not lora_config:
        mock_config.lora_config = None
    else:
        # Create a real LoRAConfig with specialize_active_lora enabled
        mock_config.lora_config = LoRAConfig(
            max_loras=4,
            specialize_active_lora=True,
        )
    # Mimic the behavior of VllmConfig.__post_init__()
    if compilation_config.mode == CompilationMode.VLLM_COMPILE:
        compilation_config.set_splitting_ops_for_v1(
            all2all_backend=mock_config.parallel_config.all2all_backend,
            data_parallel_size=mock_config.parallel_config.data_parallel_size,
        )

    # mimic VllmConfig.__post_init__
    if compilation_config.cudagraph_capture_sizes:
        compilation_config.max_cudagraph_capture_size = (
            compilation_config.cudagraph_capture_sizes[-1]
        )

        compilation_config.post_init_cudagraph_sizes()

    return mock_config


class TestCudagraphDispatcher:
    @pytest.mark.parametrize(
        "cudagraph_mode_str,compilation_mode,lora_config",
        [
            # Test case 0: Full CG for mixed batches, no separate routine
            ("FULL", CompilationMode.NONE, False),
            # Test case 1: Full CG for uniform batches, piecewise for mixed
            ("FULL_AND_PIECEWISE", CompilationMode.NONE, False),
            # Test case 2: Full CG for uniform batches, no CG for mixed
            ("FULL_DECODE_ONLY", CompilationMode.NONE, False),
            # Test case 3: PIECEWISE for all
            ("PIECEWISE", CompilationMode.VLLM_COMPILE, False),
            # Test case 4: PIECEWISE for all, specialize LoRA cases
            ("PIECEWISE", CompilationMode.VLLM_COMPILE, True),
        ],
    )
    def test_dispatcher(self, cudagraph_mode_str, compilation_mode, lora_config):
        # Setup dispatcher
        comp_config = CompilationConfig(
            cudagraph_mode=cudagraph_mode_str,
            mode=compilation_mode,
            cudagraph_capture_sizes=[1, 8],
        )

        config = _create_vllm_config(
            comp_config, max_num_seqs=8, lora_config=lora_config
        )
        if (
            cudagraph_mode_str == "FULL_AND_PIECEWISE"
            and compilation_mode == CompilationMode.NONE
        ):
            with pytest.raises(AssertionError):
                dispatcher = CudagraphDispatcher(config)
            return

        dispatcher = CudagraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode=comp_config.cudagraph_mode, uniform_decode_query_len=1
        )

        # Verify the key is initialized correctly
        # With LoRA specialization (max_loras=4, specialize_active_lora=True):
        # - lora_cases = [0, 1, 2, 4, 5] (no-lora + powers of 2 up to 4 + max_loras+1)
        # - capture_sizes = [1, 8]
        # - Total keys = 2 sizes × 5 lora_cases = 10
        if cudagraph_mode_str in ["FULL_AND_PIECEWISE", "PIECEWISE"]:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == (
                10 if lora_config else 2
            )
        else:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 0
        if cudagraph_mode_str not in ["NONE", "PIECEWISE"]:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == (
                10 if lora_config else 2
            )
        else:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 0

        # Test dispatch logic
        # 1. non-uniform batch, size in cudagraph size list
        # FULL mode uses exact keys with num_reqs set
        desc_full_with_reqs = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=False)
        # PIECEWISE mode uses relaxed keys with num_reqs=None
        desc_piecewise = BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False)
        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=False, has_lora=False
        )
        if cudagraph_mode_str == "FULL":
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_full_with_reqs
        elif cudagraph_mode_str in ["FULL_AND_PIECEWISE", "PIECEWISE"]:
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == desc_piecewise
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 2. uniform decode batch, size in cudagraph size list
        desc_uniform_exact = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)
        desc_non_uniform = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=False)
        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=True, has_lora=False
        )
        if cudagraph_mode_str == "FULL":
            # Pure FULL mode uses non-uniform keys for all batches
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_non_uniform
        elif cudagraph_mode_str in ["FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"]:
            # These modes have separate uniform decode keys
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_uniform_exact
        elif cudagraph_mode_str == "PIECEWISE":
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == replace(desc_uniform_exact, num_reqs=None, uniform=False)
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 3. No key match
        rt_mode, key = dispatcher.dispatch(
            num_tokens=15, uniform_decode=False, has_lora=False
        )
        assert rt_mode == CUDAGraphMode.NONE
        assert key == BatchDescriptor(num_tokens=15)

        # 4. invalid_modes={FULL} should have a fall back mode
        #    (e.g., cascade attention)
        desc_full_exact = BatchDescriptor(num_tokens=8, uniform=False)
        rt_mode, key = dispatcher.dispatch(
            num_tokens=8,
            uniform_decode=False,
            has_lora=False,
            invalid_modes={CUDAGraphMode.FULL},
        )

        if "PIECEWISE" in cudagraph_mode_str:  # string contains check
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == replace(desc_full_exact, num_reqs=None, uniform=False)
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 5. valid_modes={NONE} always returns NONE even when keys exist
        rt_mode, key = dispatcher.dispatch(
            num_tokens=8,
            uniform_decode=False,
            has_lora=False,
            valid_modes={CUDAGraphMode.NONE},
        )
        assert rt_mode == CUDAGraphMode.NONE
        assert key == BatchDescriptor(num_tokens=8)

    @pytest.mark.parametrize(
        "cudagraph_mode_str,compilation_mode,expected_modes",
        [
            # FULL mode: only FULL keys, no PIECEWISE
            ("FULL", CompilationMode.NONE, [CUDAGraphMode.FULL]),
            # PIECEWISE mode: only PIECEWISE keys
            ("PIECEWISE", CompilationMode.VLLM_COMPILE, [CUDAGraphMode.PIECEWISE]),
            # FULL_DECODE_ONLY: only FULL keys for uniform decode
            ("FULL_DECODE_ONLY", CompilationMode.NONE, [CUDAGraphMode.FULL]),
            # NONE mode: no keys
            ("NONE", CompilationMode.NONE, []),
        ],
    )
    def test_get_capture_descs(
        self, cudagraph_mode_str, compilation_mode, expected_modes
    ):
        """Test get_capture_descs returns correctly grouped and ordered descs."""
        comp_config = CompilationConfig(
            cudagraph_mode=cudagraph_mode_str,
            mode=compilation_mode,
            cudagraph_capture_sizes=[1, 4, 8, 16],
        )

        config = _create_vllm_config(comp_config, max_num_seqs=16)
        dispatcher = CudagraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode=comp_config.cudagraph_mode, uniform_decode_query_len=1
        )

        capture_descs = dispatcher.get_capture_descs()

        # Verify we get the expected modes
        actual_modes = [mode for mode, _ in capture_descs]
        assert actual_modes == expected_modes

        # Verify each group is sorted largest-first
        for mode, descs in capture_descs:
            assert len(descs) > 0, "Each group should have at least one descriptor"
            num_tokens_list = [d.num_tokens for d in descs]
            assert num_tokens_list == sorted(num_tokens_list, reverse=True), (
                f"Descriptors for {mode} should be sorted largest-first"
            )

            # All descriptors in a group should have same uniform value
            uniform_values = [d.uniform for d in descs]
            assert len(set(uniform_values)) == 1, (
                "All descriptors in a group should have the same uniform value"
            )

    def test_get_capture_descs_empty_when_not_initialized(self):
        """Test that get_capture_descs returns empty list when keys not initialized."""
        comp_config = CompilationConfig(
            cudagraph_mode="FULL",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[1, 8],
        )
        config = _create_vllm_config(comp_config, max_num_seqs=8)
        dispatcher = CudagraphDispatcher(config)
        # Don't initialize keys

        assert dispatcher.get_capture_descs() == []

    def test_runtime_key_registration_is_explicit(self):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[1, 8],
        )
        dispatcher = CudagraphDispatcher(
            _create_vllm_config(comp_config, max_num_seqs=8)
        )
        dispatcher.initialize_cudagraph_keys(
            comp_config.cudagraph_mode, uniform_decode_query_len=1
        )
        metadata = CUDAGraphRuntimeMetadata(
            token_offset=4,
            variant="parallel_replay",
            backend_tag="test_backend",
        )
        initial_keys = dispatcher.cudagraph_keys[CUDAGraphMode.FULL].copy()

        # Explicit None keeps the test independent of the ambient platform
        # strategy (the assertion below is about core's own opt-in gate).
        with _patch_key_strategy(None):
            mode, descriptor = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=metadata,
            )
            assert mode == CUDAGraphMode.NONE
            assert descriptor == BatchDescriptor(
                num_tokens=4, runtime_metadata=metadata
            )
            assert dispatcher.cudagraph_keys[CUDAGraphMode.FULL] == initial_keys

            mode, descriptor = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=metadata,
                allow_runtime_key_registration=True,
            )
        assert mode == CUDAGraphMode.FULL
        assert descriptor == BatchDescriptor(
            num_tokens=4,
            num_reqs=4,
            uniform=True,
            runtime_metadata=metadata,
        )
        assert descriptor in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]

    @pytest.mark.parametrize(
        ("metadata", "dispatch_kwargs"),
        [
            ("invalid_metadata", {}),
            (
                CUDAGraphRuntimeMetadata(0, "parallel_replay", "test_backend"),
                {},
            ),
            (
                CUDAGraphRuntimeMetadata(True, "parallel_replay", "test_backend"),
                {},
            ),
            (CUDAGraphRuntimeMetadata(4, "", "test_backend"), {}),
            (CUDAGraphRuntimeMetadata(4, "parallel_replay", ""), {}),
            (
                CUDAGraphRuntimeMetadata(4, " parallel_replay", "test_backend"),
                {},
            ),
            (
                CUDAGraphRuntimeMetadata(4, "parallel replay", "test_backend"),
                {},
            ),
            (
                CUDAGraphRuntimeMetadata(
                    4, "parallel_replay", "test_backend", " metadata"
                ),
                {},
            ),
            (
                CUDAGraphRuntimeMetadata(4, "parallel_replay", "test_backend"),
                {"uniform_decode": False},
            ),
            (
                CUDAGraphRuntimeMetadata(4, "parallel_replay", "test_backend"),
                {"has_lora": True},
            ),
            (
                CUDAGraphRuntimeMetadata(4, "parallel_replay", "test_backend"),
                {"num_active_loras": 1},
            ),
            (
                CUDAGraphRuntimeMetadata(4, "parallel_replay", "test_backend"),
                {"invalid_modes": {CUDAGraphMode.FULL}},
            ),
        ],
    )
    def test_invalid_runtime_key_fails_without_mutation(
        self,
        metadata: object,
        dispatch_kwargs: dict,
    ):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[1, 8],
        )
        dispatcher = CudagraphDispatcher(
            _create_vllm_config(comp_config, max_num_seqs=8)
        )
        dispatcher.initialize_cudagraph_keys(
            comp_config.cudagraph_mode, uniform_decode_query_len=1
        )
        initial_keys = {
            mode: keys.copy() for mode, keys in dispatcher.cudagraph_keys.items()
        }

        kwargs = {
            "num_tokens": 4,
            "uniform_decode": True,
            "runtime_metadata": metadata,
            "allow_runtime_key_registration": True,
        }
        kwargs.update(dispatch_kwargs)
        mode, _ = dispatcher.dispatch(
            **kwargs,
        )

        assert mode == CUDAGraphMode.NONE
        assert dispatcher.cudagraph_keys == initial_keys

    @pytest.mark.parametrize(
        ("num_tokens", "token_offset"),
        [(3, 2), (4, 3), (10, 2)],
    )
    def test_runtime_key_alignment_and_capacity_fail_closed(
        self, num_tokens: int, token_offset: int
    ):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[2, 10],
        )
        dispatcher = CudagraphDispatcher(
            _create_vllm_config(comp_config, max_num_seqs=4)
        )
        dispatcher.initialize_cudagraph_keys(
            comp_config.cudagraph_mode, uniform_decode_query_len=2
        )
        initial_keys = {
            mode: keys.copy() for mode, keys in dispatcher.cudagraph_keys.items()
        }
        metadata = CUDAGraphRuntimeMetadata(
            token_offset=token_offset,
            variant="parallel_replay",
            backend_tag="test_backend",
        )

        mode, _ = dispatcher.dispatch(
            num_tokens=num_tokens,
            uniform_decode=True,
            runtime_metadata=metadata,
            allow_runtime_key_registration=True,
        )

        assert mode == CUDAGraphMode.NONE
        assert dispatcher.cudagraph_keys == initial_keys


class TestRuntimeKeyCaptureLifecycle:
    """Host-level contract for the runtime-key capture lifecycle.

    A runtime key is only ever returned with a graph mode when it is
    guaranteed to have a captured graph: either it was admitted while the
    startup capture window was open, or it was registered (by the platform
    plugin) before capture and is already in the key registry. Anything else
    fails closed to eager execution without mutating the registry.
    """

    def test_no_strategy_keeps_core_neutral_admission(self):
        # No plugin strategy (the base hook returns None, and an explicit
        # None patch keeps the test independent of the ambient platform):
        # core enforces only its generic schema, and the standard dispatch
        # path is unaffected by the runtime-key machinery.
        from vllm.platforms.interface import Platform

        dispatcher = _make_full_decode_dispatcher()
        assert Platform.get_cudagraph_key_strategy(dispatcher.vllm_config) is None

        static_mode, static_desc = dispatcher.dispatch(
            num_tokens=8, uniform_decode=True
        )
        assert static_mode == CUDAGraphMode.FULL
        assert static_desc == BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)

        metadata = _runtime_metadata()
        with _patch_key_strategy(None):
            mode, desc = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=metadata,
                allow_runtime_key_registration=True,
            )
        assert mode == CUDAGraphMode.FULL
        assert desc == BatchDescriptor(
            num_tokens=4, num_reqs=4, uniform=True, runtime_metadata=metadata
        )
        assert desc in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]

    @pytest.mark.parametrize(
        "strategy", [_RejectingKeyStrategy(), _RaisingKeyStrategy()]
    )
    def test_strategy_rejection_or_failure_fails_closed(self, strategy):
        dispatcher = _make_full_decode_dispatcher()
        initial = {m: s.copy() for m, s in dispatcher.cudagraph_keys.items()}
        metadata = _runtime_metadata()

        with _patch_key_strategy(strategy):
            mode, desc = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=metadata,
                allow_runtime_key_registration=True,
            )

        assert mode == CUDAGraphMode.NONE
        assert desc == BatchDescriptor(num_tokens=4, runtime_metadata=metadata)
        assert dispatcher.cudagraph_keys == initial

    def test_duplicate_runtime_key_is_idempotent(self):
        dispatcher = _make_full_decode_dispatcher()
        metadata = _runtime_metadata()

        with _patch_key_strategy(None):
            first_mode, first_desc = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=metadata,
                allow_runtime_key_registration=True,
            )
            assert first_mode == CUDAGraphMode.FULL
            size_after_first = len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL])

            second_mode, second_desc = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=metadata,
                allow_runtime_key_registration=True,
            )
        assert (second_mode, second_desc) == (first_mode, first_desc)
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == size_after_first

        # Duplicate registration through the plugin-facing API is a no-op.
        dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, first_desc)
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == size_after_first

    def test_pre_registered_runtime_key_replays_after_capture_window(self):
        dispatcher = _make_full_decode_dispatcher()
        metadata = _runtime_metadata()
        descriptor = BatchDescriptor(
            num_tokens=4, num_reqs=4, uniform=True, runtime_metadata=metadata
        )

        # Plugin-owned startup contract: register before capture so
        # get_capture_descs enumerates the key for capture_model.
        dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, descriptor)
        assert any(
            mode == CUDAGraphMode.FULL and descriptor in descs
            for mode, descs in dispatcher.get_capture_descs()
        )

        # Post-startup (capture disabled): the admitted key still replays.
        set_cudagraph_capturing_enabled(False)
        mode, dispatched = dispatcher.dispatch(
            num_tokens=4,
            uniform_decode=True,
            runtime_metadata=metadata,
            allow_runtime_key_registration=True,
        )
        assert mode == CUDAGraphMode.FULL
        assert dispatched == descriptor

    def test_new_runtime_key_fails_closed_after_capture_window(self):
        dispatcher = _make_full_decode_dispatcher()
        initial = {m: s.copy() for m, s in dispatcher.cudagraph_keys.items()}
        metadata = _runtime_metadata()

        # Simulate completed startup capture.
        set_cudagraph_capturing_enabled(False)

        with _patch_key_strategy(None):
            mode, desc = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=metadata,
                allow_runtime_key_registration=True,
            )
        assert mode == CUDAGraphMode.NONE
        assert desc == BatchDescriptor(num_tokens=4, runtime_metadata=metadata)
        assert dispatcher.cudagraph_keys == initial

    def test_strategy_owned_admission_after_capture_window(self):
        """A registered platform strategy owns post-startup admission.

        Admitted keys are registered (bounded, atomic); rejected keys and
        non-opt-in requests still fail closed to eager without mutation.
        """
        admitted_desc = BatchDescriptor(
            num_tokens=4,
            num_reqs=4,
            uniform=True,
            runtime_metadata=_runtime_metadata(token_offset=4),
        )

        # Simulate completed startup capture.
        set_cudagraph_capturing_enabled(False)

        # Phase 1: an admitting strategy registers post-startup keys.
        strategy = _AdmittingKeyStrategy()
        dispatcher = _make_full_decode_dispatcher()
        initial_full_keys = dispatcher.cudagraph_keys[CUDAGraphMode.FULL].copy()
        with _patch_key_strategy(strategy):
            mode, desc = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=_runtime_metadata(token_offset=4),
                allow_runtime_key_registration=True,
            )
            assert mode == CUDAGraphMode.FULL
            assert desc == admitted_desc
            assert desc in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
            assert strategy.admitted == [(4, _runtime_metadata(token_offset=4))]

            # Non-opt-in requests still fail closed under strategy ownership.
            mode, _ = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=_runtime_metadata(token_offset=16),
                allow_runtime_key_registration=False,
            )
            assert mode == CUDAGraphMode.NONE

        # Phase 2: a rejecting strategy fails closed for brand-new keys.
        rejecting_dispatcher = _make_full_decode_dispatcher()
        rejecting_initial = {
            m: s.copy() for m, s in rejecting_dispatcher.cudagraph_keys.items()
        }
        with _patch_key_strategy(_RejectingKeyStrategy()):
            mode, desc = rejecting_dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=_runtime_metadata(token_offset=8),
                allow_runtime_key_registration=True,
            )
        assert mode == CUDAGraphMode.NONE
        assert desc.runtime_metadata == _runtime_metadata(token_offset=8)
        assert rejecting_dispatcher.cudagraph_keys == rejecting_initial

        # Phase 3: a strategy-admitted key still replays post-startup (the
        # already-registered fast path), while a fresh neutral dispatcher
        # fail-closes brand-new keys.
        mode, dispatched = dispatcher.dispatch(
            num_tokens=4,
            uniform_decode=True,
            runtime_metadata=_runtime_metadata(token_offset=4),
            allow_runtime_key_registration=True,
        )
        assert mode == CUDAGraphMode.FULL
        assert dispatched == admitted_desc

        neutral_dispatcher = _make_full_decode_dispatcher()
        neutral_initial = {
            m: s.copy() for m, s in neutral_dispatcher.cudagraph_keys.items()
        }
        with _patch_key_strategy(None):
            mode, _ = neutral_dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=_runtime_metadata(token_offset=12),
                allow_runtime_key_registration=True,
            )
        assert mode == CUDAGraphMode.NONE
        assert neutral_dispatcher.cudagraph_keys == neutral_initial
        assert dispatcher.cudagraph_keys[CUDAGraphMode.FULL] == (
            initial_full_keys | {admitted_desc}
        )

    def test_runtime_key_count_is_bounded(self):
        dispatcher = _make_full_decode_dispatcher()
        with (
            _patch_key_strategy(None),
            patch.object(
                cudagraph_dispatcher_module, "MAX_RUNTIME_GRAPH_KEYS_PER_MODE", 2
            ),
        ):
            admitted = []
            for offset in (4, 8, 12, 16):
                mode, desc = dispatcher.dispatch(
                    num_tokens=4,
                    uniform_decode=True,
                    runtime_metadata=_runtime_metadata(token_offset=offset),
                    allow_runtime_key_registration=True,
                )
                if mode == CUDAGraphMode.FULL:
                    admitted.append(desc)

            assert len(admitted) == 2
            runtime_keys = [
                d
                for d in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
                if d.runtime_metadata is not None
            ]
            assert len(runtime_keys) == 2
            assert dispatcher._runtime_key_cap_warned

        # Releasing the test bound admits further keys again.
        with _patch_key_strategy(None):
            mode, _ = dispatcher.dispatch(
                num_tokens=4,
                uniform_decode=True,
                runtime_metadata=_runtime_metadata(token_offset=12),
                allow_runtime_key_registration=True,
            )
        assert mode == CUDAGraphMode.FULL

    @pytest.mark.parametrize(
        "use_strategy", [False, True], ids=["neutral", "strategy_owned"]
    )
    def test_concurrent_runtime_key_registration_is_safe(self, use_strategy):
        if use_strategy:
            # Post-startup with strategy ownership: admission still
            # exercised concurrently.
            set_cudagraph_capturing_enabled(False)
        dispatcher = _make_full_decode_dispatcher()
        # The neutral case pins an explicit None strategy so the test stays
        # independent of the ambient platform.
        strategy_patch = _patch_key_strategy(
            _AdmittingKeyStrategy() if use_strategy else None
        )
        strategy_patch.start()
        try:
            with patch.object(
                cudagraph_dispatcher_module, "MAX_RUNTIME_GRAPH_KEYS_PER_MODE", 8
            ):
                offsets = list(range(1, 33))
                errors: list[Exception] = []

                def dispatch_one(offset: int):
                    try:
                        mode, desc = dispatcher.dispatch(
                            num_tokens=4,
                            uniform_decode=True,
                            runtime_metadata=_runtime_metadata(token_offset=offset),
                            allow_runtime_key_registration=True,
                        )
                        return offset, mode, desc
                    except Exception as exc:  # pragma: no cover - contract guard
                        errors.append(exc)
                        return offset, None, None

                with ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(dispatch_one, offsets))

                assert not errors
                full = {off for off, mode, _ in results if mode == CUDAGraphMode.FULL}
                none = {off for off, mode, _ in results if mode == CUDAGraphMode.NONE}
                assert len(full) + len(none) == len(offsets)

                registered = {
                    d.runtime_metadata.token_offset
                    for d in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
                    if d.runtime_metadata is not None
                }
                assert len(registered) == 8
                assert full == registered
                assert none == set(offsets) - registered
        finally:
            strategy_patch.stop()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
class TestCUDAGraphWrapper:
    def setup_method(self):
        self.vllm_config = _create_vllm_config(CompilationConfig())
        self.model = SimpleMLP().to(DEVICE_TYPE)
        self.persistent_input_buffer = torch.zeros(1, 10, device=DEVICE_TYPE)
        self.input_tensor = torch.randn(1, 10, device=DEVICE_TYPE)

    def test_capture_and_replay(self):
        wrapper = CUDAGraphWrapper(
            self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        batch_descriptor = BatchDescriptor(num_tokens=10)

        # 0. global warmup
        with set_forward_context(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=None,
        ):
            wrapper(self.input_tensor)

        # 1. Capture
        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor,
            ),
            patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_cuda_graph,
        ):
            output1 = wrapper(self.input_tensor)
            # capturing phase should generate a zero output
            assert torch.allclose(output1, torch.zeros_like(output1))
            mock_cuda_graph.assert_called_once()

        assert batch_descriptor in wrapper.concrete_cudagraph_entries
        entry = wrapper.concrete_cudagraph_entries[batch_descriptor]
        assert entry.cudagraph is not None

        # 2. Replay
        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor,
            ),
            patch.object(
                entry.cudagraph, "replay", wraps=entry.cudagraph.replay
            ) as mock_replay,
        ):
            output2 = wrapper(self.input_tensor)
            mock_replay.assert_called_once()

        # Compare with eager output
        eager_output = self.model(self.input_tensor)
        torch.testing.assert_close(eager_output, output2)

    def test_bypass_on_mode_mismatch(self):
        wrapper = CUDAGraphWrapper(
            self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        batch_descriptor = BatchDescriptor(num_tokens=10)

        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=batch_descriptor,
            ),
            patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_cuda_graph,
            patch.object(
                self.model, "forward", wraps=self.model.forward
            ) as mock_forward,
        ):
            wrapper(self.input_tensor)
            mock_cuda_graph.assert_not_called()
            mock_forward.assert_called_once()
        assert not wrapper.concrete_cudagraph_entries

    def test_bypass_on_mode_none(self):
        wrapper = CUDAGraphWrapper(
            self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        batch_descriptor = BatchDescriptor(num_tokens=10)

        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=batch_descriptor,
            ),
            patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_cuda_graph,
        ):
            wrapper(self.input_tensor)
            mock_cuda_graph.assert_not_called()
        assert not wrapper.concrete_cudagraph_entries


def _run_and_monitor_call(
    wrapper, input_tensor, runtime_mode, batch_descriptor, vllm_config
):
    """Helper to run a single call and monitor the action."""

    with (
        patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_graph_context,
        patch.object(wrapper, "runnable", wraps=wrapper.runnable) as mock_runnable,
    ):
        entry = wrapper.concrete_cudagraph_entries.get(batch_descriptor, None)

        context = set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_config,
            cudagraph_runtime_mode=runtime_mode,
            batch_descriptor=batch_descriptor,
        )
        mock_replay = MagicMock()
        if entry and entry.cudagraph:
            with (
                context,
                patch.object(
                    entry.cudagraph, "replay", new_callable=MagicMock
                ) as mock_replay,
            ):
                wrapper(input_tensor)
        else:
            with context:
                wrapper(input_tensor)

        if mock_graph_context.called:
            # note that this is globally mocked, so it will be detected
            # even whether called by the inner or outer wrapper
            return "capture_global"
        if mock_replay.called:
            # only for outer wrapper
            return "replay"
        if mock_runnable.call_count > 0:
            # only for outer wrapper
            return "bypass"
        return "unknown"


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_capture_replay_bypass_logic():
    comp_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=[1, 2],
    )
    vllm_config = _create_vllm_config(comp_config)
    dispatcher = CudagraphDispatcher(vllm_config)
    dispatcher.initialize_cudagraph_keys(
        comp_config.cudagraph_mode, uniform_decode_query_len=1
    )
    model = SimpleMLP().to(DEVICE_TYPE)
    full_wrapper = CUDAGraphWrapper(model, vllm_config, CUDAGraphMode.FULL)
    max_bs = 16
    persistent_input_buffer = torch.zeros(max_bs, 10, device=DEVICE_TYPE)
    input_1 = persistent_input_buffer[:1]
    input_2 = persistent_input_buffer[:2]
    input_3 = persistent_input_buffer[:3]

    desc_1 = BatchDescriptor(num_tokens=1)
    desc_2 = BatchDescriptor(num_tokens=2)
    desc_3_unseen = BatchDescriptor(num_tokens=3)

    # 0. global warmup
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        full_wrapper(input_1)

    rt_mode, key = dispatcher.dispatch(num_tokens=desc_1.num_tokens)
    # 1. Capture first shape
    action = _run_and_monitor_call(full_wrapper, input_1, rt_mode, key, vllm_config)
    assert action == "capture_global"

    # 2. Replay first shape
    action = _run_and_monitor_call(full_wrapper, input_1, rt_mode, key, vllm_config)
    assert action == "replay"

    rt_mode, key = dispatcher.dispatch(num_tokens=desc_2.num_tokens)
    # 3. Capture second shape
    action = _run_and_monitor_call(full_wrapper, input_2, rt_mode, key, vllm_config)
    assert action == "capture_global"

    # 4. Replay second shape
    action = _run_and_monitor_call(
        full_wrapper, input_2, CUDAGraphMode.FULL, key, vllm_config
    )
    assert action == "replay"

    # 5. Bypass if no key match
    rt_mode, key = dispatcher.dispatch(num_tokens=desc_3_unseen.num_tokens)
    assert rt_mode == CUDAGraphMode.NONE
    action = _run_and_monitor_call(full_wrapper, input_3, rt_mode, key, vllm_config)
    assert action == "bypass"

    # capture unseen shape is not allowed after disable
    set_cudagraph_capturing_enabled(False)
    with pytest.raises(RuntimeError):
        _run_and_monitor_call(
            full_wrapper, input_3, CUDAGraphMode.FULL, desc_3_unseen, vllm_config
        )
    set_cudagraph_capturing_enabled(True)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_runtime_key_capture_before_replay_and_fail_closed_after():
    """Integration regression for the runtime-key capture lifecycle.

    A previously unseen runtime descriptor must actually be captured before
    it is replayed (during the startup capture window), and once startup
    capture is disabled, admitting a brand-new runtime key must fail closed
    to eager execution instead of raising or replaying a missing graph.
    """
    comp_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=[1, 4],
    )
    vllm_config = _create_vllm_config(comp_config)
    dispatcher = CudagraphDispatcher(vllm_config)
    dispatcher.initialize_cudagraph_keys(
        comp_config.cudagraph_mode, uniform_decode_query_len=1
    )
    model = SimpleMLP().to(DEVICE_TYPE)
    full_wrapper = CUDAGraphWrapper(model, vllm_config, CUDAGraphMode.FULL)
    persistent_input_buffer = torch.zeros(4, 10, device=DEVICE_TYPE)
    input_a = persistent_input_buffer[:4]

    # 0. global warmup
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        full_wrapper(input_a)

    # 1. Capture window open: an unseen runtime key is admitted, exposed to
    #    capture_model via get_capture_descs, captured, then replayed.
    set_cudagraph_capturing_enabled(True)
    metadata = CUDAGraphRuntimeMetadata(
        token_offset=2,
        variant="parallel_replay",
        backend_tag="test_backend",
    )
    rt_mode, key = dispatcher.dispatch(
        num_tokens=4,
        uniform_decode=True,
        runtime_metadata=metadata,
        allow_runtime_key_registration=True,
    )
    assert rt_mode == CUDAGraphMode.FULL
    assert key in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
    assert any(
        key in descs
        for mode, descs in dispatcher.get_capture_descs()
        if mode == CUDAGraphMode.FULL
    )
    action = _run_and_monitor_call(full_wrapper, input_a, rt_mode, key, vllm_config)
    assert action == "capture_global"
    action = _run_and_monitor_call(full_wrapper, input_a, rt_mode, key, vllm_config)
    assert action == "replay"

    # 2. Capture window closed: a brand-new runtime key fails closed to
    #    eager. The wrapper is bypassed and must never raise or attempt to
    #    replay a missing graph.
    set_cudagraph_capturing_enabled(False)
    new_metadata = CUDAGraphRuntimeMetadata(
        token_offset=4,
        variant="parallel_replay",
        backend_tag="test_backend",
    )
    rt_mode, key = dispatcher.dispatch(
        num_tokens=4,
        uniform_decode=True,
        runtime_metadata=new_metadata,
        allow_runtime_key_registration=True,
    )
    assert rt_mode == CUDAGraphMode.NONE
    assert key.runtime_metadata == new_metadata
    action = _run_and_monitor_call(full_wrapper, input_a, rt_mode, key, vllm_config)
    assert action == "bypass"


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_nested_wrappers():
    """Tests a scenario with a PIECEWISE wrapper inside a FULL one."""
    comp_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=[1],
    )
    vllm_config = _create_vllm_config(comp_config)
    dispatcher = CudagraphDispatcher(vllm_config)
    dispatcher.initialize_cudagraph_keys(
        comp_config.cudagraph_mode, uniform_decode_query_len=1
    )
    model = SimpleMLP().to(DEVICE_TYPE)
    full_wrapper = CUDAGraphWrapper(model, vllm_config, CUDAGraphMode.FULL)
    input_1 = torch.randn(1, 10, device=DEVICE_TYPE)

    # Setup: Inner model is wrapped with PIECEWISE, outer with FULL
    inner_model = SimpleMLP().to(DEVICE_TYPE)
    piecewise_wrapper = CUDAGraphWrapper(
        inner_model, vllm_config, CUDAGraphMode.PIECEWISE
    )
    inner_model.forward = MagicMock(wraps=inner_model.forward)
    outer_model = SimpleMLP().to(DEVICE_TYPE)
    # When outer model is called, it calls the piecewise_wrapper
    outer_model.forward = MagicMock(
        wraps=outer_model.forward, side_effect=piecewise_wrapper
    )
    full_wrapper = CUDAGraphWrapper(outer_model, vllm_config, CUDAGraphMode.FULL)

    desc_1 = BatchDescriptor(num_tokens=1)

    # 0. global warmup
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        full_wrapper(input_1)

    # --- Test runtime mode FULL---
    # Run with FULL mode context. Expect outer wrapper to capture.
    # The inner mock should be called once inside the graph capture.
    outer_model.forward.reset_mock()
    inner_model.forward.reset_mock()
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.FULL, desc_1, vllm_config
    )
    assert action == "capture_global"
    assert outer_model.forward.call_count == 1
    assert inner_model.forward.call_count == 1

    # Run again. Expect outer wrapper to replay.
    # The outer model should NOT be called because the whole graph
    # is replayed.
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.FULL, desc_1, vllm_config
    )
    assert action == "replay"
    assert outer_model.forward.call_count == 1  # No new call
    assert inner_model.forward.call_count == 1

    # --- Test runtime mode PIECEWISE ---
    outer_model.forward.reset_mock()
    inner_model.forward.reset_mock()
    # Run with PIECEWISE mode context.
    # Expect outer wrapper to bypass and call inner wrapper.
    # Inner wrapper should capture.
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.PIECEWISE, desc_1, vllm_config
    )
    assert action == "capture_global"
    assert outer_model.forward.call_count == 1
    assert inner_model.forward.call_count == 1

    # Run again with PIECEWISE.
    # Outer bypasses, inner replays.
    action = _run_and_monitor_call(
        full_wrapper, input_1, CUDAGraphMode.PIECEWISE, desc_1, vllm_config
    )
    assert action == "bypass"
    assert outer_model.forward.call_count == 2
    assert inner_model.forward.call_count == 1
