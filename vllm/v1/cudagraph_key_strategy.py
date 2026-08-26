# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Cudagraph key strategies (extended capture/replay key policy).

The standard cudagraph dispatcher only understands padded keys derived from
(num_tokens, num_reqs, uniform, has_lora, num_active_loras). Split-batch
(dual-stream) replay additionally needs offset / variant-aware keys, for
example DUAL_INPLACE second-split offset views into the original buffer.

The construction and dispatch *policy* for these extended keys is owned by a
pluggable :class:`CudagraphKeyStrategy` instead of being inlined into the
dispatcher control flow. Core keeps only:

  - the stable replay entry point (``CudagraphDispatcher.dispatch``),
  - the versioned key metadata (``BatchDescriptor``, including the extended
    metadata fields consumed by this strategy),
  - fail-closed eager fallback (no registered key -> ``CUDAGraphMode.NONE``).

The default implementation (:class:`InplaceSplitKeyStrategy`) preserves the
previous inline behavior exactly. Platforms that wish to own this policy
implement ``Platform.get_cudagraph_key_strategy()`` and return their own
strategy; the default implementation can then be removed from core.
"""

from __future__ import annotations

from typing import Any, Protocol

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor


class CudagraphKeyStrategy(Protocol):
    """Protocol for extended cudagraph key dispatch strategies.

    A strategy inspects the extended key metadata and either returns a
    ``(runtime mode, batch descriptor)`` pair or ``None`` to fall through to
    the standard padded-key dispatch path.
    """

    def dispatch(
        self,
        *,
        dispatcher: Any,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int,
        start_num_tokens: int,
        allow_inplace_lazy_key: bool,
        graph_variant: str,
        attention_backend: str,
        capture_metadata_mode: str,
    ) -> tuple[CUDAGraphMode, BatchDescriptor] | None:
        """Dispatch an extended (offset / variant-aware) key.

        Args:
            dispatcher: The calling ``CudagraphDispatcher`` instance; the
                strategy reads ``cudagraph_mode`` / ``cudagraph_keys`` and
                registers extra keys via ``add_cudagraph_key``.
            start_num_tokens: Starting token offset for an offset view.
                ``0`` means the key is not an offset key.
            allow_inplace_lazy_key: Whether lazy capture of a missing offset /
                variant key is allowed. When ``False`` the strategy fails
                closed (returns ``CUDAGraphMode.NONE`` with a no-graph
                descriptor carrying the extended metadata).
            graph_variant / attention_backend / capture_metadata_mode:
                Variant and backend tags distinguishing descriptor-aware keys
                from standard keys with the same ``num_tokens``.
        """
        ...


class InplaceSplitKeyStrategy:
    """Default strategy replicating the historical inline DUAL_INPLACE key
    policy.

    Transitional implementation currently living in core so that existing
    split-batch callers keep working unchanged; platforms should move this
    policy out of tree via ``Platform.get_cudagraph_key_strategy()`` and core
    can then drop this class.
    """

    def _create_inplace_offset_batch_descriptor(
        self,
        dispatcher: Any,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        start_num_tokens: int,
        graph_variant: str = "",
        attention_backend: str = "",
        capture_metadata_mode: str = "",
    ) -> BatchDescriptor:
        """Create a BatchDescriptor for inplace offset graph keys.

        Inplace offset keys are used by DUAL_INPLACE split-batch mode where
        the second split replays from an offset view into the original buffer.
        Unlike normal padded descriptors, offset descriptors use the exact
        num_tokens (no padding lookup) and require uniform decode with
        request-aligned token counts.
        """
        if num_tokens <= 0:
            raise ValueError(
                "num_tokens must be positive for inplace offset keys")
        if start_num_tokens <= 0:
            raise ValueError(
                "start_num_tokens must be positive for inplace offset keys")
        if not uniform_decode:
            raise ValueError(
                "inplace offset keys only support uniform decode")
        if num_tokens % dispatcher.uniform_decode_query_len != 0:
            raise ValueError(
                "num_tokens must be divisible by uniform_decode_query_len "
                "for inplace offset keys")

        return BatchDescriptor(
            num_tokens=num_tokens,
            num_reqs=num_tokens // dispatcher.uniform_decode_query_len,
            uniform=True,
            has_lora=has_lora,
            start_num_tokens=start_num_tokens,
            graph_variant=graph_variant,
            attention_backend=attention_backend,
            capture_metadata_mode=capture_metadata_mode,
        )

    def _create_descriptor_aware_batch_descriptor(
        self,
        dispatcher: Any,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        *,
        num_active_loras: int = 0,
        graph_variant: str = "",
        attention_backend: str = "",
        capture_metadata_mode: str = "",
    ) -> BatchDescriptor:
        """Create a BatchDescriptor for descriptor-aware graph keys.

        Descriptor-aware keys carry variant/backend metadata that
        differentiates them from normal graph keys with the same num_tokens.
        Unlike inplace offset keys, they do not require start_num_tokens.
        """
        if num_tokens <= 0:
            raise ValueError(
                "num_tokens must be positive for descriptor-aware keys")
        num_reqs: int | None = None
        if uniform_decode:
            if num_tokens % dispatcher.uniform_decode_query_len != 0:
                raise ValueError(
                    "num_tokens must be divisible by "
                    "uniform_decode_query_len for uniform descriptor-aware "
                    "keys")
            num_reqs = num_tokens // dispatcher.uniform_decode_query_len

        return BatchDescriptor(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            graph_variant=graph_variant,
            attention_backend=attention_backend,
            capture_metadata_mode=capture_metadata_mode,
        )

    def dispatch(
        self,
        *,
        dispatcher: Any,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int,
        start_num_tokens: int,
        allow_inplace_lazy_key: bool,
        graph_variant: str,
        attention_backend: str,
        capture_metadata_mode: str,
    ) -> tuple[CUDAGraphMode, BatchDescriptor] | None:
        """Try to dispatch an extended key.

        Returns None if the key carries no extended metadata (standard key),
        otherwise returns the dispatch result.
        """
        # Inplace offset keys take precedence over descriptor-aware keys.
        if start_num_tokens > 0:
            no_graph_desc = BatchDescriptor(
                num_tokens=num_tokens,
                start_num_tokens=start_num_tokens,
                graph_variant=graph_variant,
                attention_backend=attention_backend,
                capture_metadata_mode=capture_metadata_mode,
            )
            if not allow_inplace_lazy_key:
                return CUDAGraphMode.NONE, no_graph_desc

            runtime_mode = dispatcher.cudagraph_mode.decode_mode()
            if runtime_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
                return CUDAGraphMode.NONE, no_graph_desc

            batch_desc = self._create_inplace_offset_batch_descriptor(
                dispatcher,
                num_tokens=num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
                start_num_tokens=start_num_tokens,
                graph_variant=graph_variant,
                attention_backend=attention_backend,
                capture_metadata_mode=capture_metadata_mode,
            )
            if batch_desc not in dispatcher.cudagraph_keys[runtime_mode]:
                dispatcher.add_cudagraph_key(runtime_mode, batch_desc)
            return runtime_mode, batch_desc

        # Handle descriptor-aware keys (non-offset keys with variant/backend)
        has_descriptor_variant = bool(
            graph_variant or attention_backend or capture_metadata_mode)
        if has_descriptor_variant:
            descriptor = self._create_descriptor_aware_batch_descriptor(
                dispatcher,
                num_tokens=num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
                graph_variant=graph_variant,
                attention_backend=attention_backend,
                capture_metadata_mode=capture_metadata_mode,
            )
            if not allow_inplace_lazy_key:
                return CUDAGraphMode.NONE, descriptor

            runtime_mode = dispatcher.cudagraph_mode.decode_mode()
            if runtime_mode not in (CUDAGraphMode.FULL,
                                    CUDAGraphMode.PIECEWISE):
                return CUDAGraphMode.NONE, descriptor
            if descriptor not in dispatcher.cudagraph_keys[runtime_mode]:
                dispatcher.add_cudagraph_key(runtime_mode, descriptor)
            return runtime_mode, descriptor

        return None


def resolve_cudagraph_key_strategy(vllm_config: Any) -> CudagraphKeyStrategy | None:
    """Resolve the extended-key strategy for the current platform.

    Consults ``Platform.get_cudagraph_key_strategy`` first so platforms can
    own the policy out of tree. Falls back to the transitional core default
    (which replicates the historical inline behavior). Returns None only if
    the platform explicitly opts out and the default is disabled; extended
    keys then fail closed through the standard dispatch path.
    """
    from vllm.platforms import current_platform

    hook = getattr(current_platform, "get_cudagraph_key_strategy", None)
    if hook is not None:
        strategy = hook(vllm_config)
        if strategy is not None:
            return strategy

    # Transitional default: replicate the historical inline policy so that
    # existing split-batch callers observe zero behavior change.
    return InplaceSplitKeyStrategy()