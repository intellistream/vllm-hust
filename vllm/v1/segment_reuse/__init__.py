"""Segment reuse (minimal stitch) for vLLM v1.

This module implements body-aware KV cache reuse for structured prompts
that follow the envelope||body decomposition pattern. Instead of relying
solely on prefix caching (which requires front-aligned token matches),
this module provides:

- Body block registry: pins body KV blocks across requests
- Stitch-aware prefix lookup: combines prefix cache hits on E with
  body registry hits on B
- Two-span prefill: only prefills the fresh envelope tokens while
  reusing the body's cached KV blocks

Enable via environment variable:
    VLLM_SEGMENT_REUSE=stitch
"""

from vllm.v1.segment_reuse.types import (
    EnvelopeBodySpec,
    BodyBlockEntry,
    StitchResult,
    StitchState,
)
from vllm.v1.segment_reuse.body_registry import BodyBlockRegistry

__all__ = [
    "EnvelopeBodySpec",
    "BodyBlockEntry",
    "StitchResult",
    "StitchState",
    "BodyBlockRegistry",
]
