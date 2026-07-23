# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned, device-independent KV cache compression contracts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config.kv_cache_compression import KVCacheCompressionConfig

KV_CACHE_COMPRESSION_SCHEMA_VERSION = 1


class KVCacheCompressionError(RuntimeError):
    """Raised before KV allocation when compression is not compatible."""


@dataclass(frozen=True)
class KVCacheCompressionCompatibility:
    """Serializable compatibility result returned by one worker."""

    schema_version: int
    provider: str
    supported: bool
    reasons: tuple[str, ...]
    platform: str
    provider_factory: str | None = None
    backend: str | None = None
    model_architecture: str | None = None
    dtype: str | None = None
    cache_layout: str | None = None
    block_size: int | None = None


@dataclass(frozen=True)
class KVCacheCompressionPlan:
    """Worker-to-scheduler request compaction transaction."""

    schema_version: int
    provider: str
    request_id: str
    semantic_num_tokens: int
    physical_num_tokens: int
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...]
    expected_block_ids: tuple[tuple[int, ...], ...]
    kv_cache_group_id: int = 0


def ensure_kv_cache_compression_compatible(
    config: "KVCacheCompressionConfig",
    reports: list[KVCacheCompressionCompatibility],
) -> None:
    """Validate every worker report and aggregate all incompatibilities."""
    errors: list[str] = []
    if not reports:
        errors.append("no worker compatibility reports were returned")

    for rank, report in enumerate(reports):
        prefix = f"worker {rank} ({report.platform})"
        if report.schema_version != config.schema_version:
            errors.append(
                f"{prefix}: schema_version {report.schema_version} does not "
                f"match requested {config.schema_version}"
            )
        if report.provider != config.provider:
            errors.append(
                f"{prefix}: provider {report.provider!r} does not match "
                f"requested {config.provider!r}"
            )
        if report.supported and report.reasons:
            errors.append(
                f"{prefix}: supported report unexpectedly contains reasons: "
                + "; ".join(report.reasons)
            )
        if not report.supported:
            if report.reasons:
                errors.extend(f"{prefix}: {reason}" for reason in report.reasons)
            else:
                errors.append(f"{prefix}: provider is unsupported")

    if errors:
        details = "\n - ".join(errors)
        raise KVCacheCompressionError(
            "KV cache compression compatibility check failed for provider "
            f"{config.provider!r}:\n - {details}"
        )
