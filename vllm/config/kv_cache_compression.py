# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight configuration for optional KV cache compression providers."""

import json
import math
from dataclasses import field
from typing import TypeAlias

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

JsonScalar: TypeAlias = str | int | float | bool | None


@config
class KVCacheCompressionConfig:
    """Configuration for an optional KV cache compression provider.

    Provider-specific options intentionally remain opaque to vLLM core. The
    selected platform resolves and validates them before KV cache allocation.
    """

    schema_version: int = 1
    """Version of the core/provider configuration contract."""

    provider: str = ""
    """Registered provider name. An empty name is invalid when enabled."""

    provider_config: dict[str, JsonScalar] = field(default_factory=dict)
    """Flat, JSON-scalar provider options validated by the provider."""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "Unsupported KV cache compression schema_version "
                f"{self.schema_version}; expected 1"
            )
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("KV cache compression provider must be a non-empty string")

        for key, value in self.provider_config.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "KV cache compression provider_config keys must be "
                    "non-empty strings"
                )
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(
                    "KV cache compression provider_config values must be "
                    f"JSON scalars; key {key!r} has {type(value).__name__}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    "KV cache compression provider_config values must be "
                    f"finite; key {key!r} has {value!r}"
                )

    def compute_hash(self) -> str:
        """Return a stable hash for enabled compression configuration."""
        payload = {
            "provider": self.provider,
            "provider_config": self.provider_config,
            "schema_version": self.schema_version,
        }
        serialized = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return safe_hash(serialized.encode(), usedforsecurity=False).hexdigest()
