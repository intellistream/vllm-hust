# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="vllm-b134-events",
    version="0.1.0",
    description="B134 KV-tiering benchmark JSONL event sink (vLLM plugin)",
    packages=["vllm_b134_events"],
    python_requires=">=3.10",
    entry_points={
        "vllm.general_plugins": [
            "register_b134_events = vllm_b134_events:register"
        ]
    },
)
