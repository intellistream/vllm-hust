# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="dummy_stat_logger",
    version="0.1",
    packages=["dummy_stat_logger"],
    package_data={"dummy_stat_logger": ["manifests/*.json"]},
    entry_points={
        "vllm.stat_logger_plugins": [
            "dummy_stat_logger = dummy_stat_logger.dummy_stat_logger:DummyStatLogger"  # noqa
        ],
        "vllm.extension_bundles": [
            "org.vllm-hust.dummy-stat-logger = dummy_stat_logger.manifests"
        ],
    },
)
