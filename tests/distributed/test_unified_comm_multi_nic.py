# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

from vllm.distributed.unified_comm.backend import CommConfig
from vllm.distributed.unified_comm.backends.nccl_backend import NCCLBackend
from vllm.distributed.unified_comm.collective import CollectiveGroup

MULTI_NIC_ENV_VARS = (
    "NCCL_CROSS_NIC",
    "NCCL_IB_HCA",
    "NCCL_IB_QPS_PER_CONNECTION",
    "UNIFIED_COMM_NCCL_MULTI_NIC_COUNT",
    "UNIFIED_COMM_NCCL_MULTI_NIC_DEVICES",
    "UNIFIED_COMM_NCCL_MULTI_NIC_ENABLE",
    "UNIFIED_COMM_NCCL_MULTI_NIC_FORCE",
    "UNIFIED_COMM_NCCL_MULTI_NIC_QPS_PER_CONNECTION",
)


def _clear_multi_nic_env(monkeypatch) -> None:
    for name in MULTI_NIC_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_multi_nic_explicit_devices_are_normalized_and_limited(monkeypatch) -> None:
    _clear_multi_nic_env(monkeypatch)
    backend = NCCLBackend()
    config = CommConfig(
        extra={
            "multi_nic_devices": " mlx5_2,mlx5_0, mlx5_1 ",
            "multi_nic_count": 2,
        }
    )

    backend._maybe_enable_multi_nic_aggregation(config)

    assert os.environ["NCCL_IB_HCA"] == "mlx5_2,mlx5_0"
    assert os.environ["NCCL_CROSS_NIC"] == "1"
    assert os.environ["NCCL_IB_QPS_PER_CONNECTION"] == "4"


def test_multi_nic_preserves_existing_nccl_selection(monkeypatch) -> None:
    _clear_multi_nic_env(monkeypatch)
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_7:1")

    NCCLBackend()._maybe_enable_multi_nic_aggregation(
        CommConfig(extra={"multi_nic_devices": "mlx5_0,mlx5_1"})
    )

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_7:1"
    assert "NCCL_CROSS_NIC" not in os.environ


def test_multi_nic_force_override_and_invalid_numeric_values(monkeypatch) -> None:
    _clear_multi_nic_env(monkeypatch)
    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_old")
    config = CommConfig(
        extra={
            "multi_nic_devices": [f"mlx5_{index}" for index in range(10)],
            "multi_nic_force": True,
            "multi_nic_count": "invalid",
            "multi_nic_qps_per_connection": "invalid",
        }
    )

    NCCLBackend()._maybe_enable_multi_nic_aggregation(config)

    assert os.environ["NCCL_IB_HCA"] == ",".join(f"mlx5_{index}" for index in range(8))
    assert os.environ["NCCL_IB_QPS_PER_CONNECTION"] == "4"


def test_discover_rdma_hcas_uses_infiniband_device_names(tmp_path: Path) -> None:
    for name in ("mlx5_2", "mlx5_bond_0", "mlx5_0"):
        (tmp_path / name).mkdir()
    (tmp_path / "not-a-device").write_text("", encoding="utf-8")

    assert NCCLBackend()._discover_rdma_hcas(tmp_path) == [
        "mlx5_bond_0",
        "mlx5_0",
        "mlx5_2",
    ]


def test_topology_nic_detection_parses_exact_hcas_and_ports(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_1:1,mlx5_0,mlx5_1:2")

    assert CollectiveGroup._detect_inter_node_nics(tmp_path) == [
        "mlx5_0",
        "mlx5_1",
    ]


def test_topology_nic_detection_falls_back_for_exclusion_syntax(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NCCL_IB_HCA", "^mlx5_3")
    (tmp_path / "mlx5_0").mkdir()
    (tmp_path / "mlx5_1").mkdir()

    assert CollectiveGroup._detect_inter_node_nics(tmp_path) == [
        "mlx5_0",
        "mlx5_1",
    ]
