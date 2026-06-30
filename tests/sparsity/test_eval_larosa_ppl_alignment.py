# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_larosa_ppl_alignment import HFLarosaReference, run_hf_reference  # noqa: E402


class _FakeLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(
            q_proj=nn.Linear(4, 4, bias=False),
            k_proj=nn.Linear(4, 4, bias=False),
            v_proj=nn.Linear(4, 4, bias=False),
            o_proj=nn.Linear(4, 4, bias=False),
        )
        self.mlp = SimpleNamespace(
            gate_proj=nn.Linear(4, 4, bias=False),
            up_proj=nn.Linear(4, 4, bias=False),
            down_proj=nn.Linear(4, 4, bias=False),
        )


class _FakeModel(nn.Module):

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            layers=nn.ModuleList(_FakeLayer() for _ in range(num_layers))
        )


def _write_larosa_rotation(root: Path, layer_idx: int) -> None:
    path = root / "histograms" / f"layer-{layer_idx}" / "self_attn"
    path.mkdir(parents=True)
    torch.save(torch.eye(4), path / "D.pt")


def test_hf_larosa_reference_honors_target_layers(tmp_path: Path) -> None:
    model = _FakeModel(num_layers=3)
    _write_larosa_rotation(tmp_path, layer_idx=1)

    reference = HFLarosaReference(
        model,
        tmp_path,
        torch.float32,
        target_projections=["mlp.gate_up"],
        target_layers=[1],
        prefill_sparsify="all",
    )

    assert sorted(reference.rotations) == [1]
    assert len(reference.handles) == 2


def test_hf_larosa_prefill_none_keeps_prompt_dense(tmp_path: Path) -> None:
    model = _FakeModel(num_layers=1)
    _write_larosa_rotation(tmp_path, layer_idx=0)
    reference = HFLarosaReference(
        model,
        tmp_path,
        torch.float32,
        target_projections=["mlp.gate_up"],
        target_layers=None,
        prefill_sparsify="none",
    )

    x = torch.ones(2, 3, 4)
    sparse_x = torch.zeros_like(x)
    decode_x = torch.ones(2, 1, 4)
    sparse_decode_x = torch.zeros_like(decode_x)

    assert reference._apply_prefill_policy(x, sparse_x) is x
    assert (
        reference._apply_prefill_policy(decode_x, sparse_decode_x)
        is sparse_decode_x
    )


def test_official_repo_rejects_selective_layer_targeting() -> None:
    args = SimpleNamespace(
        hf_reference_impl="official_repo",
        target_projection=None,
        target_layer=[1],
        vllm_prefill_sparsify="all",
    )

    with pytest.raises(ValueError, match="target-layer"):
        run_hf_reference(args, windows=[], model_type="qwen2")
