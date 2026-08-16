# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from transformers import OPTConfig

from tests.v1.core.utils import create_requests, create_scheduler


def _local_opt_model(tmp_path: Path) -> str:
    model_path = tmp_path / "tiny-opt"
    OPTConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        ffn_dim=128,
        num_attention_heads=1,
        max_position_embeddings=256,
    ).save_pretrained(model_path)
    return str(model_path)


def test_capacity_blocked_head_preserves_fcfs_while_running(tmp_path: Path) -> None:
    """Running work may release capacity, so an infeasible head stays first."""
    scheduler = create_scheduler(
        model=_local_opt_model(tmp_path),
        num_blocks=8,
        max_num_batched_tokens=128,
        max_model_len=128,
        enable_chunked_prefill=False,
        skip_tokenizer_init=True,
        device="cpu",
    )
    (running,) = create_requests(
        num_requests=1,
        num_tokens=32,
        max_tokens=4,
        req_ids=["running"],
    )
    scheduler.add_request(running)
    scheduler.schedule()

    large = create_requests(
        num_requests=1,
        num_tokens=96,
        req_ids=["large"],
    )[0]
    small = create_requests(
        num_requests=1,
        num_tokens=16,
        req_ids=["small"],
    )[0]
    scheduler.add_request(large)
    scheduler.add_request(small)

    output = scheduler.schedule()

    assert output.scheduled_new_reqs == []
    assert [request.request_id for request in scheduler.waiting] == [
        "large",
        "small",
    ]


def test_capacity_blocked_head_is_scanned_when_nothing_can_release(
    tmp_path: Path,
) -> None:
    """The HUST anti-wedge policy scans the queue when no work is running."""
    scheduler = create_scheduler(
        model=_local_opt_model(tmp_path),
        num_blocks=5,
        max_num_batched_tokens=128,
        max_model_len=128,
        enable_chunked_prefill=False,
        skip_tokenizer_init=True,
        device="cpu",
    )
    large = create_requests(
        num_requests=1,
        num_tokens=96,
        req_ids=["large"],
    )[0]
    small = create_requests(
        num_requests=1,
        num_tokens=16,
        req_ids=["small"],
    )[0]
    scheduler.add_request(large)
    scheduler.add_request(small)

    output = scheduler.schedule()

    assert [request.req_id for request in output.scheduled_new_reqs] == ["small"]
    queued = list(scheduler.skipped_waiting) + list(scheduler.waiting)
    assert [request.request_id for request in queued] == ["large"]
