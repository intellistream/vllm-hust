from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler


def test_chunk_budget_isolates_decode_from_long_prefill():
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=32,
        enable_chunked_prefill=True,
    )
    decode_req, prefill_req = create_requests(
        num_requests=2, num_tokens=64, req_ids=["decode", "prefill"]
    )
    scheduler.add_request(decode_req)

    for step in range(2):
        output = scheduler.schedule()
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["decode"],
                req_id_to_index={"decode": 0},
                sampled_token_ids=[[0] if step == 1 else []],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    scheduler.add_request(prefill_req)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["decode"] == 1
    assert output.num_scheduled_tokens["prefill"] == 32
    assert sum(output.num_scheduled_tokens.values()) <= 64
