# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter()

logger = init_logger(__name__)
LOAD_METRICS_TIMEOUT_S = 1.0


def base(request: Request) -> ServingTokenization:
    # Reuse the existing instance
    return tokenization(request)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/responses
    # - /v1/responses/{response_id}
    # - /v1/responses/{response_id}/cancel
    # - /v1/messages
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/audio/translations
    # - /v1/embeddings
    # - /pooling
    # - /classify
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    current_engine_client = request.app.state.engine_client
    if current_engine_client is None:
        return JSONResponse(
            content={"server_load": request.app.state.server_load_metrics}
        )

    rank_header = request.headers.get("x-data-parallel-rank")
    data_parallel_rank: int | None = None

    if rank_header is not None:
        try:
            data_parallel_rank = int(rank_header)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="x-data-parallel-rank must be an integer",
            ) from exc

        if data_parallel_rank < 0:
            raise HTTPException(
                status_code=400,
                detail="x-data-parallel-rank must be non-negative",
            )

    try:
        engine_load_metrics = await asyncio.wait_for(
            current_engine_client.get_load_metrics(data_parallel_rank),
            timeout=LOAD_METRICS_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="timed out while fetching EngineCore load metrics",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        content={
            "server_load": request.app.state.server_load_metrics,
            **engine_load_metrics,
        }
    )


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)
