# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

_QBI_SCHEDULER_MECHANISM = "scheduler_batching"
_QBI_SCHEDULER_FIELDS = {"max_num_scheduled_tokens", "max_num_running_reqs"}


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _qbi_utility(request: Request):
    utility = getattr(engine_client(request), "call_utility_async", None)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    return utility


async def _qbi_json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {error}",
        ) from error
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Request body must be a JSON object",
        )
    return body


def _qbi_scheduler_config(body: dict[str, Any]) -> dict[str, int]:
    if body.get("mechanism_name") != _QBI_SCHEDULER_MECHANISM:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only scheduler_batching is supported",
        )
    config = body.get("config")
    if not isinstance(config, dict) or set(config) != _QBI_SCHEDULER_FIELDS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Scheduler config has unexpected or missing fields",
        )
    try:
        return {field: int(config[field]) for field in sorted(_QBI_SCHEDULER_FIELDS)}
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Scheduler config values must be integers",
        ) from error


async def _qbi_call(request: Request, method: str, *args: Any) -> JSONResponse:
    try:
        result = await _qbi_utility(request)(method, *args)
    except (RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Engine returned a non-object runtime transition receipt",
        )
    return JSONResponse(content=result)


@router.post("/collective_rpc")
async def collective_rpc(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    method = body.get("method")
    if method is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'method' in request body",
        )
    # For security reason, only serialized string args/kwargs are passed.
    # User-defined `method` is responsible for deserialization if needed.
    args: list[str] = body.get("args", [])
    kwargs: dict[str, str] = body.get("kwargs", {})
    timeout: float | None = body.get("timeout")
    results = await engine_client(raw_request).collective_rpc(
        method=method, timeout=timeout, args=tuple(args), kwargs=kwargs
    )
    if results is None:
        return Response(status_code=200)
    response: list[Any] = []
    for result in results:
        if result is None or isinstance(result, dict | list):
            response.append(result)
        else:
            response.append(str(result))
    return JSONResponse(content={"results": response})


@router.get("/qbi/native-recapture/scope")
async def native_recapture_scope(raw_request: Request, after_sequence: int = 0):
    """Read prompt-free scheduler receipts from the isolated control surface."""

    utility = getattr(engine_client(raw_request), "call_utility_async", None)
    if not callable(utility):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Engine client does not expose utility calls",
        )
    try:
        result = await utility("get_native_recapture_scope_state", after_sequence)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(error),
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Engine returned a non-object native recapture state",
        )
    return JSONResponse(content=result)


@router.get("/qbi/runtime-transition/state")
async def qbi_runtime_transition_state(raw_request: Request):
    """Return the live EngineCore epoch and scheduler-limit identity."""

    return await _qbi_call(raw_request, "get_runtime_transition_state")


@router.post("/qbi/runtime-transition/stage")
async def qbi_runtime_transition_stage(raw_request: Request):
    """Stage one bounded scheduler-limit change without accepting mixed epochs."""

    body = await _qbi_json_body(raw_request)
    config = _qbi_scheduler_config(body)
    return await _qbi_call(
        raw_request,
        "stage_runtime_transition",
        _QBI_SCHEDULER_MECHANISM,
        config,
    )


@router.post("/qbi/runtime-transition/commit")
async def qbi_runtime_transition_commit(raw_request: Request):
    """Commit a staged change after the previous scheduler epoch drains."""

    body = await _qbi_json_body(raw_request)
    if body.get("mechanism_name") != _QBI_SCHEDULER_MECHANISM:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only scheduler_batching is supported",
        )
    try:
        next_epoch = int(body["next_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="next_epoch must be an integer",
        ) from error
    return await _qbi_call(
        raw_request,
        "commit_staged_runtime_transition",
        _QBI_SCHEDULER_MECHANISM,
        next_epoch,
    )


@router.post("/qbi/runtime-transition/abort")
async def qbi_runtime_transition_abort(raw_request: Request):
    """Abort an uncommitted scheduler transition and release pending requests."""

    body = await _qbi_json_body(raw_request)
    if body.get("mechanism_name") != _QBI_SCHEDULER_MECHANISM:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only scheduler_batching is supported",
        )
    return await _qbi_call(
        raw_request,
        "abort_staged_runtime_transition",
        _QBI_SCHEDULER_MECHANISM,
    )


@router.post("/qbi/runtime-transition/verify")
async def qbi_runtime_transition_verify(raw_request: Request):
    """Return the EngineCore effectiveness receipt for a committed epoch."""

    body = await _qbi_json_body(raw_request)
    config = _qbi_scheduler_config(body)
    try:
        epoch = int(body["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="epoch must be an integer",
        ) from error
    return await _qbi_call(
        raw_request,
        "verify_runtime_transition",
        _QBI_SCHEDULER_MECHANISM,
        config,
        epoch,
    )


@router.post("/qbi/runtime-transition/rollback")
async def qbi_runtime_transition_rollback(raw_request: Request):
    """Restore the staged transition's previous limits at an idle boundary."""

    body = await _qbi_json_body(raw_request)
    if body.get("mechanism_name") != _QBI_SCHEDULER_MECHANISM:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only scheduler_batching is supported",
        )
    prepared = body.get("prepared")
    if not isinstance(prepared, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="prepared transition receipt must be an object",
        )
    try:
        previous_epoch = int(body["previous_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="previous_epoch must be an integer",
        ) from error
    return await _qbi_call(
        raw_request,
        "rollback_runtime_transition",
        _QBI_SCHEDULER_MECHANISM,
        prepared,
        previous_epoch,
    )


def attach_router(app: FastAPI):
    app.include_router(router)
