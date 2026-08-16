# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI API prefix-routing proxy integration."""

import asyncio
import hashlib
import inspect
import json
import os
import secrets
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp
import msgspec
import zmq
import zmq.asyncio
from fastapi import APIRouter, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import Response
from starlette.datastructures import URL, Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.distributed.kv_events import (
    KVEventBatch,
    ZmqEventReplayRequest,
    ZmqEventReplayResponse,
)
from vllm.distributed.prefix_scheduler import (
    GlobalPrefixScheduler,
    PrefixRouteDecision,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request as VllmRequest

logger = init_logger(__name__)

PREFIX_ROUTING_BYPASS_HEADER = "x-vllm-prefix-routing"
DATA_PARALLEL_RANK_HEADER = "x-data-parallel-rank"
WORKER_INCARNATION_HEADER = "x-vllm-prefix-routing-worker-incarnation"
ROUTE_VALIDATION_MODE_NATIVE_CONTROL = "native-control"
ROUTE_VALIDATION_MODE_GENERATION_SCOPED = "generation-scoped"
ROUTE_VALIDATION_MODES = frozenset(
    {ROUTE_VALIDATION_MODE_NATIVE_CONTROL, ROUTE_VALIDATION_MODE_GENERATION_SCOPED}
)
PREFIX_ROUTING_VALIDATION_MODE_HEADER = "x-vllm-prefix-routing-validation-mode"
EXPECTED_NODE_ID_HEADER = "x-vllm-prefix-routing-expected-node-id"
EXPECTED_GENERATION_HEADER = "x-vllm-prefix-routing-expected-generation"
EXPECTED_WORKER_INCARNATION_HEADER = (
    "x-vllm-prefix-routing-expected-worker-incarnation"
)
FENCE_RESULT_HEADER = "x-vllm-prefix-routing-fence"
FENCE_RESULT_REJECTED = "rejected"
ACK_GENERATION_HEADER = "x-vllm-prefix-routing-ack-generation"
ACK_WORKER_INCARNATION_HEADER = "x-vllm-prefix-routing-ack-worker-incarnation"
ROUTE_RESULT_HEADER = "x-vllm-prefix-routing-result"
ROUTE_SELECTED_NODE_HEADER = "x-vllm-prefix-routing-selected-node"
ROUTE_MATCHED_TOKENS_HEADER = "x-vllm-prefix-routing-matched-tokens"
ROUTE_GENERATION_HEADER = "x-vllm-prefix-routing-generation"
ROUTE_WORKER_INCARNATION_HEADER = "x-vllm-prefix-routing-worker-incarnation-result"
ROUTE_RESULT_LOCAL_NO_ROUTE = "local-no-route"
ROUTE_RESULT_LOCAL_NODE = "local-node"
ROUTE_RESULT_REMOTE = "remote"
ROUTE_RESULT_LOCAL_FALLBACK = "local-fallback"
REQUEST_ID_HEADERS = (
    "x-request-id",
    "x-correlation-id",
    "x-openai-request-id",
)
DEFAULT_MAX_REQUEST_BODY_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_EVENT_INGEST_BODY_SIZE = 16 * 1024 * 1024

router = APIRouter()


@router.post("/prefix_routing/kv_events/{node_id}")
async def ingest_prefix_routing_kv_events(
    node_id: str, raw_request: FastAPIRequest
) -> Response:
    """Receive worker KV-cache events and update the global prefix index."""
    proxy = getattr(raw_request.app.state, "prefix_routing_proxy", None)
    if proxy is None:
        raise HTTPException(status_code=503, detail="Prefix routing is not enabled")

    expected_token = proxy.config.event_ingest_token
    if expected_token is None:
        raise HTTPException(
            status_code=503, detail="Prefix routing HTTP event ingestion is disabled"
        )
    authorization = raw_request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    token_matches = scheme.lower() == "bearer" and secrets.compare_digest(
        hashlib.sha256(token.encode("utf-8")).digest(),
        hashlib.sha256(expected_token.encode("utf-8")).digest(),
    )
    if not token_matches:
        raise HTTPException(
            status_code=401,
            detail="Invalid prefix routing event credential",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if node_id not in proxy.nodes:
        raise HTTPException(status_code=404, detail=f"Unknown prefix node {node_id!r}")

    try:
        body = await _read_limited_fastapi_body(
            raw_request, proxy.config.max_event_ingest_body_size
        )
    except RequestBodyTooLarge as exc:
        raise HTTPException(
            status_code=413, detail="KV event batch is too large"
        ) from exc
    try:
        batch = msgspec.msgpack.decode(body, type=KVEventBatch)
    except msgspec.DecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid KV event batch") from exc

    worker_incarnation = raw_request.headers.get(WORKER_INCARNATION_HEADER)
    if worker_incarnation == "":
        worker_incarnation = None
    try:
        proxy.scheduler.apply_event_batch(
            node_id, batch, worker_incarnation=worker_incarnation
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="KV event rank does not match the configured prefix node",
        ) from exc
    ack_headers: dict[str, str] = {}
    get_generation = getattr(proxy.scheduler, "get_node_route_generation", None)
    if get_generation is not None:
        generation = get_generation(node_id, batch.data_parallel_rank)
        if generation is not None:
            view_epoch, ack_incarnation = generation
            ack_headers[ACK_GENERATION_HEADER] = str(view_epoch)
            if ack_incarnation is not None:
                ack_headers[ACK_WORKER_INCARNATION_HEADER] = ack_incarnation
    return Response(status_code=204, headers=ack_headers)


def attach_prefix_routing_router(app: Any) -> None:
    app.include_router(router)


@dataclass(frozen=True)
class PrefixRoutingNode:
    node_id: str
    url: str | None
    event_endpoint: str | None = None
    replay_endpoint: str | None = None
    data_parallel_rank: int | None = None
    routing_token: str | None = None
    local: bool = False


@dataclass(frozen=True)
class PrefixRoutingConfig:
    nodes: list[PrefixRoutingNode]
    hash_block_size: int
    event_topic: str = ""
    request_timeout: float = 6 * 60 * 60
    event_ingest_token: str | None = None
    routing_token: str | None = None
    event_replay_timeout: float = 2.0
    event_sync_interval: float = 5.0
    view_ttl_seconds: float | None = 15.0
    idempotency_cache_size: int = 4096
    max_request_body_size: int = DEFAULT_MAX_REQUEST_BODY_SIZE
    max_event_ingest_body_size: int = DEFAULT_MAX_EVENT_INGEST_BODY_SIZE
    route_validation_mode: str = ROUTE_VALIDATION_MODE_GENERATION_SCOPED
    emit_response_headers: bool = False


@dataclass
class _ZmqRecoveryState:
    publisher_epoch: str | None = None
    next_seq: int | None = None


class PrefixRoutingProxy:
    def __init__(
        self,
        *,
        config: PrefixRoutingConfig,
        app_state: Any,
    ) -> None:
        self.config = config
        self.app_state = app_state
        self.scheduler = GlobalPrefixScheduler(
            view_ttl_seconds=config.view_ttl_seconds,
            strict_cache_generation=(
                config.route_validation_mode
                == ROUTE_VALIDATION_MODE_GENERATION_SCOPED
            ),
        )
        self.nodes = {node.node_id: node for node in config.nodes}
        self._tasks: list[asyncio.Task] = []
        self._session: aiohttp.ClientSession | None = None
        self._route_decision_cache: OrderedDict[str, PrefixRouteDecision | None] = (
            OrderedDict()
        )
        self._route_decision_locks: dict[str, asyncio.Lock] = {}

        vllm_config = app_state.vllm_config
        caching_hash_fn = get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )
        init_none_hash(caching_hash_fn)
        self._block_hasher = get_request_block_hasher(
            config.hash_block_size, caching_hash_fn
        )

        group_block_sizes = {0: config.hash_block_size}
        for node in config.nodes:
            self.scheduler.register_node(
                node.node_id,
                hash_block_size=config.hash_block_size,
                data_parallel_rank=node.data_parallel_rank,
                group_block_sizes=group_block_sizes,
            )

    def start(self) -> None:
        if self._session is None and any(
            not node.local and node.url is not None for node in self.nodes.values()
        ):
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            )
        for node in self.nodes.values():
            if node.event_endpoint is None:
                continue
            task = asyncio.create_task(self._subscribe_node_events(node))
            self._tasks.append(task)

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("prefix routing HTTP session is not initialized")
        return self._session

    async def choose_node_for_request(
        self, path: str, payload: Mapping[str, Any]
    ) -> PrefixRouteDecision | None:
        rendered = await self._render_request(path, payload)
        if rendered is None:
            return None
        engine_inputs, lora_request = rendered
        if not engine_inputs:
            return None

        supported_tasks = await self.app_state.engine_client.get_supported_tasks()

        best_decision: PrefixRouteDecision | None = None
        for engine_input in engine_inputs:
            cache_key_request = self._make_cache_key_request(
                engine_input,
                lora_request,
                supported_tasks,
            )
            decision = self.scheduler.choose_node(
                block_hashes=cache_key_request.block_hashes,
                prompt_num_tokens=cache_key_request.num_prompt_tokens,
            )
            if decision is None or decision.matched_tokens <= 0:
                continue
            if (
                best_decision is None
                or decision.matched_tokens > best_decision.matched_tokens
            ):
                best_decision = decision
        return best_decision

    async def choose_node_for_request_id(
        self,
        path: str,
        payload: Mapping[str, Any],
        request_id: str | None,
    ) -> PrefixRouteDecision | None:
        if request_id is None or self.config.idempotency_cache_size <= 0:
            return await self.choose_node_for_request(path, payload)

        lock = self._route_decision_locks.setdefault(request_id, asyncio.Lock())
        async with lock:
            if request_id in self._route_decision_cache:
                decision = self._route_decision_cache[request_id]
                self._route_decision_cache.move_to_end(request_id)
                if decision is not None and not self.scheduler.is_decision_current(
                    decision
                ):
                    self.record_route_decision(request_id, None)
                    return None
                return decision

            decision = await self.choose_node_for_request(path, payload)
            self.record_route_decision(request_id, decision)
            return decision

    def record_route_decision(
        self, request_id: str | None, decision: PrefixRouteDecision | None
    ) -> None:
        if request_id is None or self.config.idempotency_cache_size <= 0:
            return
        self._route_decision_cache[request_id] = decision
        self._route_decision_cache.move_to_end(request_id)
        while len(self._route_decision_cache) > self.config.idempotency_cache_size:
            evicted_request_id, _ = self._route_decision_cache.popitem(last=False)
            self._route_decision_locks.pop(evicted_request_id, None)

    async def _render_request(
        self, path: str, payload: Mapping[str, Any]
    ) -> tuple[list[EngineInput], LoRARequest | None] | None:
        if path == "/v1/completions":
            serving = getattr(self.app_state, "openai_serving_completion", None)
            if serving is None:
                return None
            completion_request = CompletionRequest.model_validate(payload)
            result = await serving.render_completion_request(completion_request)
            if isinstance(result, ErrorResponse):
                return None
            return result, serving._maybe_get_adapters(completion_request)
        if path == "/v1/chat/completions":
            serving = getattr(self.app_state, "openai_serving_chat", None)
            if serving is None:
                return None
            chat_request = ChatCompletionRequest.model_validate(payload)
            result = await serving.render_chat_request(chat_request)
            if isinstance(result, ErrorResponse):
                return None
            _, engine_inputs = result
            lora_request = serving._maybe_get_adapters(
                chat_request, supports_default_mm_loras=True
            )
            return engine_inputs, lora_request
        return None

    def _make_cache_key_request(
        self,
        engine_input: EngineInput,
        lora_request: LoRARequest | None,
        supported_tasks: tuple[SupportedTask, ...],
    ) -> VllmRequest:
        input_processor = self.app_state.engine_client.input_processor
        engine_core_request = input_processor.process_inputs(
            request_id="prefix-routing",
            prompt=engine_input,
            params=SamplingParams(max_tokens=1),
            supported_tasks=supported_tasks,
            lora_request=lora_request,
        )
        return VllmRequest.from_engine_core_request(
            engine_core_request,
            self._block_hasher,
        )

    async def _subscribe_node_events(self, node: PrefixRoutingNode) -> None:
        assert node.event_endpoint is not None
        ctx = zmq.asyncio.Context.instance()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        topic = self.config.event_topic.encode("utf-8")
        recovery_state = _ZmqRecoveryState()
        recovery_lock = asyncio.Lock()
        recovery_task: asyncio.Task | None = None
        reconnect_delay = 0.1
        try:
            while True:
                socket = ctx.socket(zmq.SUB)
                try:
                    socket.setsockopt(zmq.SUBSCRIBE, topic)
                    socket.connect(node.event_endpoint)
                    logger.info(
                        "Prefix routing subscribed to KV events from %s at %s",
                        node.node_id,
                        node.event_endpoint,
                    )
                    if node.replay_endpoint is not None:
                        async with recovery_lock:
                            await self._recover_zmq_events(
                                node,
                                recovery_state,
                                force_snapshot=recovery_task is None,
                            )
                        if recovery_task is None:
                            recovery_task = asyncio.create_task(
                                self._periodic_zmq_recovery(
                                    node,
                                    recovery_state,
                                    recovery_lock,
                                )
                            )

                    while True:
                        frames = await socket.recv_multipart()
                        reconnect_delay = 0.1
                        try:
                            await self._process_zmq_event_frames(
                                node,
                                recovery_state,
                                recovery_lock,
                                decoder,
                                frames,
                            )
                        except (msgspec.DecodeError, KeyError, ValueError) as exc:
                            logger.warning(
                                "Ignoring invalid KV event message from %s: %s",
                                node.node_id,
                                type(exc).__name__,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Prefix routing event subscriber disconnected from %s; "
                        "reconnecting",
                        node.node_id,
                    )
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 5.0)
                finally:
                    socket.close(linger=0)
        finally:
            if recovery_task is not None:
                recovery_task.cancel()
                await asyncio.gather(recovery_task, return_exceptions=True)

    async def _process_zmq_event_frames(
        self,
        node: PrefixRoutingNode,
        recovery_state: "_ZmqRecoveryState",
        recovery_lock: asyncio.Lock,
        decoder: Any,
        frames: list[bytes],
    ) -> None:
        if len(frames) != 3:
            logger.warning(
                "Ignoring malformed KV event message with %d frames",
                len(frames),
            )
            return
        _, seq_bytes, payload = frames
        if len(seq_bytes) != 8:
            raise ValueError("invalid ZMQ event sequence number")
        if node.replay_endpoint is None:
            batch = decoder.decode(payload)
            self.scheduler.apply_event_batch(node.node_id, batch)
            return
        seq = int.from_bytes(seq_bytes, "big")
        async with recovery_lock:
            await self._apply_live_zmq_event(
                node,
                recovery_state,
                seq,
                payload,
                decoder,
            )

    async def _apply_live_zmq_event(
        self,
        node: PrefixRoutingNode,
        state: "_ZmqRecoveryState",
        seq: int,
        payload: bytes,
        decoder: Any,
    ) -> None:
        if state.next_seq is None:
            # Recovery may be temporarily unavailable. Keep consuming live
            # events; the periodic control-plane sync will reconcile later.
            state.next_seq = seq

        if seq > state.next_seq:
            await self._recover_zmq_events(node, state)
        if seq < state.next_seq:
            return
        if seq > state.next_seq:
            logger.warning(
                "Deferring out-of-order KV event from %s: expected %d, got %d",
                node.node_id,
                state.next_seq,
                seq,
            )
            return

        try:
            batch = decoder.decode(payload)
            self.scheduler.apply_event_batch(
                node.node_id,
                batch,
                worker_incarnation=state.publisher_epoch,
            )
        except (msgspec.DecodeError, KeyError, ValueError):
            await self._recover_zmq_events(node, state, force_snapshot=True)
            raise
        state.next_seq += 1

    async def _periodic_zmq_recovery(
        self,
        node: PrefixRoutingNode,
        state: "_ZmqRecoveryState",
        recovery_lock: asyncio.Lock,
    ) -> None:
        while True:
            await asyncio.sleep(self.config.event_sync_interval)
            async with recovery_lock:
                await self._recover_zmq_events(node, state)

    async def _recover_zmq_events(
        self,
        node: PrefixRoutingNode,
        state: "_ZmqRecoveryState",
        *,
        force_snapshot: bool = False,
    ) -> bool:
        if node.replay_endpoint is None:
            return False

        ctx = zmq.asyncio.Context.instance()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(node.replay_endpoint)
        request = ZmqEventReplayRequest(
            publisher_epoch=state.publisher_epoch,
            start_seq=0 if state.next_seq is None else state.next_seq,
            force_snapshot=force_snapshot,
        )
        try:
            await socket.send(msgspec.msgpack.encode(request))
            payload = await asyncio.wait_for(
                socket.recv(),
                timeout=self.config.event_replay_timeout,
            )
            response = msgspec.msgpack.decode(payload, type=ZmqEventReplayResponse)
            self._apply_zmq_recovery_response(node, state, response)
            return True
        except (TimeoutError, msgspec.DecodeError, KeyError, ValueError, zmq.ZMQError):
            logger.warning(
                "ZMQ event recovery failed for %s; retaining last known state",
                node.node_id,
                exc_info=True,
            )
            return False
        finally:
            socket.close(linger=0)

    def _apply_zmq_recovery_response(
        self,
        node: PrefixRoutingNode,
        state: "_ZmqRecoveryState",
        response: ZmqEventReplayResponse,
    ) -> None:
        if response.next_seq < 0:
            raise ValueError("negative ZMQ recovery sequence")

        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        decoded: list[KVEventBatch] = []
        if response.snapshot is not None:
            if response.replayed_batches:
                raise ValueError("snapshot response also contains replay batches")
            decoded.append(decoder.decode(response.snapshot))
        else:
            if state.publisher_epoch != response.publisher_epoch:
                self.scheduler.invalidate_node(
                    node.node_id,
                    node.data_parallel_rank,
                    worker_incarnation=response.publisher_epoch,
                )
                raise ValueError("publisher epoch changed without a snapshot")
            expected_seq = 0 if state.next_seq is None else state.next_seq
            for seq, payload in response.replayed_batches:
                if seq != expected_seq:
                    raise ValueError(
                        f"non-contiguous replay: expected {expected_seq}, got {seq}"
                    )
                decoded.append(decoder.decode(payload))
                expected_seq += 1
            if expected_seq != response.next_seq:
                raise ValueError(
                    "replay response does not reach the advertised sequence"
                )

        for batch in decoded:
            self.scheduler.apply_event_batch(
                node.node_id,
                batch,
                worker_incarnation=response.publisher_epoch,
            )
        if not decoded:
            self.scheduler.record_worker_receipt(
                node.node_id,
                node.data_parallel_rank,
                worker_incarnation=response.publisher_epoch,
            )
        state.publisher_epoch = response.publisher_epoch
        state.next_seq = response.next_seq


class PrefixRoutingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        root_path = scope.get("root_path", "")
        path = URL(scope=scope).path.removeprefix(root_path)
        method = scope.get("method")
        headers = Headers(scope=scope)
        if method != "POST" or path not in ("/v1/completions", "/v1/chat/completions"):
            await self.app(scope, receive, send)
            return

        proxy = getattr(scope["app"].state, "prefix_routing_proxy", None)
        if proxy is None:
            await self.app(scope, receive, send)
            return
        emit_response_headers = bool(
            getattr(proxy.config, "emit_response_headers", False)
        )

        if _has_authenticated_bypass(headers, proxy.config.routing_token):
            if not await _validate_generation_scoped_forwarded_request(
                scope, headers, proxy
            ):
                response = Response(
                    content="Prefix routing generation fence rejected",
                    status_code=409,
                    headers={FENCE_RESULT_HEADER: FENCE_RESULT_REJECTED},
                    media_type="text/plain",
                )
                await response(scope, receive, send)
                return
            await self.app(_without_bypass_header(scope), receive, send)
            return

        scope = _without_external_routing_headers(scope)
        try:
            body = await _receive_body(receive, proxy.config.max_request_body_size)
        except RequestBodyTooLarge:
            response = Response(
                content="Prefix routing request body is too large",
                status_code=413,
                media_type="text/plain",
            )
            await response(scope, receive, send)
            return
        try:
            payload = json.loads(body)
            request_id = _extract_prefix_routing_request_id(headers, payload)
            decision = await _choose_node_for_request_id(
                proxy, path, payload, request_id
            )
            logger.info(
                "Prefix routing chose node=%s matched_tokens=%s",
                decision.node_id if decision else None,
                decision.matched_tokens if decision else None,
            )
        except Exception:
            logger.exception("Prefix routing failed; falling back to local handling")
            await self.app(
                scope,
                _replay_body(body),
                _with_response_headers(
                    send,
                    _routing_response_headers(
                        ROUTE_RESULT_LOCAL_FALLBACK,
                        None,
                    )
                    if emit_response_headers
                    else [],
                ),
            )
            return

        if decision is None:
            await self.app(
                scope,
                _replay_body(body),
                _with_response_headers(
                    send,
                    _routing_response_headers(
                        ROUTE_RESULT_LOCAL_NO_ROUTE,
                        None,
                    )
                    if emit_response_headers
                    else [],
                ),
            )
            return

        node = proxy.nodes.get(decision.node_id)
        if node is None or node.local or node.url is None:
            await self.app(
                _with_data_parallel_rank(scope, decision.data_parallel_rank),
                _replay_body(body),
                _with_response_headers(
                    send,
                    _routing_response_headers(
                        ROUTE_RESULT_LOCAL_NODE,
                        decision,
                        selected_node=decision.node_id,
                    )
                    if emit_response_headers
                    else [],
                ),
            )
            return

        forwarded = await _forward_request(
            scope,
            body,
            send,
            node,
            proxy.session,
            decision.data_parallel_rank,
            decision=decision,
            route_validation_mode=getattr(
                proxy.config,
                "route_validation_mode",
                ROUTE_VALIDATION_MODE_GENERATION_SCOPED,
            ),
            route_response_headers=(
                _routing_response_headers(
                    ROUTE_RESULT_REMOTE,
                    decision,
                    selected_node=decision.node_id,
                )
                if emit_response_headers
                else []
            ),
        )
        if not forwarded:
            _record_route_decision(proxy, request_id, None)
            logger.warning(
                "Prefix routing upstream %s failed before responding; "
                "falling back to local handling",
                node.node_id,
            )
            await self.app(
                scope,
                _replay_body(body),
                _with_response_headers(
                    send,
                    _routing_response_headers(
                        ROUTE_RESULT_LOCAL_FALLBACK,
                        decision,
                    )
                    if emit_response_headers
                    else [],
                ),
            )


async def init_prefix_routing(app_state: Any, args: Any) -> None:
    raw_config = getattr(args, "prefix_routing_config", None)
    if not getattr(args, "enable_prefix_routing", False):
        return
    config = _parse_prefix_routing_config(raw_config, app_state.vllm_config)
    if os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "Prefix routing is enabled but PYTHONHASHSEED is not set. "
            "Set the same PYTHONHASHSEED on every vLLM node and the routing "
            "master so prefix-cache block hashes are comparable across nodes."
        )
    proxy = PrefixRoutingProxy(config=config, app_state=app_state)
    app_state.prefix_routing_proxy = proxy
    proxy.start()
    logger.info("Prefix routing enabled with %d nodes", len(config.nodes))


def _parse_prefix_routing_config(
    raw_config: Mapping[str, Any] | str | None, vllm_config: Any
) -> PrefixRoutingConfig:
    if raw_config is None:
        raise ValueError("--enable-prefix-routing requires --prefix-routing-config")
    if isinstance(raw_config, str):
        raw_config = json.loads(raw_config)
    if not isinstance(raw_config, Mapping):
        raise ValueError("prefix routing config must be a JSON object")

    raw_nodes = raw_config.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise ValueError("prefix routing config 'nodes' must be a list")

    nodes: list[PrefixRoutingNode] = []
    node_ids: set[str] = set()
    for node in raw_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("prefix routing node config must be a JSON object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("prefix routing node 'id' must be a non-empty string")
        if node_id in node_ids:
            raise ValueError(f"duplicate prefix routing node id: {node_id!r}")
        node_ids.add(node_id)

        node_url = node.get("url")
        local = bool(node.get("local", node_url == "local"))
        if local:
            if node_url not in (None, "local"):
                raise ValueError(
                    f"local prefix routing node {node_id!r} cannot have a remote URL"
                )
            node_url = None
        elif not isinstance(node_url, str) or not node_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                f"remote prefix routing node {node_id!r} requires an HTTP(S) URL"
            )

        event_endpoint = node.get("event_endpoint")
        if event_endpoint is not None and (
            not isinstance(event_endpoint, str) or not event_endpoint
        ):
            raise ValueError(
                f"prefix routing node {node_id!r} event_endpoint must be a string"
            )

        replay_endpoint = node.get("replay_endpoint")
        if replay_endpoint is not None and (
            not isinstance(replay_endpoint, str) or not replay_endpoint
        ):
            raise ValueError(
                f"prefix routing node {node_id!r} replay_endpoint must be a string"
            )
        if event_endpoint is not None and replay_endpoint is None:
            raise ValueError(
                f"prefix routing node {node_id!r} with event_endpoint requires "
                "replay_endpoint"
            )
        if replay_endpoint is not None and event_endpoint is None:
            raise ValueError(
                f"prefix routing node {node_id!r} replay_endpoint requires "
                "event_endpoint"
            )

        data_parallel_rank = node.get("data_parallel_rank")
        if data_parallel_rank is not None and (
            not isinstance(data_parallel_rank, int)
            or isinstance(data_parallel_rank, bool)
            or data_parallel_rank < 0
        ):
            raise ValueError(
                f"prefix routing node {node_id!r} data_parallel_rank "
                "must be a non-negative integer"
            )

        node_routing_token = node.get("routing_token")
        if not local and (
            not isinstance(node_routing_token, str) or not node_routing_token
        ):
            raise ValueError(
                f"remote prefix routing node {node_id!r} requires a non-empty "
                "routing_token"
            )
        if local:
            node_routing_token = None

        nodes.append(
            PrefixRoutingNode(
                node_id=node_id,
                url=node_url,
                event_endpoint=event_endpoint,
                replay_endpoint=replay_endpoint,
                data_parallel_rank=data_parallel_rank,
                routing_token=node_routing_token,
                local=local,
            )
        )
    if not nodes:
        raise ValueError("prefix routing config must include at least one node")

    cache_config = vllm_config.cache_config
    hash_block_size = raw_config.get("hash_block_size")
    if hash_block_size is None:
        hash_block_size = cache_config.hash_block_size or cache_config.block_size
    if (
        not isinstance(hash_block_size, int)
        or isinstance(hash_block_size, bool)
        or hash_block_size <= 0
    ):
        raise ValueError("prefix routing hash_block_size must be a positive integer")

    request_timeout = raw_config.get("request_timeout", 6 * 60 * 60)
    if (
        not isinstance(request_timeout, int | float)
        or isinstance(request_timeout, bool)
        or request_timeout <= 0
    ):
        raise ValueError("prefix routing request_timeout must be positive")

    event_ingest_token = raw_config.get("event_ingest_token")
    if event_ingest_token is not None and (
        not isinstance(event_ingest_token, str) or not event_ingest_token
    ):
        raise ValueError("prefix routing event_ingest_token must be a non-empty string")

    routing_token = raw_config.get("routing_token")
    if routing_token is not None and (
        not isinstance(routing_token, str) or not routing_token
    ):
        raise ValueError("prefix routing routing_token must be a non-empty string")

    event_replay_timeout = raw_config.get("event_replay_timeout", 2.0)
    if (
        not isinstance(event_replay_timeout, int | float)
        or isinstance(event_replay_timeout, bool)
        or event_replay_timeout <= 0
    ):
        raise ValueError("prefix routing event_replay_timeout must be positive")

    event_sync_interval = raw_config.get("event_sync_interval", 5.0)
    if (
        not isinstance(event_sync_interval, int | float)
        or isinstance(event_sync_interval, bool)
        or event_sync_interval <= 0
    ):
        raise ValueError("prefix routing event_sync_interval must be positive")

    view_ttl_seconds = raw_config.get(
        "view_ttl_seconds", 3 * float(event_sync_interval)
    )
    if view_ttl_seconds is not None and (
        not isinstance(view_ttl_seconds, int | float)
        or isinstance(view_ttl_seconds, bool)
        or view_ttl_seconds <= 0
    ):
        raise ValueError("prefix routing view_ttl_seconds must be positive or null")

    idempotency_cache_size = raw_config.get("idempotency_cache_size", 4096)
    if (
        not isinstance(idempotency_cache_size, int)
        or isinstance(idempotency_cache_size, bool)
        or idempotency_cache_size < 0
    ):
        raise ValueError("prefix routing idempotency_cache_size must be non-negative")

    max_request_body_size = raw_config.get(
        "max_request_body_size", DEFAULT_MAX_REQUEST_BODY_SIZE
    )
    if (
        not isinstance(max_request_body_size, int)
        or isinstance(max_request_body_size, bool)
        or max_request_body_size <= 0
    ):
        raise ValueError(
            "prefix routing max_request_body_size must be a positive integer"
        )

    max_event_ingest_body_size = raw_config.get(
        "max_event_ingest_body_size", DEFAULT_MAX_EVENT_INGEST_BODY_SIZE
    )
    if (
        not isinstance(max_event_ingest_body_size, int)
        or isinstance(max_event_ingest_body_size, bool)
        or max_event_ingest_body_size <= 0
    ):
        raise ValueError(
            "prefix routing max_event_ingest_body_size must be a positive integer"
        )

    route_validation_mode = raw_config.get(
        "route_validation_mode", ROUTE_VALIDATION_MODE_GENERATION_SCOPED
    )
    if (
        not isinstance(route_validation_mode, str)
        or route_validation_mode not in ROUTE_VALIDATION_MODES
    ):
        raise ValueError(
            "prefix routing route_validation_mode must be "
            "'native-control' or 'generation-scoped'"
        )

    emit_response_headers = raw_config.get("emit_response_headers", False)
    if not isinstance(emit_response_headers, bool):
        raise ValueError("prefix routing emit_response_headers must be a boolean")

    return PrefixRoutingConfig(
        nodes=nodes,
        hash_block_size=hash_block_size,
        event_topic=str(raw_config.get("event_topic", "")),
        request_timeout=float(request_timeout),
        event_ingest_token=event_ingest_token,
        routing_token=routing_token,
        event_replay_timeout=float(event_replay_timeout),
        event_sync_interval=float(event_sync_interval),
        view_ttl_seconds=(
            None if view_ttl_seconds is None else float(view_ttl_seconds)
        ),
        idempotency_cache_size=idempotency_cache_size,
        max_request_body_size=max_request_body_size,
        max_event_ingest_body_size=max_event_ingest_body_size,
        route_validation_mode=route_validation_mode,
        emit_response_headers=emit_response_headers,
    )


class RequestBodyTooLarge(Exception):
    pass


async def _read_limited_fastapi_body(request: FastAPIRequest, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    body_size = 0
    async for chunk in request.stream():
        body_size += len(chunk)
        if body_size > max_bytes:
            raise RequestBodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


async def _receive_body(receive: Receive, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    body_size = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        if message["type"] != "http.request":
            continue
        chunk = message.get("body", b"")
        body_size += len(chunk)
        if body_size > max_bytes:
            raise RequestBodyTooLarge
        chunks.append(chunk)
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _replay_body(body: bytes) -> Receive:
    sent = False
    wait_forever = asyncio.Event()

    async def receive() -> Message:
        nonlocal sent
        if sent:
            await wait_forever.wait()
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _routing_response_headers(
    result: str,
    decision: PrefixRouteDecision | None,
    *,
    selected_node: str = "local",
) -> list[tuple[bytes, bytes]]:
    headers = [
        (ROUTE_RESULT_HEADER.encode(), result.encode("ascii")),
        (ROUTE_SELECTED_NODE_HEADER.encode(), selected_node.encode("utf-8")),
    ]
    if decision is not None:
        headers.extend(
            [
                (
                    ROUTE_MATCHED_TOKENS_HEADER.encode(),
                    str(decision.matched_tokens).encode("ascii"),
                ),
                (
                    ROUTE_GENERATION_HEADER.encode(),
                    str(decision.view_epoch).encode("ascii"),
                ),
            ]
        )
        if decision.worker_incarnation is not None:
            headers.append(
                (
                    ROUTE_WORKER_INCARNATION_HEADER.encode(),
                    decision.worker_incarnation.encode("utf-8"),
                )
            )
    return headers


def _with_response_headers(send: Send, headers: list[tuple[bytes, bytes]]) -> Send:
    if not headers:
        return send

    async def send_with_headers(message: Message) -> None:
        if message["type"] == "http.response.start":
            message = dict(message)
            message["headers"] = [
                (key, value)
                for key, value in message.get("headers", [])
                if key.lower() not in {header[0].lower() for header in headers}
            ] + headers
        await send(message)

    return send_with_headers


def _with_data_parallel_rank(scope: Scope, rank: int | None) -> Scope:
    if rank is None:
        return scope
    routed_scope = dict(scope)
    routed_scope["headers"] = [
        (key, value)
        for key, value in scope["headers"]
        if key.lower() != DATA_PARALLEL_RANK_HEADER.encode()
    ] + [(DATA_PARALLEL_RANK_HEADER.encode(), str(rank).encode("ascii"))]
    return routed_scope


def _without_external_routing_headers(scope: Scope) -> Scope:
    routed_scope = dict(scope)
    stripped_headers = {
        PREFIX_ROUTING_BYPASS_HEADER.encode(),
        DATA_PARALLEL_RANK_HEADER.encode(),
        PREFIX_ROUTING_VALIDATION_MODE_HEADER.encode(),
        EXPECTED_NODE_ID_HEADER.encode(),
        EXPECTED_GENERATION_HEADER.encode(),
        EXPECTED_WORKER_INCARNATION_HEADER.encode(),
        FENCE_RESULT_HEADER.encode(),
        ROUTE_RESULT_HEADER.encode(),
        ROUTE_SELECTED_NODE_HEADER.encode(),
        ROUTE_MATCHED_TOKENS_HEADER.encode(),
        ROUTE_GENERATION_HEADER.encode(),
        ROUTE_WORKER_INCARNATION_HEADER.encode(),
    }
    routed_scope["headers"] = [
        (key, value)
        for key, value in scope["headers"]
        if key.lower() not in stripped_headers
    ]
    return routed_scope


def _without_bypass_header(scope: Scope) -> Scope:
    routed_scope = dict(scope)
    stripped_headers = {
        PREFIX_ROUTING_BYPASS_HEADER.encode(),
        PREFIX_ROUTING_VALIDATION_MODE_HEADER.encode(),
        EXPECTED_NODE_ID_HEADER.encode(),
        EXPECTED_GENERATION_HEADER.encode(),
        EXPECTED_WORKER_INCARNATION_HEADER.encode(),
        FENCE_RESULT_HEADER.encode(),
        ROUTE_RESULT_HEADER.encode(),
        ROUTE_SELECTED_NODE_HEADER.encode(),
        ROUTE_MATCHED_TOKENS_HEADER.encode(),
        ROUTE_GENERATION_HEADER.encode(),
        ROUTE_WORKER_INCARNATION_HEADER.encode(),
    }
    routed_scope["headers"] = [
        (key, value)
        for key, value in scope["headers"]
        if key.lower() not in stripped_headers
    ]
    return routed_scope


def _has_authenticated_bypass(headers: Headers, expected_token: str | None) -> bool:
    if expected_token is None:
        return False
    token = headers.get(PREFIX_ROUTING_BYPASS_HEADER)
    if token is None:
        return False
    return secrets.compare_digest(
        hashlib.sha256(token.encode("utf-8")).digest(),
        hashlib.sha256(expected_token.encode("utf-8")).digest(),
    )


async def _validate_generation_scoped_forwarded_request(
    scope: Scope, headers: Headers, proxy: Any
) -> bool:
    mode = headers.get(PREFIX_ROUTING_VALIDATION_MODE_HEADER)
    if mode is None or mode == ROUTE_VALIDATION_MODE_NATIVE_CONTROL:
        return True
    if mode != ROUTE_VALIDATION_MODE_GENERATION_SCOPED:
        return False

    node_id = headers.get(EXPECTED_NODE_ID_HEADER)
    generation = headers.get(EXPECTED_GENERATION_HEADER)
    worker_incarnation = headers.get(EXPECTED_WORKER_INCARNATION_HEADER)
    if not node_id or generation is None or not worker_incarnation:
        return False
    try:
        view_epoch = int(generation)
    except ValueError:
        return False
    data_parallel_rank_header = headers.get(DATA_PARALLEL_RANK_HEADER)
    data_parallel_rank: int | None
    if data_parallel_rank_header is None:
        data_parallel_rank = None
    else:
        try:
            data_parallel_rank = int(data_parallel_rank_header)
        except ValueError:
            return False

    decision = PrefixRouteDecision(
        node_id=node_id,
        matched_tokens=0,
        data_parallel_rank=data_parallel_rank,
        view_epoch=view_epoch,
        worker_incarnation=worker_incarnation,
    )

    if not _matches_configured_local_node(proxy, node_id, data_parallel_rank):
        return False

    validator = getattr(scope["app"].state, "prefix_routing_route_validator", None)
    if validator is not None:
        return bool(validator(decision))

    try:
        route_fence_state = await _get_engine_route_fence_state(scope)
    except Exception:
        logger.warning(
            "Prefix routing generation fence state lookup failed",
            exc_info=True,
        )
        return False
    if route_fence_state is not None:
        return _route_fence_state_matches(
            route_fence_state,
            view_epoch=view_epoch,
            worker_incarnation=worker_incarnation,
            data_parallel_rank=data_parallel_rank,
        )

    scheduler = getattr(proxy, "scheduler", None)
    is_decision_current = getattr(scheduler, "is_decision_current", None)
    if is_decision_current is None:
        return False
    return bool(is_decision_current(decision))


async def _get_engine_route_fence_state(scope: Scope) -> Mapping[str, Any] | None:
    engine_client = getattr(scope["app"].state, "engine_client", None)
    if engine_client is None:
        return None
    get_state = getattr(engine_client, "get_prefix_cache_route_fence_state", None)
    if get_state is None:
        return None
    state = get_state()
    if inspect.isawaitable(state):
        state = await state
    if state is None:
        return None
    if not isinstance(state, Mapping):
        return None
    return state


def _matches_configured_local_node(
    proxy: Any, node_id: str, data_parallel_rank: int | None
) -> bool:
    nodes = getattr(proxy, "nodes", None)
    if not isinstance(nodes, Mapping):
        return True
    node = nodes.get(node_id)
    if node is None or not getattr(node, "local", False):
        return False
    configured_rank = getattr(node, "data_parallel_rank", None)
    return configured_rank is None or configured_rank == data_parallel_rank


def _route_fence_state_matches(
    state: Mapping[str, Any],
    *,
    view_epoch: int,
    worker_incarnation: str,
    data_parallel_rank: int | None,
) -> bool:
    try:
        current_generation = int(state["cache_generation"])
    except (KeyError, TypeError, ValueError):
        return False
    current_incarnation = state.get("worker_incarnation")
    if not isinstance(current_incarnation, str) or not current_incarnation:
        return False

    current_rank = state.get("data_parallel_rank")
    if current_rank is not None:
        try:
            current_rank = int(current_rank)
        except (TypeError, ValueError):
            return False
    if (
        data_parallel_rank is not None
        and current_rank is not None
        and data_parallel_rank != current_rank
    ):
        return False

    return (
        current_generation == view_epoch
        and current_incarnation == worker_incarnation
    )


def _extract_prefix_routing_request_id(
    headers: Headers, payload: Mapping[str, Any]
) -> str | None:
    for header_name in REQUEST_ID_HEADERS:
        request_id = headers.get(header_name)
        if request_id:
            return request_id
    payload_request_id = payload.get("request_id")
    if isinstance(payload_request_id, str) and payload_request_id:
        return payload_request_id
    return None


async def _choose_node_for_request_id(
    proxy: Any,
    path: str,
    payload: Mapping[str, Any],
    request_id: str | None,
) -> PrefixRouteDecision | None:
    choose = getattr(proxy, "choose_node_for_request_id", None)
    if choose is not None:
        return await choose(path, payload, request_id)
    return await proxy.choose_node_for_request(path, payload)


def _record_route_decision(
    proxy: Any, request_id: str | None, decision: PrefixRouteDecision | None
) -> None:
    record = getattr(proxy, "record_route_decision", None)
    if record is not None:
        record(request_id, decision)


async def _forward_request(
    scope: Scope,
    body: bytes,
    send: Send,
    node: PrefixRoutingNode,
    session: aiohttp.ClientSession,
    data_parallel_rank: int | None = None,
    *,
    decision: PrefixRouteDecision | None = None,
    route_validation_mode: str = ROUTE_VALIDATION_MODE_GENERATION_SCOPED,
    route_response_headers: list[tuple[bytes, bytes]] | None = None,
) -> bool:
    if node.url is None or node.routing_token is None:
        raise RuntimeError("remote prefix routing node is missing forwarding config")
    if route_validation_mode not in ROUTE_VALIDATION_MODES:
        raise ValueError(
            f"unknown prefix route validation mode {route_validation_mode!r}"
        )
    url = f"{node.url.rstrip('/')}{scope['path']}"
    if scope.get("query_string"):
        url += "?" + scope["query_string"].decode("latin-1")

    headers = [
        (key.decode("latin-1"), value.decode("latin-1"))
        for key, value in scope["headers"]
        if key.lower() not in (b"host", b"content-length")
    ]
    stripped_headers = {
        PREFIX_ROUTING_BYPASS_HEADER,
        DATA_PARALLEL_RANK_HEADER,
        PREFIX_ROUTING_VALIDATION_MODE_HEADER,
        EXPECTED_NODE_ID_HEADER,
        EXPECTED_GENERATION_HEADER,
        EXPECTED_WORKER_INCARNATION_HEADER,
        FENCE_RESULT_HEADER,
        ROUTE_RESULT_HEADER,
        ROUTE_SELECTED_NODE_HEADER,
        ROUTE_MATCHED_TOKENS_HEADER,
        ROUTE_GENERATION_HEADER,
        ROUTE_WORKER_INCARNATION_HEADER,
    }
    headers = [
        (key, value)
        for key, value in headers
        if key.lower() not in stripped_headers
    ]
    headers.append((PREFIX_ROUTING_BYPASS_HEADER, node.routing_token))
    if data_parallel_rank is not None:
        headers.append((DATA_PARALLEL_RANK_HEADER, str(data_parallel_rank)))
    decision_view_epoch = getattr(decision, "view_epoch", None)
    if (
        route_validation_mode == ROUTE_VALIDATION_MODE_GENERATION_SCOPED
        and decision is not None
        and decision_view_epoch is not None
    ):
        headers.extend(
            [
                (
                    PREFIX_ROUTING_VALIDATION_MODE_HEADER,
                    ROUTE_VALIDATION_MODE_GENERATION_SCOPED,
                ),
                (EXPECTED_NODE_ID_HEADER, decision.node_id),
                (EXPECTED_GENERATION_HEADER, str(decision_view_epoch)),
            ]
        )
        worker_incarnation = getattr(decision, "worker_incarnation", None)
        if worker_incarnation is not None:
            headers.append(
                (
                    EXPECTED_WORKER_INCARNATION_HEADER,
                    worker_incarnation,
                )
            )

    response_started = False
    try:
        async with session.request(
            method=scope["method"],
            url=url,
            data=body,
            headers=headers,
            allow_redirects=False,
        ) as response:
            if (
                response.status == 409
                and response.headers.get(FENCE_RESULT_HEADER) == FENCE_RESULT_REJECTED
            ):
                await response.read()
                return False
            upstream_response_headers = [
                (key.encode("latin-1"), value.encode("latin-1"))
                for key, value in response.headers.items()
                if key.lower()
                not in ("transfer-encoding", "content-encoding", "content-length")
            ]
            if route_response_headers:
                upstream_response_headers.extend(route_response_headers)
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status,
                    "headers": upstream_response_headers,
                }
            )
            response_started = True
            async for chunk in response.content.iter_chunked(1024):
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b""})
            return True
    except (aiohttp.ClientError, TimeoutError):
        if response_started:
            raise
        return False
