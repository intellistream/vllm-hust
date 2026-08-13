"""Thin LMCache adapter that applies formula-selected prefix truncation.

It deliberately delegates lookup, layerwise retrieve, CPU/NPU streams and
model execution to LMCache/vLLM.  The only behavioral change is replacing the
longest external hit H with a shorter aligned prefix h <= H before LoadSpec is
consumed by the native worker.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator

import lmcache_ascend  # noqa: F401  # activates Ascend LMCache integration
from lmcache.logging import init_logger
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

from kv_materialization_plugin.lmcache_formula_reuse_policy import (
    FormulaReuseConfig,
    FormulaReuseDecision,
    select_formula_reuse_tokens,
)
from kv_materialization_plugin.lmcache_layer_prefetch import (
    install_layer_prefetch,
    wait_for_prefetched_layer,
)
from kv_materialization_plugin.lmcache_lifecycle_telemetry import (
    LifecycleTelemetry,
)
from kv_materialization_plugin.store_admission_policy import (
    StoreAdmissionConfig,
    StoreAdmissionInput,
    decide_store_admission,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _formula_config(vllm_config: "VllmConfig") -> FormulaReuseConfig:
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    return FormulaReuseConfig(
        block_size=int(extra.get("formula_block_size", vllm_config.cache_config.block_size)),
        lmcache_chunk_size=int(extra.get("formula_lmcache_chunk_size", extra.get("lmcache.chunk_size", 128))),
        layer_count=int(extra.get("formula_layer_count", vllm_config.model_config.get_num_layers(vllm_config.parallel_config))),
        copy_fixed_ms=float(extra.get("formula_copy_fixed_ms", 1.058769054878092)),
        copy_per_token_ms=float(extra.get("formula_copy_per_token_ms", 0.009721907080673584)),
        prefill_fixed_ms=float(extra.get("formula_prefill_fixed_ms", 8.85512586947587)),
        prefill_per_token_ms=float(extra.get("formula_prefill_per_token_ms", 0.007009970208182733)),
        prefill_token_context_ms=float(extra.get("formula_prefill_token_context_ms", 1.2829906283732969e-06)),
        min_predicted_improvement_ms=float(extra.get("formula_min_predicted_improvement_ms", 0.0)),
    )


class FormulaLMCacheConnectorImpl(LMCacheAscendConnectorV1Impl):
    """Scheduler-side policy hook plus worker-side native LMCache behavior."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, parent: Any) -> None:
        super().__init__(vllm_config, role, parent)
        self._formula_config = _formula_config(vllm_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._formula_available_hits: dict[str, int] = {}
        self._formula_selected_hits: dict[str, int] = {}
        self._formula_decisions: dict[str, FormulaReuseDecision] = {}
        audit_path = extra.get("formula_audit_output_path")
        self._formula_audit = Path(audit_path).open("a", encoding="utf-8", buffering=1) if audit_path else None
        # `retrieve_layer()` in upstream LMCache only submits a storage future
        # when that layer is reached.  This optional, bounded wrapper submits a
        # few *CPU-side* layer reads ahead of consumption.  It intentionally
        # leaves the native NPU connector and all layer dependencies untouched:
        # it is therefore a safe ablation for isolating storage-read bubbles.
        self._formula_cpu_prefetch_window = max(
            0, int(extra.get("formula_cpu_prefetch_window", 0))
        )
        self._formula_cpu_prefetch_installed = False
        self._formula_device_prefetch_window = max(
            0, int(extra.get("formula_device_prefetch_window", 0))
        )
        self._formula_device_prefetch_installed = False
        self._formula_prefetch_audit_path = extra.get("formula_prefetch_audit_path")
        lifecycle_path = extra.get("lifecycle_audit_output_path")
        lifecycle_bytes_per_token = int(
            extra.get(
                "lifecycle_bytes_per_token",
                self._logical_kv_bytes_per_token(vllm_config),
            )
        )
        self._lifecycle_bytes_per_token = lifecycle_bytes_per_token
        self._lifecycle_telemetry = (
            LifecycleTelemetry(lifecycle_path, lifecycle_bytes_per_token)
            if lifecycle_path
            else None
        )
        self._lifecycle_wrapped_connector: Any | None = None
        self._lifecycle_request_ids: list[str] = []
        self._store_admission_enabled = bool(
            extra.get("store_admission_enabled", False)
        )
        self._store_admission_config = StoreAdmissionConfig(
            min_value_ms_per_gib=float(
                extra.get("store_admission_min_value_ms_per_gib", 800.0)
            ),
            deadline_safety_margin_ms=float(
                extra.get("store_admission_deadline_safety_margin_ms", 10.0)
            ),
            reuse_ttft_slo_ms=float(
                extra.get("store_admission_reuse_ttft_slo_ms", 400.0)
            ),
            slo_rescue_bonus_ms=float(
                extra.get("store_admission_slo_rescue_bonus_ms", 400.0)
            ),
            store_fixed_ms=float(
                extra.get("store_admission_store_fixed_ms", 50.0)
            ),
            store_ms_per_gib=float(
                extra.get("store_admission_store_ms_per_gib", 950.0)
            ),
        )
        admission_path = extra.get("store_admission_audit_output_path")
        self._store_admission_audit = (
            Path(admission_path).open("a", encoding="utf-8", buffering=1)
            if admission_path
            else None
        )

    @staticmethod
    def _logical_kv_bytes_per_token(vllm_config: "VllmConfig") -> int:
        """计算当前 TP worker 的未压缩 KV 逻辑字节数。"""
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        dtype_bytes = {
            "torch.float": 4,
            "torch.float32": 4,
            "torch.half": 2,
            "torch.float16": 2,
            "torch.bfloat16": 2,
            "torch.float8_e4m3fn": 1,
        }.get(str(model.dtype), 2)
        return (
            2
            * model.get_num_layers(parallel)
            * model.get_num_kv_heads(parallel)
            * model.get_head_size()
            * dtype_bytes
        )

    @staticmethod
    def _transfer_tokens(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        """从 LMCache 批传输参数中提取 token 区间总长度。"""
        starts = kwargs.get("starts")
        ends = kwargs.get("ends")
        if starts is None or ends is None:
            if len(args) < 2:
                return 0
            starts, ends = args[-2], args[-1]
        try:
            return sum(
                max(0, int(end) - int(start))
                for start, end in zip(starts, ends, strict=False)
            )
        except TypeError:
            return 0

    @staticmethod
    def _transfer_batch_items(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> int:
        """返回批传输中的 KV chunk 数，而不是 layer 数。"""
        starts = kwargs.get("starts")
        if starts is None and len(args) >= 2:
            starts = args[-2]
        try:
            return len(starts)
        except TypeError:
            return 0

    def _wrap_lifecycle_transfer(self, connector: Any, method_name: str) -> None:
        """在 connector 实例上包装一次原生同步批传输调用。"""
        telemetry = self._lifecycle_telemetry
        native: Callable[..., Any] = getattr(connector, method_name)
        direction = "load" if method_name == "batched_to_gpu" else "store"

        is_layerwise = (
            inspect.isgeneratorfunction(native)
            or "Layerwise" in type(connector).__name__
        )

        def measured(*args: Any, **kwargs: Any) -> Any:
            tokens = self._transfer_tokens(args, kwargs)
            fields = {
                "batch_items": self._transfer_batch_items(args, kwargs),
                "request_ids": list(self._lifecycle_request_ids),
            }
            assert telemetry is not None
            if is_layerwise:
                generator = native(*args, **kwargs)
                extra_advances = 2 if direction == "load" else 1
                return telemetry.measure_generator(
                    direction,
                    tokens,
                    generator,
                    expected_advances=int(connector.num_layers) + extra_advances,
                    fields=fields,
                )
            return telemetry.measure(
                direction,
                tokens,
                lambda: native(*args, **kwargs),
                fields=fields,
            )

        setattr(connector, method_name, measured)

    def _install_lifecycle_telemetry(self) -> None:
        """幂等安装实际 NPU connector 的 H2D/D2H 批次计时。"""
        if self._lifecycle_telemetry is None or self.lmcache_engine is None:
            return
        connector = self.lmcache_engine.gpu_connector
        if connector is self._lifecycle_wrapped_connector:
            return
        for method_name in ("batched_to_gpu", "batched_from_gpu"):
            if hasattr(connector, method_name):
                self._wrap_lifecycle_transfer(connector, method_name)
        self._lifecycle_wrapped_connector = connector

    def _record_lifecycle_step(self) -> None:
        """记录本轮真实 connector metadata 中的请求计划。"""
        telemetry = self._lifecycle_telemetry
        if telemetry is None:
            return
        metadata = self._parent._get_connector_metadata()
        requests = getattr(metadata, "requests", ())
        self._lifecycle_request_ids = [str(request.req_id) for request in requests]
        telemetry.record_step(requests)

    def _write_prefetch_audit(self, event: str, **fields: Any) -> None:
        """Low-level worker audit; avoids relying on process logger routing."""
        if not self._formula_prefetch_audit_path:
            return
        payload = {"event": event, "pid": os.getpid(), **fields}
        with Path(self._formula_prefetch_audit_path).open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(payload, sort_keys=True) + "\n")

    def _install_cpu_prefetch_wrapper(self) -> None:
        """Eagerly submit at most W layer reads before yielding the first.

        The returned futures, their order, and the consumer are unchanged.
        Thus a layer is still copied and consumed only through native
        ``retrieve_layer`` / ``wait_for_layer_load``.  This is deliberately
        narrower than device-side prefetch, which needs per-layer NPU events
        and a staging-buffer ring to preserve correctness.
        """
        if self._formula_cpu_prefetch_installed or self._formula_cpu_prefetch_window < 2:
            return
        if self.lmcache_engine is None or self.lmcache_engine.storage_manager is None:
            return

        storage_manager = self.lmcache_engine.storage_manager
        native_layerwise_get = storage_manager.layerwise_batched_get
        window = self._formula_cpu_prefetch_window

        def windowed_layerwise_get(keys, location=None) -> Generator[Any, None, None]:
            native_tasks = native_layerwise_get(keys, location=location)
            pending: list[Any] = []
            for _ in range(min(window, len(keys))):
                pending.append(next(native_tasks))
            while pending:
                task = pending.pop(0)
                try:
                    pending.append(next(native_tasks))
                except StopIteration:
                    pass
                yield task

        storage_manager.layerwise_batched_get = windowed_layerwise_get
        self._formula_cpu_prefetch_installed = True

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._install_lifecycle_telemetry()
        self._record_lifecycle_step()
        self._write_prefetch_audit(
            "start_load_enter",
            requested_window=self._formula_device_prefetch_window,
            has_engine=self.lmcache_engine is not None,
        )
        # Device prefetch subsumes CPU-future prefetch: it submits all storage
        # reads and then records a per-layer NPU event.  Keep the CPU-only
        # wrapper available as an isolated ablation, but never stack it.
        if self._formula_device_prefetch_window >= 2 and self.lmcache_engine is not None:
            engine = self.lmcache_engine
            engine._kvmat_audit_path = self._formula_prefetch_audit_path
            native_post_init = engine.post_init

            def post_init_then_rebind(*args, **post_kwargs):
                result = native_post_init(*args, **post_kwargs)
                engine._kvmat_audit_path = self._formula_prefetch_audit_path
                installed = install_layer_prefetch(
                    engine, self._formula_device_prefetch_window
                )
                self._formula_device_prefetch_installed = (
                    installed or self._formula_device_prefetch_installed
                )
                self._write_prefetch_audit(
                    "post_init_rebind",
                    installed=installed,
                    connector=type(engine.gpu_connector).__name__,
                )
                return result

            engine.post_init = post_init_then_rebind
            # `post_init` has already run in this vLLM-HUST worker path, so
            # installing a wrapper around a hypothetical future call never
            # fires.  The engine and its Ascend load stream already exist now;
            # bind the instance-local methods before upstream creates the
            # layerwise retriever for this request.
            self._formula_device_prefetch_installed = install_layer_prefetch(
                engine, self._formula_device_prefetch_window
            ) or self._formula_device_prefetch_installed
            self._write_prefetch_audit(
                "direct_install",
                installed=self._formula_device_prefetch_installed,
                connector=type(engine.gpu_connector).__name__,
            )
            logger.info(
                "KVMaterialization device prefetch W=%s installed=%s connector=%s",
                self._formula_device_prefetch_window,
                self._formula_device_prefetch_installed,
                type(engine.gpu_connector).__name__,
            )
        try:
            return super().start_load_kv(forward_context, **kwargs)
        finally:
            if (
                self._formula_device_prefetch_window >= 2
                and self.lmcache_engine is not None
                and hasattr(self.lmcache_engine, "post_init")
            ):
                # Restore the native method after this request.  The installed
                # engine/connector methods themselves remain instance-local.
                post_init = self.lmcache_engine.post_init
                if getattr(post_init, "__name__", "") == "post_init_then_rebind":
                    self.lmcache_engine.post_init = native_post_init

    def wait_for_layer_load(self, layer_name: str) -> None:
        # The native connector uses wait_stream(load_stream), which waits for
        # every queued future transfer.  The experimental path instead adds
        # only the required edge T_l -> C_l; native bookkeeping still advances
        # the generator immediately afterwards.
        if self._formula_device_prefetch_installed and self.lmcache_engine is not None:
            wait_for_prefetched_layer(self.lmcache_engine, self.current_layer)
        return super().wait_for_layer_load(layer_name)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int):
        native_external_tokens = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if native_external_tokens is None:
            return None
        spec = self.load_specs.get(request.request_id)
        if spec is None or spec.lmcache_cached_tokens <= num_computed_tokens:
            return native_external_tokens

        available = int(spec.lmcache_cached_tokens)
        decision = select_formula_reuse_tokens(
            available_tokens=available,
            local_tokens=int(num_computed_tokens),
            prompt_tokens=int(request.num_tokens),
            config=self._formula_config,
        )
        self._formula_available_hits[request.request_id] = available
        self._formula_selected_hits[request.request_id] = decision.selected_tokens
        self._formula_decisions[request.request_id] = decision
        self._write_formula_audit(request.request_id, decision)

        if decision.selected_tokens == available:
            return native_external_tokens

        # LoadSpec is the single source of truth used by both scheduler and
        # worker.  Native retrieve_layer() subsequently slices tokens and slot
        # mappings to this exact selected prefix, so no hidden tail copy remains.
        spec.lmcache_cached_tokens = decision.selected_tokens
        spec.can_load = False
        return decision.selected_tokens - num_computed_tokens

    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        super().update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)

        if self._store_admission_enabled:
            for request_meta in metadata.requests:
                save_spec = request_meta.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue
                configs = request_meta.request_configs or {}
                required = (
                    "lmcache.predicted_reuse_probability",
                    "lmcache.predicted_reuse_delay_ms",
                    "lmcache.predicted_recompute_ttft_ms",
                    "lmcache.predicted_load_ttft_ms",
                )
                if not all(name in configs for name in required):
                    # 兼容普通用户流量：没有预测字段时保持 LMCache 原生行为。
                    continue
                kv_bytes = (
                    len(request_meta.token_ids)
                    * self._lifecycle_bytes_per_token
                )
                decision = decide_store_admission(
                    StoreAdmissionInput(
                        predicted_reuse_probability=float(configs[required[0]]),
                        predicted_reuse_delay_ms=float(configs[required[1]]),
                        recompute_ttft_ms=float(configs[required[2]]),
                        load_ttft_ms=float(configs[required[3]]),
                        kv_bytes=kv_bytes,
                    ),
                    self._store_admission_config,
                )
                if not decision.admit:
                    save_spec.can_save = False
                if self._store_admission_audit is not None:
                    self._store_admission_audit.write(json.dumps({
                        "event": "store_admission",
                        "request_id": request_meta.req_id,
                        "token_count": len(request_meta.token_ids),
                        "kv_bytes": kv_bytes,
                        **asdict(decision),
                    }, sort_keys=True) + "\n")

        # LMCache already owns the longer prefix.  Do not re-store the tail that
        # this policy deliberately recomputes merely because it was not loaded.
        # Keep this state separate from LoadSpec: LoadSpec must remain selected h
        # to make worker-side retrieve exactly [0,h].  `super()` has already
        # advanced RequestTracker.num_saved_tokens to this step's input length;
        # changing it back would make later save bookkeeping inconsistent.
        for request_meta in metadata.requests:
            available = self._formula_available_hits.get(request_meta.req_id)
            selected = self._formula_selected_hits.get(request_meta.req_id)
            if available is None or selected is None or available <= selected:
                continue
            if request_meta.save_spec is not None:
                request_meta.save_spec.skip_leading_tokens = max(
                    request_meta.save_spec.skip_leading_tokens,
                    available,
                )

        for request_id in scheduler_output.finished_req_ids:
            self._formula_available_hits.pop(request_id, None)
            self._formula_selected_hits.pop(request_id, None)
            self._formula_decisions.pop(request_id, None)
        return metadata

    def shutdown(self):
        if self._lifecycle_telemetry is not None:
            self._lifecycle_telemetry.close()
        if self._formula_audit is not None:
            self._formula_audit.close()
            self._formula_audit = None
        if self._store_admission_audit is not None:
            self._store_admission_audit.close()
            self._store_admission_audit = None
        return super().shutdown()

    def _write_formula_audit(self, request_id: str, decision: FormulaReuseDecision) -> None:
        if self._formula_audit is None:
            return
        payload = {
            "request_id": request_id,
            "policy": "formula_contiguous_prefix",
            "model": asdict(self._formula_config),
            "decision": asdict(decision),
            "predicted_improvement_ms": decision.predicted_improvement_ms,
        }
        self._formula_audit.write(json.dumps(payload, sort_keys=True) + "\n")


class FormulaLMCacheConnector(LMCacheConnectorV1):
    """Externally loadable vLLM connector; all normal methods are inherited."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # Bypass LMCacheConnectorV1.__init__ only to inject our tiny impl.  Its
        # public worker/scheduler methods remain inherited unchanged.
        KVConnectorBase_V1.__init__(
            self,
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._lmcache_engine = FormulaLMCacheConnectorImpl(vllm_config, role, self)
        self._kv_cache_events = None


__all__ = ["FormulaLMCacheConnector"]
