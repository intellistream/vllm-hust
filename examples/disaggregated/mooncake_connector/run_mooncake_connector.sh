#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# Process-owned, evidence-preserving Mooncake direct-connector runner.
# Every invocation runs exactly one mutually exclusive migration mode:
# legacy, typed, or rollback. A successful strict preflight record is required.

MODE=${MODE:-}
MODEL=${MODEL:-}
PYTHON_BIN=${PYTHON_BIN:-}
PREFLIGHT_RECORD=${PREFLIGHT_RECORD:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-18000}
PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1}
PREFILL_PORTS=${PREFILL_PORTS:-18010}
BOOTSTRAP_PORTS=${BOOTSTRAP_PORTS:-18998}
DECODE_PORTS=${DECODE_PORTS:-18020}
BENCHMARK_SEED=${BENCHMARK_SEED:-42}
BENCHMARK_INPUT_LEN=${BENCHMARK_INPUT_LEN:-128}
BENCHMARK_OUTPUT_LEN=${BENCHMARK_OUTPUT_LEN:-32}
BENCHMARK_NUM_PROMPTS=${BENCHMARK_NUM_PROMPTS:-8}
BENCHMARK_REQUEST_RATE=${BENCHMARK_REQUEST_RATE:-2}
MIN_FREE_GPU_MEMORY_MIB=${MIN_FREE_GPU_MEMORY_MIB:-20000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MANIFEST="$PROJECT_ROOT/vllm/plugins/builtin_kv_bundles/mooncake.bundle.json"
PIDS=()

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

start_child() {
    local log_file=$1
    shift
    # Each retained PID is also the process-group ID for exactly one service.
    setsid "$@" > "$OUTPUT_DIR/$log_file" 2>&1 &
    PIDS+=("$!")
}

cleanup() {
    local status=${1:-0}
    trap - EXIT INT TERM USR1
    for process_group in "${PIDS[@]}"; do
        kill -TERM -- "-${process_group}" 2>/dev/null || true
    done
    for _ in {1..20}; do
        local running=false
        for process_group in "${PIDS[@]}"; do
            if kill -0 -- "-${process_group}" 2>/dev/null; then
                running=true
                break
            fi
        done
        [[ "$running" == false ]] && break
        sleep 0.5
    done
    for process_group in "${PIDS[@]}"; do
        if kill -0 -- "-${process_group}" 2>/dev/null; then
            kill -KILL -- "-${process_group}" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}

wait_for_server() {
    local port=$1
    local started
    started=$(date +%s)
    while ! curl --fail --silent --show-error "http://127.0.0.1:${port}/health" \
        > /dev/null 2>&1; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "A retained service process exited before port $port became ready" >&2
                return 1
            fi
        done
        if (( $(date +%s) - started >= TIMEOUT_SECONDS )); then
            echo "Timed out waiting for server on port $port" >&2
            return 1
        fi
        sleep 1
    done
}

validate_configuration() {
    case "$MODE" in
        legacy|typed|rollback) ;;
        *) fail "MODE must be exactly one of legacy, typed, or rollback" ;;
    esac
    [[ -n "$MODEL" ]] || fail "MODEL is required"
    [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] \
        || fail "PYTHON_BIN must name an executable controlled interpreter"
    [[ -n "$PREFLIGHT_RECORD" && -f "$PREFLIGHT_RECORD" ]] \
        || fail "PREFLIGHT_RECORD must name a strict preflight JSON record"
    [[ -n "$OUTPUT_DIR" ]] || fail "OUTPUT_DIR is required"
    [[ ! -e "$OUTPUT_DIR" ]] || fail "OUTPUT_DIR must not already exist"
    [[ -f "$MANIFEST" ]] || fail "Mooncake bundle manifest is missing"
    [[ -f "$SCRIPT_DIR/mooncake_connector_proxy.py" ]] \
        || fail "Mooncake proxy is missing"

    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra BOOTSTRAP_PORT_ARRAY <<< "$BOOTSTRAP_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    [[ ${#PREFILL_GPU_ARRAY[@]} -eq ${#PREFILL_PORT_ARRAY[@]} ]] \
        || fail "PREFILL_GPUS and PREFILL_PORTS must have equal lengths"
    [[ ${#PREFILL_GPU_ARRAY[@]} -eq ${#BOOTSTRAP_PORT_ARRAY[@]} ]] \
        || fail "PREFILL_GPUS and BOOTSTRAP_PORTS must have equal lengths"
    [[ ${#DECODE_GPU_ARRAY[@]} -eq ${#DECODE_PORT_ARRAY[@]} ]] \
        || fail "DECODE_GPUS and DECODE_PORTS must have equal lengths"

    local revision
    revision=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
    "$PYTHON_BIN" - "$PREFLIGHT_RECORD" "$PROJECT_ROOT" "$MODEL" "$revision" \
        "$PROXY_PORT" "$PREFILL_PORTS" "$BOOTSTRAP_PORTS" "$DECODE_PORTS" \
        "$PREFILL_GPUS" "$DECODE_GPUS" "$MIN_FREE_GPU_MEMORY_MIB" <<'PY'
import json
import pathlib
import socket
import subprocess
import sys

(
    record_path,
    root,
    model,
    revision,
    proxy,
    prefill,
    bootstrap,
    decode,
    prefill_gpus,
    decode_gpus,
    minimum_free_memory_mib,
) = sys.argv[1:]
record = json.loads(pathlib.Path(record_path).read_text(encoding="utf-8"))
checks = {item["name"]: item for item in record.get("checks", [])}
minimum_free_memory_mib = int(minimum_free_memory_mib)
required_ports = {int(proxy)}
for values in (prefill, bootstrap, decode):
    required_ports.update(int(value) for value in values.split(","))
required_gpus = set(prefill_gpus.split(",")) | set(decode_gpus.split(","))

errors = []
if record.get("provenance_label") != "preflight-only":
    errors.append("record is not labelled preflight-only")
if not record.get("ready_for_real_online"):
    errors.append("preflight did not pass")
if record.get("launch_performed") is not False:
    errors.append("preflight record claims a launch")
if record.get("topology") != "direct":
    errors.append("runner requires direct topology")
if checks.get("revision", {}).get("evidence", {}).get("vllm") != revision:
    errors.append("preflight revision differs from current checkout")
if pathlib.Path(checks.get("project_root", {}).get("evidence", "")).resolve() != pathlib.Path(root).resolve():
    errors.append("preflight project root differs from current checkout")
if pathlib.Path(checks.get("model", {}).get("evidence", "")).resolve() != pathlib.Path(model).resolve():
    errors.append("preflight model differs from requested model")
recorded_ports = set(checks.get("ports", {}).get("evidence", {}).get("ports", []))
if not required_ports.issubset(recorded_ports):
    errors.append("runner ports were not all admitted by preflight")
accelerator = checks.get("accelerator_inventory", {}).get("evidence", {})
eligible_gpus = set(accelerator.get("eligible_device_ids", []))
if accelerator.get("minimum_free_memory_mib", 0) < minimum_free_memory_mib:
    errors.append("preflight free-memory threshold is lower than runner threshold")
if not required_gpus.issubset(eligible_gpus):
    errors.append("runner GPUs were not all admitted by preflight")

# Repeat volatile resource checks immediately before creating evidence or services.
live = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
live_free = {
    fields[0]: int(fields[1])
    for line in live.splitlines()
    if len(fields := [field.strip() for field in line.split(",")]) == 2
}
if any(live_free.get(gpu, -1) < minimum_free_memory_mib for gpu in required_gpus):
    errors.append("runner GPUs no longer satisfy the free-memory threshold")
for port in required_ports:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind(("127.0.0.1", port))
        except OSError:
            errors.append(f"runner port {port} is no longer available")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

configure_mode() {
    LEGACY_PRODUCER='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
    LEGACY_CONSUMER='{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
    local selection='{"schema_version":"1.0","composition":"single","connectors":[{"connector_id":"mooncake-direct","scheduler_component":"vllm-core.mooncake-bridges/direct-scheduler","worker_component":"vllm-core.mooncake-bridges/direct-worker","telemetry_component":"vllm-core.mooncake-bridges/direct-telemetry","scheduler_capabilities":{"supports_hma":true},"worker_capabilities":{"supports_hma":true,"requires_piecewise_for_cudagraph":false,"required_kv_cache_layout":"HND"}}]}'
    TYPED_PRODUCER="{\"kv_role\":\"kv_producer\",\"kv_connector_selection\":${selection}}"
    TYPED_CONSUMER="{\"kv_role\":\"kv_consumer\",\"kv_connector_selection\":${selection}}"

    if [[ "$MODE" == typed ]]; then
        PRODUCER_CONFIG=$TYPED_PRODUCER
        CONSUMER_CONFIG=$TYPED_CONSUMER
        export VLLM_EXTENSION_MANIFESTS="$MANIFEST"
        export VLLM_EXTENSION_BUNDLES="vllm-core.mooncake-bridges"
        export VLLM_EXTENSION_ALLOWED_PERMISSIONS="device_access,filesystem_read,ipc,network_egress,shared_memory"
    else
        PRODUCER_CONFIG=$LEGACY_PRODUCER
        CONSUMER_CONFIG=$LEGACY_CONSUMER
        unset VLLM_EXTENSION_MANIFESTS VLLM_EXTENSION_BUNDLES \
            VLLM_EXTENSION_ALLOWED_PERMISSIONS || true
    fi
}

write_provenance() {
    mkdir -p "$OUTPUT_DIR"
    chmod 700 "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$OUTPUT_DIR/run.json" <<PY
import json
import pathlib
import platform
import subprocess

payload = {
    "schema_version": "1.0",
    "provenance_label": "real-online-run",
    "mode": ${MODE@Q},
    "topology": "direct",
    "model": ${MODEL@Q},
    "project_root": ${PROJECT_ROOT@Q},
    "vllm_revision": subprocess.check_output(
        ["git", "-C", ${PROJECT_ROOT@Q}, "rev-parse", "HEAD"], text=True
    ).strip(),
    "python": ${PYTHON_BIN@Q},
    "python_version": platform.python_version(),
    "preflight_record": ${PREFLIGHT_RECORD@Q},
    "ports": {
        "proxy": int(${PROXY_PORT@Q}),
        "prefill": ${PREFILL_PORTS@Q}.split(","),
        "bootstrap": ${BOOTSTRAP_PORTS@Q}.split(","),
        "decode": ${DECODE_PORTS@Q}.split(","),
    },
    "workload": {
        "seed": int(${BENCHMARK_SEED@Q}),
        "input_length": int(${BENCHMARK_INPUT_LEN@Q}),
        "output_length": int(${BENCHMARK_OUTPUT_LEN@Q}),
        "num_prompts": int(${BENCHMARK_NUM_PROMPTS@Q}),
        "request_rate": float(${BENCHMARK_REQUEST_RATE@Q}),
    },
    "resource_admission": {
        "minimum_free_gpu_memory_mib": int(${MIN_FREE_GPU_MEMORY_MIB@Q}),
        "gpu_memory_utilization": float(${GPU_MEMORY_UTILIZATION@Q}),
    },
    "typed_manifest": ${MANIFEST@Q} if ${MODE@Q} == "typed" else None,
    "rollback_invariant": (
        "fresh process and output directory; built-in connector names; "
        "typed manifest environment absent"
    ) if ${MODE@Q} == "rollback" else None,
}
pathlib.Path(${OUTPUT_DIR@Q}, "run.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    cp "$PREFLIGHT_RECORD" "$OUTPUT_DIR/preflight.json"
}

main() {
    validate_configuration
    configure_mode
    write_provenance
    trap 'cleanup $?' EXIT
    trap 'exit 130' INT
    trap 'exit 138' USR1
    trap 'exit 143' TERM

    local proxy_args=()
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local bootstrap_port=${BOOTSTRAP_PORT_ARRAY[$i]}
        start_child "prefill$((i+1)).log" env \
            VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap_port" \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL" \
            --port "$port" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --kv-transfer-config "$PRODUCER_CONFIG"
        proxy_args+=(--prefill "http://127.0.0.1:${port}" "$bootstrap_port")
    done

    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        start_child "decode$((i+1)).log" env CUDA_VISIBLE_DEVICES="$gpu_id" \
            "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL" \
            --port "$port" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --kv-transfer-config "$CONSUMER_CONFIG"
        proxy_args+=(--decode "http://127.0.0.1:${port}")
    done

    start_child "proxy.log" "$PYTHON_BIN" "$SCRIPT_DIR/mooncake_connector_proxy.py" \
        "${proxy_args[@]}" --port "$PROXY_PORT"

    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        wait_for_server "$port"
    done
    wait_for_server "$PROXY_PORT"

    set +e
    "$PYTHON_BIN" -m vllm.entrypoints.cli.main bench serve \
        --port "$PROXY_PORT" --seed "$BENCHMARK_SEED" \
        --backend vllm --model "$MODEL" --dataset-name random \
        --random-input-len "$BENCHMARK_INPUT_LEN" \
        --random-output-len "$BENCHMARK_OUTPUT_LEN" \
        --num-prompts "$BENCHMARK_NUM_PROMPTS" \
        --request-rate "$BENCHMARK_REQUEST_RATE" \
        > >(tee "$OUTPUT_DIR/benchmark.log") \
        2> >(tee "$OUTPUT_DIR/benchmark.stderr.log" >&2)
    local benchmark_status=$?
    set -e
    printf '%s\n' "$benchmark_status" > "$OUTPUT_DIR/benchmark.exit-code"
    return "$benchmark_status"
}

main "$@"
