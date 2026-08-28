#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# =============================================================================
# vLLM Disaggregated Serving Script for Mooncake Connector
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# Mooncake Connector.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   BOOTSTRAP_PORTS: Bootstrap server port launched by prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup P/D disaggregated connection.
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-8000}

PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1}
PREFILL_PORTS=${PREFILL_PORTS:-8010}
BOOTSTRAP_PORTS=${BOOTSTRAP_PORTS:-8998}
DECODE_PORTS=${DECODE_PORTS:-8020}

echo "Warning: Mooncake Connector support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS, Bootstrap Port:$BOOTSTRAP_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

start_child() {
    local log_file=$1
    shift
    # Each retained PID is also the process-group ID for exactly one service.
    setsid "$@" > "$log_file" 2>&1 &
    PIDS+=("$!")
}

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("mooncake_connector_proxy.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        echo "Example: export HF_TOKEN=your_token_here"
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    local status=${1:-0}
    echo "Stopping everything…"
    trap - INT TERM USR1

    # Stop only the process groups created and retained by this invocation.
    for process_group in "${PIDS[@]}"; do
        kill -TERM -- "-${process_group}" 2>/dev/null || true
    done

    # Give services a bounded graceful-shutdown window before escalation.
    for _ in {1..20}; do
        local running=false
        for process_group in "${PIDS[@]}"; do
            if kill -0 -- "-${process_group}" 2>/dev/null; then
                running=true
                break
            fi
        done
        if [[ "$running" == false ]]; then
            break
        fi
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
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_required_files
    check_hf_token
    check_num_gpus
    ensure_python_library_installed vllm
    ensure_python_library_installed mooncake.engine

    trap 'cleanup 130' INT
    trap 'cleanup 138' USR1
    trap 'cleanup 143' TERM

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"

    # Parse GPU and port arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra BOOTSTRAP_PORT_ARRAY <<< "$BOOTSTRAP_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    proxy_args=()

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local bootstrap_port=${BOOTSTRAP_PORT_ARRAY[$i]}

        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, Bootstrap Port $bootstrap_port"
        start_child "prefill$((i+1)).log" env \
            VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap_port" \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            vllm serve "$MODEL" \
        --port "$port" \
        --kv-transfer-config \
        "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\"}"
        proxy_args+=(--prefill "http://0.0.0.0:${port}" "$bootstrap_port")
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}

        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port"
        start_child "decode$((i+1)).log" env \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            vllm serve "$MODEL" \
        --port "$port" \
        --kv-transfer-config \
        "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\"}"
        proxy_args+=(--decode "http://0.0.0.0:${port}")
    done

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    start_child "proxy.log" python3 mooncake_connector_proxy.py \
        "${proxy_args[@]}" --port "$PROXY_PORT"

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$port"; then
            echo "Failed to start server on port $port"
            cleanup 1
        fi
    done

    echo ""
    echo "All servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================
    vllm bench serve --port "$PROXY_PORT" --seed "$(date +%s)" \
        --backend vllm --model "$MODEL" \
        --dataset-name random --random-input-len 7500 --random-output-len 200 \
        --num-prompts 200 --burstiness 100 --request-rate 2 | tee benchmark.log

    echo "Benchmarking done. Cleaning up..."

    cleanup 0
}

main
