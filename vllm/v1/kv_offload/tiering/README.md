# KV Cache Tiering Offload and Lifecycle Management

本文档说明 `vllm/v1/kv_offload/tiering` 目录中的 KV 分层卸载框架和会话级生命周期管理，并记录启动 `vllm-hust serve` 时需要配置的参数。

## 1. KV 分层卸载框架

### 1.1 目标

KV Cache 分层卸载的目标是将部分 KV block 从设备侧 HBM 转移到更大但更慢的存储层级中，从而降低 HBM 压力，支撑更长上下文或更高并发请求。

当前实现采用两级或多级结构：

- **HBM KV cache**：vLLM 原生设备侧 KV cache，由调度器和模型执行路径直接使用。
- **Primary tier**：CPU/DDR 中的一级卸载缓存，由 `CPUPrimaryTierOffloadingManager` 管理。
- **Secondary tiers**：更低层级的二级介质，例如文件系统、对象存储或 P2P 远端节点，由 `SecondaryTierManager` 子类管理。

核心入口：

- `spec.py`：`TieringOffloadingSpec`，解析配置并创建 manager、worker、primary tier 和 secondary tiers。
- `manager.py`：`TieringOffloadingManager`，负责 primary/secondary tier 之间的状态机和数据流编排。
- `base.py`：secondary tier 的抽象接口。
- `factory.py`：secondary tier 注册和构造。
- `fs/manager.py`：基于文件系统的 secondary tier。
- `lifecycle.py`：会话级 idle 生命周期元数据管理。

### 1.2 数据流

#### Store 路径：HBM -> CPU primary -> secondary tier

当请求产生新的可卸载 KV block 时：

1. `TieringOffloadingManager.prepare_store()` 在 CPU primary tier 中为 block 分配空间。
2. worker 将 KV 从 HBM 写入 CPU primary tier。
3. `TieringOffloadingManager.complete_store()` 标记 primary tier 写入完成。
4. manager 对每个 secondary tier 调用 `submit_store()`。
5. secondary tier 异步从 CPU primary tier 读取 block，并写入自身介质。

对 `fs` tier 来说，block 会写成 `.bin` 文件，路径由 block hash、rank 和 group id 决定。

#### Load/Promotion 路径：secondary tier -> CPU primary -> HBM

当请求需要读取某个 block 时：

1. `lookup()` 先查 CPU primary tier。
2. 如果 primary tier MISS，则依次查询 secondary tiers。
3. 如果某个 secondary tier HIT，则 manager 在 primary tier 中预留空间，并返回 `RETRY`。
4. `on_schedule_end()` 批量提交 promotion 任务。
5. secondary tier 异步将 block 读回 CPU primary tier。
6. 后续 `lookup()` 命中 primary tier，worker 再从 CPU primary tier 加载到 HBM 使用。

### 1.3 调度策略

当前分层卸载遵循以下策略：

- **CPU primary tier 是所有 secondary tier 的网关**。secondary tier 不直接访问 HBM。
- **store 时级联写入 secondary tiers**。新 block 先进入 CPU primary tier，再异步写入所有配置的 secondary tiers。
- **load 时按需 promotion**。只有 primary MISS 且 secondary HIT 时，才从 secondary tier 提升回 primary tier。
- **primary tier 内部使用缓存替换策略**，当前支持 `lru` 和 `arc`。
- **secondary tier 通常保存持久副本**，例如 `fs` tier 中的 `.bin` 文件。

因此，配置了 `"kv_connector": "OffloadingConnector"` 并不意味着每个请求的所有 KV 都会立即被移动到 SSD。实际行为取决于：

- 请求是否产生可卸载的完整 block；
- primary tier 容量；
- primary tier 的替换策略；
- secondary tier 是否配置；
- 调度器是否认为这些 block 需要 store/load；
- prefix cache、chunked prefill 等 vLLM 调度行为。

### 1.4 FileSystem secondary tier

`fs` tier 是当前最容易验证的二级层。

配置项：

- `type`: 固定为 `"fs"`。
- `root_dir`: block 文件根目录。
- `n_read_threads`: 读优先 I/O 线程数。
- `n_write_threads`: 写优先 I/O 线程数。

验证方式：

```bash
find /tmp/vllm_kv_tiering_lifecycle_verify -type f | head
du -sh /tmp/vllm_kv_tiering_lifecycle_verify
```

如果长上下文请求后出现大量 `.bin` 文件，并且目录大小增长，说明 FS secondary tier 已经存储了 KV block。

## 2. 会话级生命周期管理

### 2.1 目标

会话级生命周期管理用于记录一个请求或多轮对话对应的 KV 状态何时处于 active、idle retained 或 expired 状态。

该机制的第一阶段实现是**元数据级生命周期管理**：

- 请求开始时注册 active session。
- 请求完成后将 session 标记为 idle retained。
- idle 超过 TTL 后标记过期并移除生命周期元数据。
- 可选删除 secondary tier 中对应的 FS block 文件。

注意：当前阶段不会主动改变 HBM 中 vLLM 原生 KV cache 的调度策略，也不会强制把所有 idle KV 从 HBM 移到 SSD。真正的 HBM 主动回收和会话恢复策略应在后续阶段结合调度器、block ownership 和引用计数继续实现。

### 2.2 状态机

生命周期状态定义在 `lifecycle.py`：

- `ACTIVE`：请求正在运行或该会话有活跃请求。
- `IDLE_RETAINED`：请求已经结束，KV 元数据被保留，等待复用或 TTL 过期。
- `EXPIRED`：idle 时间超过 TTL。
- `DELETED`：生命周期元数据被删除。若显式开启删除，也会尝试删除 FS secondary tier 文件。

状态转换：

```text
on_new_request
    -> ACTIVE

on_request_finished
    -> IDLE_RETAINED

expire_idle_sessions after lifecycle_idle_ttl_sec
    -> EXPIRED -> DELETED
```

### 2.3 Session id

生命周期管理通过 `ReqContext.kv_transfer_params` 识别同一个会话。

按优先级读取以下字段：

1. `session_id`
2. `conversation_id`
3. `kv_session_id`
4. 如果都没有，则回退到 `req_id`

这意味着如果客户端不传 session id，每个请求会被视为独立 session。多轮对话实验中，建议在请求的 `kv_transfer_params` 中传入稳定的 `session_id`。

### 2.4 TTL 和删除策略

配置项：

- `lifecycle_idle_ttl_sec`
  - 默认 `0.0`
  - `0` 或负数表示关闭 TTL 过期。
  - 正数表示 session 进入 idle 后，超过该秒数会过期。

- `lifecycle_delete_expired_secondary`
  - 默认 `false`
  - `false`：只删除 lifecycle 元数据，不删除 secondary tier 文件。
  - `true`：过期时尝试删除支持 `file_mapper.get_file_name()` 的 secondary tier 文件，例如 `fs` tier。

默认不删除 secondary 文件的原因是：不同 session 可能共享相同 hash block。没有引用计数时，主动删除 block 文件可能破坏其他 session 的复用。

### 2.5 在线验证

启动时将 TTL 设置为较小值，例如：

```json
"lifecycle_idle_ttl_sec": 5,
"lifecycle_delete_expired_secondary": false
```

运行一次长请求后等待 5 到 10 秒，再发一个短请求触发调度循环。服务端日志中出现如下信息即说明 lifecycle TTL 生效：

```text
Expired 1 idle KV lifecycle session(s)
```

如果 `lifecycle_delete_expired_secondary=false`，出现该日志只表示生命周期元数据过期，不表示 FS 文件被删除。

## 3. vLLM server 启动配置

### 3.1 完整示例

```bash
rm -rf /tmp/vllm_kv_tiering_lifecycle_verify
mkdir -p /tmp/vllm_kv_tiering_lifecycle_verify

ASCEND_RT_VISIBLE_DEVICES=1 \
VLLM_ASCEND_TORCH_PREFLIGHT=0 \
VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP=1 \
vllm-hust serve /root/models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8081 \
  --tensor-parallel-size 1 \
  --generation-config vllm \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 1073741824,
      "block_size": 128,
      "eviction_policy": "lru",
      "spec_name": "TieringOffloadingSpec",
      "lifecycle_idle_ttl_sec": 5,
      "lifecycle_delete_expired_secondary": false,
      "secondary_tiers": [
        {
          "type": "fs",
          "root_dir": "/tmp/vllm_kv_tiering_lifecycle_verify",
          "n_read_threads": 4,
          "n_write_threads": 4
        }
      ]
    }
  }'
```

### 3.2 顶层参数

这些参数位于 `--kv-transfer-config` 的顶层。

| 参数 | 示例 | 含义 |
| --- | --- | --- |
| `kv_connector` | `"OffloadingConnector"` | 启用 vLLM KV transfer 的 offloading connector。 |
| `kv_role` | `"kv_both"` | 当前进程既可以 store KV，也可以 load KV。在线单实例实验通常使用该值。 |
| `kv_connector_extra_config` | `{...}` | 传给具体 offloading spec 的扩展配置。 |

### 3.3 `kv_connector_extra_config`

这些参数由 `TieringOffloadingSpec` 和 tiering manager 使用。

| 参数 | 是否必需 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `spec_name` | 必需 | 无 | 指定 offloading spec。使用分层卸载时应为 `"TieringOffloadingSpec"`。 |
| `cpu_bytes_to_use` | 必需 | 无 | CPU primary tier 可使用的内存字节数。决定 DDR 一级卸载容量。 |
| `block_size` | 可选 | vLLM GPU block size | 卸载 block size。启用 prefix cache 或 chunked prefill 时，日志中可能会强制对齐到 128。 |
| `eviction_policy` | 可选 | `"lru"` | CPU primary tier 替换策略。当前支持 `"lru"` 和 `"arc"`。 |
| `secondary_tiers` | 可选 | `[]` | secondary tier 配置列表。可以配置 `fs`、`p2p`、`obj` 等。 |
| `lifecycle_idle_ttl_sec` | 可选 | `0.0` | idle session TTL。`0` 表示关闭过期。 |
| `lifecycle_delete_expired_secondary` | 可选 | `false` | TTL 过期时是否尝试删除 secondary tier 文件。默认关闭。 |

### 3.4 FS secondary tier 参数

这些参数位于 `secondary_tiers` 的单个元素中。

| 参数 | 是否必需 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `type` | 必需 | 无 | secondary tier 类型。FS 层使用 `"fs"`。 |
| `root_dir` | 必需 | 无 | KV block 文件根目录。 |
| `n_read_threads` | 可选 | `16` | 读优先 I/O 线程数，用于 secondary -> primary promotion。 |
| `n_write_threads` | 可选 | `16` | 写优先 I/O 线程数，用于 primary -> secondary store。 |

### 3.5 昇腾相关环境变量

| 环境变量 | 示例 | 含义 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | `1` | 选择可见 NPU。设置为 `1` 后，进程内通常使用 `npu:0` 访问该可见设备。 |
| `VLLM_ASCEND_TORCH_PREFLIGHT` | `0` | 关闭启动前 torch_npu preflight。某些环境下 preflight 子进程会超时，可临时关闭。 |
| `VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP` | `1` | 昇腾环境中常用的采样 custom op 兼容性开关。若日志提示未知变量但服务正常启动，可忽略。 |

## 4. 功能验证命令

### 4.1 长上下文触发 FS offload

```bash
python3 my_tests/test_offload.py \
  --url http://127.0.0.1:8081/v1/chat/completions \
  --fs-root /tmp/vllm_kv_tiering_lifecycle_verify \
  --max-context-chars 60000 \
  --repeat 1 \
  --max-tokens 64
```

期望现象：

- 客户端输出 `PASS: FS secondary tier appears to have stored KV block data.`
- `FS tier data files` 大于 0。
- `FS tier data bytes` 大于 0。
- 服务端日志出现 `KV Transfer metrics`，例如 `vllm:kv_offload_store_bytes=...`。

### 4.2 触发 lifecycle TTL

长请求结束后等待 TTL 时间，再发一个短请求：

```bash
python3 - <<'PY'
import json
import urllib.request

url = "http://127.0.0.1:8081/v1/chat/completions"
payload = {
    "model": "/root/models/Qwen2.5-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "一句话介绍 KV cache。"
        }
    ],
    "temperature": 0,
    "max_tokens": 16,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode("utf-8"))
PY
```

期望服务端日志：

```text
Expired 1 idle KV lifecycle session(s)
```

## 5. 性能观察建议

KV 卸载会引入 DDR/SSD I/O，因此单请求 latency 通常会上升。该机制的收益不应只用单请求 latency 判断，而应重点观察长上下文和高并发场景下的系统吞吐与可服务能力。

建议对比：

1. 不启用 `--kv-transfer-config` 的 baseline。
2. 启用 `OffloadingConnector + TieringOffloadingSpec`，只配置 CPU primary。
3. 启用 `OffloadingConnector + TieringOffloadingSpec + fs secondary_tier`。

关注指标：

- 请求成功率，是否避免 OOM。
- 总 tokens/s。
- 服务端日志中的 `Avg prompt throughput` 和 `Avg generation throughput`。
- `GPU KV cache usage`。
- `KV Transfer metrics`，尤其是 store/load bytes 和耗时。
- 相同并发下的 p50/p95 latency。
- 更高并发下 baseline 是否排队严重或失败，而 offload 配置仍能完成请求。

阶段 1 lifecycle 的主要验证点是状态管理和 TTL 生效。若要得到更直接的 throughput 提升，还需要后续实现基于生命周期的 HBM 主动回收、session 引用计数和恢复调度策略。
