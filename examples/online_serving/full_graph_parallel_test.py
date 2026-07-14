"""
服务端添加参数
 --additional-config '{
    "split_batch_config": {
      "enabled": true,
      "mode": "inplace_parallel",
      "num_splits": 2,
      "enable_parallel_streams": true,
      "enable_inplace_lazy_capture": true,
      "inplace_split_planner_policy": "largest_lower",
      "inplace_offset_match_policy": "exact",
      "inplace_parallel_replay_policy": "full_graph_parallel",
      "inplace_offset_capture_sizes": [8, 16, 32, 64],
      "parallel_capture_sizes": [8, 16, 32, 64]
    }
  }'
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 32, 64, 128]}'
"""
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty"
)

model = "/workspace/data/models/qwen3-0.6b"

messages_list = [
    [{"role": "user", "content": "你好，请介绍一下自己"}]
    for _ in range(80)
]

def send_chat(messages):
    """发送单条聊天请求"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=128,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

print("开始发送聊天请求...")
with ThreadPoolExecutor(max_workers=80) as executor:
    futures = [executor.submit(send_chat, msgs) for msgs in messages_list]
    results = [future.result() for future in futures]

print(f"共收到 {len(results)} 个响应")
for i, result in enumerate(results):
    print(f"第 {i + 1} 个响应:", result)