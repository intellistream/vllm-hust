# vllm-b134-events

B134 KV-tiering benchmark TSV event sink, implemented as an out-of-tree vLLM
plugin.

## Why a plugin

Per review feedback, the engine core only exposes a generic, typed,
default-off event outlet (`vllm.v1.events.EventBus`). Experiment-specific
emitters must live outside core; this package is the B134 emitter.

The sink is responsible for:

- **event selection** — only B134-relevant event types are serialized;
- **serialization** — typed events are mapped to the historical B134 TSV
  format (`ts\tevent\treq_id\tk=v ...`), preserving event names and field
  semantics so existing analysis scripts keep working;
- **file/exporter lifecycle** — the output file is opened lazily and flushed
  per line;
- **bounded buffering** — a bounded queue (default 65536 events) feeds a
  single background writer thread; overflow drops events and bumps
  `dropped_events` instead of blocking the engine;
- **error degradation** — writer failures are counted and swallowed, never
  propagated to the serving path.

## Install & enable

```bash
# from the repo root
pip install -e plugins/vllm-b134-events

# enable the sink (otherwise it is a no-op)
export B134_EVENTS_FILE=/path/to/b134_events.tsv

# start vLLM normally; the plugin is loaded via the entry point
```

## Output format

Line-buffered TSV, one event per line:

```
<monotonic_seconds:6f>\t<event>\t<request_id>\t<k=v ...>
```

Event names: `admission`, `scheduled`, `preempt`, `wakeup`, `finish`,
`cpu_store`, `cpu_evict`, `restore_start`, `restore_done`, `evict`,
`sched_step`, `transfer_phase`, `gather_h2d`, `swap_d2h`, `copy_done`.
