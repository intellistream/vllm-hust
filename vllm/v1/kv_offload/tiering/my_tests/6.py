# 功能验证说明：
# - 验证 CPUOffloadingManager 层的状态机和缓存策略。
# - 覆盖 prepare_store -> complete_store -> lookup -> prepare_load -> complete_load。
# - 覆盖 LRU eviction、ARC policy、cache full、HIT_PENDING、MISS、store_threshold。
#
python - <<'PY'
from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def title(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)


def key(i):
    return make_offload_key(str(i).encode(), 0)


def keys(items):
    return [key(i) for i in items]


def block_ids(spec):
    assert isinstance(spec, CPULoadStoreSpec)
    return [int(x) for x in spec.block_ids]


ctx = ReqContext(req_id="manager-test", kv_transfer_params=None)

title("Step1. Basic State")

manager = CPUOffloadingManager(
    num_blocks=4,
    cache_policy="lru",
    enable_events=True,
    store_threshold=0,
)

assert manager.lookup(key(1), ctx) is LookupResult.MISS
assert manager.lookup(key(2), ctx) is LookupResult.MISS

print("PASS: new manager returns MISS for unknown keys.")

title("Step2. prepare_store()")

store_out = manager.prepare_store(keys([1, 2]), ctx)

assert store_out is not None
assert store_out.keys_to_store == keys([1, 2])
assert isinstance(store_out.store_spec, CPULoadStoreSpec)
assert block_ids(store_out.store_spec) == [0, 1]
assert store_out.evicted_keys == []

assert manager.lookup(key(1), ctx) is LookupResult.HIT_PENDING
assert manager.lookup(key(2), ctx) is LookupResult.HIT_PENDING

print("keys_to_store:", [1, 2])
print("store block ids:", block_ids(store_out.store_spec))
print("PASS: prepare_store creates CPU store spec and marks keys HIT_PENDING.")

title("Step3. complete_store(success=True)")

manager.complete_store(store_out.keys_to_store, ctx, success=True)

assert manager.lookup(key(1), ctx) is LookupResult.HIT
assert manager.lookup(key(2), ctx) is LookupResult.HIT

print("PASS: complete_store marks pending keys ready; lookup returns HIT.")

title("Step4. prepare_load()")

load_spec = manager.prepare_load(keys([1, 2]), ctx)

assert isinstance(load_spec, CPULoadStoreSpec)
assert block_ids(load_spec) == [0, 1]

# There is no public ref_count / eviction-protection accessor.
# The public lifecycle check is: prepare_load succeeds and lookup remains HIT.
assert manager.lookup(key(1), ctx) is LookupResult.HIT
assert manager.lookup(key(2), ctx) is LookupResult.HIT

print("load block ids:", block_ids(load_spec))
print("PASS: prepare_load returns CPULoadStoreSpec and keys remain readable.")

title("Step5. complete_load()")

manager.complete_load(keys([1, 2]), ctx)

assert manager.lookup(key(1), ctx) is LookupResult.HIT
assert manager.lookup(key(2), ctx) is LookupResult.HIT

print("PASS: complete_load ends load lifecycle and cached keys remain HIT.")

title("Step6. LRU Eviction")

lru = CPUOffloadingManager(
    num_blocks=2,
    cache_policy="lru",
    enable_events=True,
    store_threshold=0,
)

out = lru.prepare_store(keys([1, 2]), ctx)
assert out is not None
assert block_ids(out.store_spec) == [0, 1]
lru.complete_store(out.keys_to_store, ctx, success=True)

assert lru.lookup(key(1), ctx) is LookupResult.HIT
assert lru.lookup(key(2), ctx) is LookupResult.HIT

# Make key1 recent, so key2 should be LRU.
lru.touch(keys([1]), ctx)

out = lru.prepare_store(keys([3]), ctx)
assert out is not None
assert out.keys_to_store == keys([3])
assert out.evicted_keys == keys([2])

lru.complete_store(out.keys_to_store, ctx, success=True)

assert lru.lookup(key(1), ctx) is LookupResult.HIT
assert lru.lookup(key(2), ctx) is LookupResult.MISS
assert lru.lookup(key(3), ctx) is LookupResult.HIT

print("evicted key: 2")
print("PASS: LRU evicts key2 after touch(key1), final HIT/MISS state is correct.")

title("Step7. Cache Full With No Evictable Blocks")

# Cache currently contains key1 and key3.
load_spec_1 = lru.prepare_load(keys([1]), ctx)
load_spec_3 = lru.prepare_load(keys([3]), ctx)

assert isinstance(load_spec_1, CPULoadStoreSpec)
assert isinstance(load_spec_3, CPULoadStoreSpec)

# Both cached blocks are currently involved in loads, so neither is evictable.
out = lru.prepare_store(keys([4]), ctx)

assert out is None

lru.complete_load(keys([1]), ctx)
lru.complete_load(keys([3]), ctx)

assert lru.lookup(key(1), ctx) is LookupResult.HIT
assert lru.lookup(key(3), ctx) is LookupResult.HIT
assert lru.lookup(key(4), ctx) is LookupResult.MISS

print("PASS: prepare_store returns None when cache is full and no block is evictable.")

title("Step8. ARC Policy")

arc = CPUOffloadingManager(
    num_blocks=2,
    cache_policy="arc",
    enable_events=True,
    store_threshold=0,
)

assert arc.lookup(key(10), ctx) is LookupResult.MISS

out = arc.prepare_store(keys([10]), ctx)
assert out is not None
assert out.keys_to_store == keys([10])
assert isinstance(out.store_spec, CPULoadStoreSpec)

assert arc.lookup(key(10), ctx) is LookupResult.HIT_PENDING

arc.complete_store(out.keys_to_store, ctx, success=True)

assert arc.lookup(key(10), ctx) is LookupResult.HIT

load_spec = arc.prepare_load(keys([10]), ctx)
assert isinstance(load_spec, CPULoadStoreSpec)

arc.complete_load(keys([10]), ctx)

assert arc.lookup(key(10), ctx) is LookupResult.HIT

out = arc.prepare_store(keys([11]), ctx)
assert out is not None
arc.complete_store(out.keys_to_store, ctx, success=True)

assert arc.lookup(key(11), ctx) is LookupResult.HIT

# Cache is full. Storing key12 should evict one existing key.
out = arc.prepare_store(keys([12]), ctx)
assert out is not None
assert out.keys_to_store == keys([12])
assert len(out.evicted_keys) == 1

arc.complete_store(out.keys_to_store, ctx, success=True)

assert arc.lookup(key(12), ctx) is LookupResult.HIT

remaining = [
    arc.lookup(key(10), ctx),
    arc.lookup(key(11), ctx),
    arc.lookup(key(12), ctx),
]

assert remaining.count(LookupResult.HIT) == 2
assert remaining.count(LookupResult.MISS) == 1

print("evicted keys:", ["10" if k == key(10) else "11" if k == key(11) else "unknown" for k in
out.evicted_keys])
print("PASS: ARC basic lifecycle and cache-full eviction behavior are correct.")

title("Step9. store_threshold")

threshold_mgr = CPUOffloadingManager(
    num_blocks=4,
    cache_policy="lru",
    enable_events=True,
    store_threshold=2,
    max_tracker_size=8,
)

assert threshold_mgr.lookup(key(100), ctx) is LookupResult.MISS

out = threshold_mgr.prepare_store(keys([100]), ctx)

assert out is not None
assert out.keys_to_store == []
assert isinstance(out.store_spec, CPULoadStoreSpec)
assert block_ids(out.store_spec) == []

print("first prepare_store keys_to_store:", out.keys_to_store)

assert threshold_mgr.lookup(key(100), ctx) is LookupResult.MISS

out = threshold_mgr.prepare_store(keys([100]), ctx)

assert out is not None
assert out.keys_to_store == keys([100])
assert isinstance(out.store_spec, CPULoadStoreSpec)
assert block_ids(out.store_spec) == [0]

print("second prepare_store keys_to_store: [100]")
print("PASS: store_threshold=2 filters first store and allows second observed key.")

print("\n" + "=" * 70)
print("FINAL RESULT : PASS")
print("=" * 70)
PY
