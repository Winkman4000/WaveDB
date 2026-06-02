"""Dedup-lookup throughput as the pool grows. The pool answers 'have I seen this value?'
once per incoming write; we want a structure whose lookup stays FLAT as it fills.
Candidates: python dict (hash, O(1)), sorted-array+bisect (O(log n), sorted-free),
trie (O(keylen), prefix-exploiting). Workloads: random strings, shared-prefix strings.
Pure measurement."""
import sys, time, bisect, random
sys.path.insert(0, '/home/jack/WaveDB/src')

random.seed(0)

def gen_random(n):
    return [f'value_{random.getrandbits(40):011x}_{i}' for i in range(n)]

def gen_prefix(n):
    # shared-prefix like paths/URLs: deep common prefixes, only the tail varies
    bases = [f'/data/warehouse/region_{r}/table_{t}/part' for r in range(8) for t in range(8)]
    return [f'{random.choice(bases)}/file_{random.getrandbits(24):06x}.parquet' for _ in range(n)]

# ---- structures: each supports add(v)->id and contains-lookup ----
class HashPool:
    def __init__(self): self.d = {}
    def get_or_add(self, v):
        i = self.d.get(v)
        if i is None: i = len(self.d); self.d[v] = i
        return i

class SortedPool:
    def __init__(self): self.keys = []; self.ids = {}
    def get_or_add(self, v):
        i = self.ids.get(v)            # membership via hash, but we maintain sorted order too
        if i is None:
            i = len(self.ids); self.ids[v] = i
            bisect.insort(self.keys, v)  # O(n) insert -- the cost of staying sorted
        return i

class TriePool:
    def __init__(self): self.root = {}; self.n = 0
    def get_or_add(self, v):
        node = self.root
        for ch in v:
            node = node.setdefault(ch, {})
        if '$' not in node: node['$'] = self.n; self.n += 1
        return node['$']

def bench_fill(struct_factory, values):
    """time inserting all distinct values (miss-heavy: every lookup is a new value)."""
    s = struct_factory()
    t = time.perf_counter()
    for v in values: s.get_or_add(v)
    return (time.perf_counter() - t)

def bench_hits(struct_factory, values, n_probe=200_000):
    """warm the pool, then time lookups that mostly HIT (steady state)."""
    s = struct_factory()
    for v in values: s.get_or_add(v)
    probes = [random.choice(values) for _ in range(n_probe)]
    t = time.perf_counter()
    for v in probes: s.get_or_add(v)
    return (time.perf_counter() - t), n_probe

structs = [('hash', HashPool), ('sorted', SortedPool), ('trie', TriePool)]

for wl_name, gen in [('random', gen_random), ('prefix', gen_prefix)]:
    print(f"\n###### workload = {wl_name} ######")
    print("--- FILL (miss-heavy: all distinct inserts), ns per op ---")
    print(f"  {'pool_size':>10} " + " ".join(f"{n:>10}" for n,_ in structs))
    for size in (10_000, 100_000, 1_000_000):
        vals = list(dict.fromkeys(gen(size)))[:size]   # ensure distinct
        row = []
        for nm, fac in structs:
            if nm == 'sorted' and size > 100_000:
                row.append('   skip(O(n))'); continue   # O(n) insert is hopeless at 1M, skip
            secs = bench_fill(fac, vals)
            row.append(f"{secs/len(vals)*1e9:9.0f}")
        print(f"  {size:>10} " + " ".join(f"{r:>10}" for r in row))
    print("--- HITS (steady-state, pool warm at 1M), ns per op ---")
    vals = list(dict.fromkeys(gen(1_000_000)))[:1_000_000]
    print(f"  {'pool_size':>10} " + " ".join(f"{n:>10}" for n,_ in structs))
    row = []
    for nm, fac in structs:
        secs, np_ = bench_hits(fac, vals)
        row.append(f"{secs/np_*1e9:9.0f}")
    print(f"  {'1,000,000':>10} " + " ".join(f"{r:>10}" for r in row))
