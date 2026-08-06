"""wdb_calib -- the machine card: measured primitive speeds for this box.

A ~2s ceremony at first open per machine, cached as .calib.json beside the
database (machine-derived sidecar; re-run when the hardware changes or the
file is deleted). The router prices plans as bytes x these rates -- exact
facts about THIS machine, never folklore."""
import json
import os
import time

import numpy as np

_CACHE = None


def _bench(fn, *args, reps=3):
    best = None
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        el = time.perf_counter() - t
        best = el if best is None or el < best else best
    return best


def machine_card(db_dir):
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = os.path.join(db_dir, '.calib.json')
    if os.path.exists(path):
        try:
            _CACHE = json.load(open(path))
            return _CACHE
        except Exception:
            pass
    import zstandard as zstd
    N = 8_000_000
    rng = np.random.default_rng(0)
    kc = rng.integers(0, 6000, N).astype(np.uint16)
    uc = rng.integers(0, 1 << 20, N).astype(np.uint32)
    tab16 = rng.integers(0, 3000, 1 << 20).astype(np.uint16)
    tab64 = tab16.astype(np.int64)
    raw = uc.tobytes()
    z = zstd.ZstdCompressor(level=3).compress(raw)
    dz = zstd.ZstdDecompressor()
    card = {}
    card['zstd_pop_mbps'] = (len(raw) / 1e6) / _bench(lambda: dz.decompress(z))
    card['gather_u16_mrps'] = (N / 1e6) / _bench(lambda: tab16[uc])
    card['gather_i64_mrps'] = (N / 1e6) / _bench(lambda: tab64[uc])
    card['bincount_w_mrps'] = (N / 1e6) / _bench(
        lambda: np.bincount(kc, weights=tab64[uc].astype(np.float64), minlength=6000))
    try:
        import wdb_kernels as WK
        jars = np.zeros(6000, np.int64)
        cnts = np.zeros(6000, np.int64)
        WK.lenagg_pour(kc[:1000], uc[:1000], tab16, jars, cnts, np.int64(-1))  # jit warm
        card['fused_pour_mrps'] = (N / 1e6) / _bench(
            lambda: WK.lenagg_pour(kc, uc, tab16, jars, cnts, np.int64(-1)))
    except Exception:
        card['fused_pour_mrps'] = None
    card['threads'] = os.cpu_count() or 1
    card['measured_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        json.dump(card, open(path, 'w'), indent=1)
    except Exception:
        pass
    _CACHE = card
    return card
